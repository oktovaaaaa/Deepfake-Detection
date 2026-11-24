# %% [markdown]
# # Deteksi Berita Hoaks Bahasa Indonesia
# Baseline panjang teks, TF-IDF + LogReg/SVM, IndoBERT, IndoBERT + BiLSTM. Label: 1 = hoaks, 0 = valid.

# %% [markdown]
# ## 1. Konfigurasi

# %%
import os, sys, gc, json, math, random, time, re

IN_COLAB = "google.colab" in sys.modules
IN_NOTEBOOK = "ipykernel" in sys.modules
if IN_COLAB:
    from google.colab import drive
    drive.mount("/content/drive")
    DATA_DIR = "/content/drive/MyDrive/certan/merged"
    OUT_DIR = "/content/drive/MyDrive/certan/hasil_training"
else:
    BASE = os.path.dirname(os.path.abspath(__file__)) if "__file__" in globals() else os.getcwd()
    DATA_DIR = os.path.join(BASE, "merged")
    OUT_DIR = os.path.join(BASE, "hasil_training")
os.makedirs(OUT_DIR, exist_ok=True)

MODEL_NAME = "indobenchmark/indobert-base-p1"
MAX_LEN = 256
EPOCHS = 3
LR_BERT = 2e-5
LR_HEAD = 1e-3
WARMUP = 0.1
PATIENCE = 1
SEED = 42
SKIP_IF_TRAINED = True

# %%
import numpy as np
import pandas as pd
import matplotlib
if not IN_NOTEBOOK:
    matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.metrics import (accuracy_score, precision_recall_fscore_support, f1_score,
                             confusion_matrix, classification_report, roc_auc_score)
from transformers import AutoTokenizer, AutoModel, get_linear_schedule_with_warmup
from transformers.utils import logging as hf_logging
hf_logging.set_verbosity_error()


def show(obj):
    if IN_NOTEBOOK:
        from IPython.display import display
        display(obj)
    else:
        print(obj.to_string() if isinstance(obj, pd.DataFrame) else obj)


def show_plot(path):
    plt.tight_layout()
    plt.savefig(path, dpi=150)
    plt.show() if IN_NOTEBOOK else plt.close()


def set_seed(s):
    random.seed(s); np.random.seed(s); torch.manual_seed(s); torch.cuda.manual_seed_all(s)


set_seed(SEED)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
if device.type == "cuda":
    free_b, total_b = torch.cuda.mem_get_info(0)
    gpu_mem = free_b / 1e9
    print(f"GPU: {torch.cuda.get_device_name(0)} | kosong {gpu_mem:.1f} / {total_b / 1e9:.1f} GB")
else:
    gpu_mem = 0
    print("PERINGATAN: GPU tidak terdeteksi, training IndoBERT akan sangat lambat.")

if gpu_mem >= 14:  BATCH, ACCUM = 32, 1
elif gpu_mem >= 7: BATCH, ACCUM = 16, 2
else:              BATCH, ACCUM = 8, 4
USE_AMP = device.type == "cuda"
print(f"BATCH={BATCH} ACCUM={ACCUM} MAX_LEN={MAX_LEN} AMP={USE_AMP}")

# %% [markdown]
# ## 2. Data

# %%
train_df = pd.read_csv(f"{DATA_DIR}/train.csv")
val_df = pd.read_csv(f"{DATA_DIR}/val.csv")
test_df = pd.read_csv(f"{DATA_DIR}/test.csv")
for d in (train_df, val_df, test_df):
    d["text"] = d["text"].astype(str)

show(pd.DataFrame({n: {"jumlah": len(d), "valid": (d.label == 0).sum(), "hoaks": (d.label == 1).sum()}
                   for n, d in [("train", train_df), ("val", val_df), ("test", test_df)]}).T)

info = train_df.assign(kata=train_df.text.str.split().str.len()).groupby("source").agg(
    jumlah=("label", "size"), persen_hoaks=("label", "mean"), median_kata=("kata", "median"))
info["persen_hoaks"] = (info.persen_hoaks * 100).round(1)
show(info.sort_values("persen_hoaks"))

# %% [markdown]
# ## 3. Evaluasi

# %%
RESULTS, PER_SOURCE, HISTORIES = {}, {}, {}


def evaluate(name, y_true, y_pred, y_score=None):
    y_true, y_pred = np.asarray(y_true), np.asarray(y_pred)
    p, r, f1, _ = precision_recall_fscore_support(y_true, y_pred, average="binary", zero_division=0)
    RESULTS[name] = {"accuracy": accuracy_score(y_true, y_pred), "precision": p, "recall": r, "f1": f1,
                     "f1_macro": f1_score(y_true, y_pred, average="macro"),
                     "roc_auc": roc_auc_score(y_true, y_score) if y_score is not None else np.nan}
    print(f"===== {name} =====")
    print(classification_report(y_true, y_pred, target_names=["valid", "hoaks"], digits=4))

    cm = confusion_matrix(y_true, y_pred)
    fig, ax = plt.subplots(figsize=(4, 3.5))
    ax.imshow(cm, cmap="Blues")
    for i in range(2):
        for j in range(2):
            ax.text(j, i, f"{cm[i, j]:,}", ha="center", va="center",
                    color="white" if cm[i, j] > cm.max() / 2 else "black")
    ax.set_xticks([0, 1], ["valid", "hoaks"]); ax.set_yticks([0, 1], ["valid", "hoaks"])
    ax.set_xlabel("Prediksi"); ax.set_ylabel("Aktual"); ax.set_title(name, fontsize=10)
    show_plot(f"{OUT_DIR}/cm_{len(RESULTS)}_{re.sub(r'[^A-Za-z0-9]+', '_', name)}.png")

    tmp = test_df[["source", "label"]].copy()
    tmp["benar"] = y_pred == tmp.label.values
    ps = tmp.groupby("source").agg(n=("label", "size"), persen_hoaks=("label", "mean"), akurasi=("benar", "mean"))
    ps["persen_hoaks"] = (ps.persen_hoaks * 100).round(1)
    PER_SOURCE[name] = ps.akurasi.round(4)
    show(ps.round(4))

# %% [markdown]
# ## 4. Baseline panjang teks

# %%
def length_feature(s):
    return np.log1p(s.str.split().str.len().values).reshape(-1, 1)


len_model = LogisticRegression().fit(length_feature(train_df.text), train_df.label)
X_len = length_feature(test_df.text)
evaluate("Baseline panjang teks", test_df.label, len_model.predict(X_len), len_model.predict_proba(X_len)[:, 1])

# %% [markdown]
# ## 5. TF-IDF + Logistic Regression / SVM

# %%
tfidf = TfidfVectorizer(ngram_range=(1, 2), min_df=2, max_df=0.9, sublinear_tf=True, max_features=300_000)
X_train = tfidf.fit_transform(train_df.text)
X_val, X_test = tfidf.transform(val_df.text), tfidf.transform(test_df.text)


def tune(make_model, Cs):
    best = None
    for C in Cs:
        m = make_model(C).fit(X_train, train_df.label)
        f1 = f1_score(val_df.label, m.predict(X_val))
        print(f"C={C}: F1 val = {f1:.4f}")
        if best is None or f1 > best[0]:
            best = (f1, m)
    return best[1]


lr_model = tune(lambda C: LogisticRegression(C=C, max_iter=3000), [0.5, 2, 8])
svm_model = tune(lambda C: LinearSVC(C=C), [0.1, 0.5, 2])
evaluate("TF-IDF + LogReg", test_df.label, lr_model.predict(X_test), lr_model.predict_proba(X_test)[:, 1])
evaluate("TF-IDF + SVM", test_df.label, svm_model.predict(X_test), svm_model.decision_function(X_test))

names = np.array(tfidf.get_feature_names_out())
coef = lr_model.coef_[0]
show(pd.DataFrame({"penanda hoaks": names[np.argsort(coef)[-25:][::-1]],
                   "penanda valid": names[np.argsort(coef)[:25]]}))

# %% [markdown]
# ## 6. Dataset IndoBERT

# %%
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

sample = train_df.text.sample(min(2000, len(train_df)), random_state=SEED).tolist()
n_tok = np.array([len(x) for x in tokenizer(sample, truncation=False)["input_ids"]])
print(f"Median token {np.median(n_tok):.0f} | terpotong di MAX_LEN: {(n_tok > MAX_LEN).mean() * 100:.1f}%")


class TextDataset(Dataset):
    def __init__(self, df):
        self.texts, self.labels = df.text.tolist(), df.label.astype(int).tolist()

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, i):
        return self.texts[i], self.labels[i]


def collate(batch):
    texts, labels = zip(*batch)
    enc = tokenizer(list(texts), truncation=True, max_length=MAX_LEN, padding=True, return_tensors="pt")
    enc["labels"] = torch.tensor(labels)
    return enc


train_loader = DataLoader(TextDataset(train_df), batch_size=BATCH, shuffle=True, collate_fn=collate,
                          generator=torch.Generator().manual_seed(SEED))
val_loader = DataLoader(TextDataset(val_df), batch_size=BATCH * 2, collate_fn=collate)
test_loader = DataLoader(TextDataset(test_df), batch_size=BATCH * 2, collate_fn=collate)

# %% [markdown]
# ## 7. Model

# %%
class IndoBERTClassifier(nn.Module):
    def __init__(self, model_name, dropout=0.1):
        super().__init__()
        self.bert = AutoModel.from_pretrained(model_name)
        self.drop = nn.Dropout(dropout)
        self.head = nn.Linear(self.bert.config.hidden_size, 2)

    def forward(self, input_ids, attention_mask, token_type_ids=None):
        out = self.bert(input_ids=input_ids, attention_mask=attention_mask, token_type_ids=token_type_ids)
        return self.head(self.drop(out.last_hidden_state[:, 0]))


class IndoBERTBiLSTM(nn.Module):
    def __init__(self, model_name, lstm_hidden=256, dropout=0.3):
        super().__init__()
        self.bert = AutoModel.from_pretrained(model_name)
        self.lstm = nn.LSTM(self.bert.config.hidden_size, lstm_hidden, batch_first=True, bidirectional=True)
        self.drop = nn.Dropout(dropout)
        self.head = nn.Linear(lstm_hidden * 4, 2)

    def forward(self, input_ids, attention_mask, token_type_ids=None):
        seq = self.bert(input_ids=input_ids, attention_mask=attention_mask,
                        token_type_ids=token_type_ids).last_hidden_state
        packed = nn.utils.rnn.pack_padded_sequence(seq, attention_mask.sum(1).cpu(), batch_first=True,
                                                   enforce_sorted=False)
        out, _ = self.lstm(packed)
        out, _ = nn.utils.rnn.pad_packed_sequence(out, batch_first=True, total_length=seq.size(1))
        out = out.float()
        mask = attention_mask.unsqueeze(-1).float()
        mean_pool = (out * mask).sum(1) / mask.sum(1).clamp(min=1)
        max_pool = out.masked_fill(mask == 0, -1e4).max(1).values
        return self.head(self.drop(torch.cat([mean_pool, max_pool], dim=1)))


MODEL_CLASSES = {"indobert": IndoBERTClassifier, "indobert_bilstm": IndoBERTBiLSTM}

# %% [markdown]
# ## 8. Training

# %%
def to_device(batch):
    return {k: v.to(device) for k, v in batch.items()}


@torch.no_grad()
def predict_proba(model, loader):
    model.eval()
    probs, labels = [], []
    for b in loader:
        labels.append(b.pop("labels"))
        with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=USE_AMP):
            logits = model(**to_device(b))
        probs.append(torch.softmax(logits.float(), dim=-1)[:, 1].cpu())
    return torch.cat(labels).numpy(), torch.cat(probs).numpy()


def train_model(key):
    ckpt, hist_path = f"{OUT_DIR}/{key}.pt", f"{OUT_DIR}/{key}_history.json"
    set_seed(SEED)
    model = MODEL_CLASSES[key](MODEL_NAME).to(device)
    if SKIP_IF_TRAINED and os.path.exists(ckpt) and os.path.exists(hist_path):
        print(f"Memuat checkpoint {ckpt}")
        model.load_state_dict(torch.load(ckpt, map_location=device))
        return model, json.load(open(hist_path))

    bert_ids = {id(p) for p in model.bert.parameters()}
    optimizer = torch.optim.AdamW([
        {"params": list(model.bert.parameters()), "lr": LR_BERT},
        {"params": [p for p in model.parameters() if id(p) not in bert_ids], "lr": LR_HEAD},
    ], weight_decay=0.01)
    total_steps = math.ceil(len(train_loader) / ACCUM) * EPOCHS
    scheduler = get_linear_schedule_with_warmup(optimizer, int(WARMUP * total_steps), total_steps)
    scaler = torch.amp.GradScaler("cuda", enabled=USE_AMP)
    loss_fn = nn.CrossEntropyLoss()

    history, best_f1, bad_epochs = [], -1.0, 0
    for epoch in range(1, EPOCHS + 1):
        model.train()
        running, t0 = 0.0, time.time()
        optimizer.zero_grad()
        for step, b in enumerate(train_loader, 1):
            y = b.pop("labels").to(device)
            with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=USE_AMP):
                logits = model(**to_device(b))
            loss = loss_fn(logits.float(), y) / ACCUM
            scaler.scale(loss).backward()
            if step % ACCUM == 0 or step == len(train_loader):
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                scaler.step(optimizer); scaler.update(); scheduler.step()
                optimizer.zero_grad()
            running += loss.item() * ACCUM
            if step % 200 == 0 or step == len(train_loader):
                el = time.time() - t0
                print(f"epoch {epoch} | {step}/{len(train_loader)} | loss {running / step:.4f} | "
                      f"{el / 60:.1f} mnt, sisa ~{el / step * (len(train_loader) - step) / 60:.1f} mnt")

        y_val, p_val = predict_proba(model, val_loader)
        pred_val = (p_val >= 0.5).astype(int)
        rec = {"epoch": epoch, "train_loss": running / len(train_loader),
               "val_acc": accuracy_score(y_val, pred_val), "val_f1": f1_score(y_val, pred_val)}
        history.append(rec)
        print(f"Epoch {epoch}: loss={rec['train_loss']:.4f} val_acc={rec['val_acc']:.4f} val_f1={rec['val_f1']:.4f}")

        if rec["val_f1"] > best_f1:
            best_f1, bad_epochs = rec["val_f1"], 0
            torch.save(model.state_dict(), ckpt)
        else:
            bad_epochs += 1
            if bad_epochs >= PATIENCE:
                print("Early stopping")
                break

    json.dump(history, open(hist_path, "w"), indent=2)
    model.load_state_dict(torch.load(ckpt, map_location=device))
    return model, history


def run(key, name):
    model, HISTORIES[name] = train_model(key)
    y, p = predict_proba(model, test_loader)
    evaluate(name, y, (p >= 0.5).astype(int), p)
    del model
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

# %% [markdown]
# ## 9. IndoBERT

# %%
run("indobert", "IndoBERT")

# %% [markdown]
# ## 10. IndoBERT + BiLSTM

# %%
run("indobert_bilstm", "IndoBERT + BiLSTM")

# %% [markdown]
# ## 11. Perbandingan

# %%
res = pd.DataFrame(RESULTS).T.round(4)
res.to_csv(f"{OUT_DIR}/perbandingan_model.csv")
show(res)

metrics = ["accuracy", "precision", "recall", "f1"]
fig, ax = plt.subplots(figsize=(9, 4))
res[metrics].plot.bar(ax=ax, width=0.8)
ax.set_ylim(max(0, res[metrics].values.min() - 0.05), 1.0)
ax.set_ylabel("Skor (test)"); ax.legend(loc="lower right")
plt.xticks(rotation=15, ha="right")
show_plot(f"{OUT_DIR}/perbandingan_model.png")

per_src = pd.DataFrame(PER_SOURCE)
per_src.insert(0, "n_test", test_df.source.value_counts())
per_src.insert(1, "persen_hoaks", (test_df.groupby("source").label.mean() * 100).round(1))
per_src.to_csv(f"{OUT_DIR}/akurasi_per_sumber.csv")
show(per_src)

fig, axes = plt.subplots(1, 2, figsize=(10, 3.5))
for name, h in HISTORIES.items():
    h = pd.DataFrame(h)
    axes[0].plot(h.epoch, h.train_loss, marker="o", label=name)
    axes[1].plot(h.epoch, h.val_f1, marker="o", label=name)
axes[0].set_title("Train loss"); axes[1].set_title("F1 validation")
for a in axes:
    a.set_xlabel("Epoch"); a.legend(); a.grid(alpha=0.3)
show_plot(f"{OUT_DIR}/kurva_training.png")

# %% [markdown]
# ## 12. Prediksi teks baru

# %%
def bersihkan(t):
    t = re.sub(r"https?://\S+|www\.\S+", " ", str(t))
    t = re.sub(r"[@#]\w+", " ", t).lower()
    return re.sub(r"\s+", " ", t).strip()


best_name = max(["IndoBERT", "IndoBERT + BiLSTM"], key=lambda n: RESULTS[n]["f1"])
best_key = {"IndoBERT": "indobert", "IndoBERT + BiLSTM": "indobert_bilstm"}[best_name]
best_model = MODEL_CLASSES[best_key](MODEL_NAME).to(device)
best_model.load_state_dict(torch.load(f"{OUT_DIR}/{best_key}.pt", map_location=device))
print("Model:", best_name)


def prediksi(teks_list):
    df = pd.DataFrame({"text": [bersihkan(t) for t in teks_list], "label": 0})
    _, p = predict_proba(best_model, DataLoader(TextDataset(df), batch_size=16, collate_fn=collate))
    for t, prob in zip(teks_list, p):
        print(f"[{'HOAKS' if prob >= 0.5 else 'VALID'} | {prob:.3f}] {t[:120]}")


prediksi([
    "Pesan berantai: minum air kelapa dicampur garam bisa menyembuhkan covid dalam 1 hari, sebarkan!",
    "Bank Indonesia mempertahankan suku bunga acuan sebesar 6 persen dalam rapat dewan gubernur bulan ini.",
])
