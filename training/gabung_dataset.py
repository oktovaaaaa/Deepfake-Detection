"""
Menggabungkan semua dataset berita hoaks berbahasa Indonesia menjadi satu dataset bersih.

Label akhir:  1 = hoaks,  0 = valid
Output (folder "Dataset Certan Fake News/merged"):
  - dataset_gabungan.csv    : semua data bersih (text, label, source)
  - train.csv / val.csv / test.csv : split 80/10/10 terstratifikasi (label x source)
  - ringkasan.txt           : statistik per sumber dan per label

Jalankan dari mana saja:  python training/gabung_dataset.py
"""
import re
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent.parent / "Dataset Certan Fake News"
DL = ROOT / "downloads"
OUT = ROOT / "merged"
SEED = 42


# ---------------------------------------------------------------- loader per dataset
def load_satria_data():  # folder archive/ (Data_latih.csv, label 1=hoaks)
    d = pd.read_csv(ROOT / "archive" / "Data_latih.csv")
    return pd.DataFrame({"text": d["judul"].fillna("") + ". " + d["narasi"].fillna(""),
                         "label": d["label"], "source": "satria_data_2020"})


def load_idnhoaxcorpus():  # tweet; label hoax/valid/? -> '?' dibuang
    d = pd.read_csv(ROOT / "IDNHoaxCorpus-main" / "IDNHoaxCorpus-main" / "dataset" / "datasetUMPOHoax.csv")
    d = d[d["label"].isin(["hoax", "valid"])]
    return pd.DataFrame({"text": d["tweet"], "label": (d["label"] == "hoax").astype(int),
                         "source": "idnhoaxcorpus_twitter"})


def load_irfanqs():  # SALAH=hoaks, BENAR=valid, kosong dibuang
    d = pd.read_csv(ROOT / "hoax-news-indonesian-dataset-master" / "hoax-news-indonesian-dataset-master" / "dataset.csv")
    d = d[d["Klasifikasi"].isin(["SALAH", "BENAR"])]
    return pd.DataFrame({"text": d["Pesan"], "label": (d["Klasifikasi"] == "SALAH").astype(int),
                         "source": "hoaxnews_irfanqs"})


def load_political_local():  # Indonesia Political Hoax Dataset (sudah 0/1)
    p = ROOT / "Indonesia Political Hoax Dataset" / "Indonesia Political Hoax Dataset"
    d = pd.concat([pd.read_csv(p / f) for f in ["combined_train.csv", "combined_val.csv", "combined_test.csv"]])
    return pd.DataFrame({"text": d["cleaned"], "label": d["label"], "source": "political_hoax_local"})


def load_linkgish():  # Kaggle: Indonesian Fact and Hoax Political News
    p = DL / "indonesian-fact-and-hoax-political-news" / "Cleaned"
    parts = []
    for f in ["dataset_cnn_10k_cleaned.xlsx", "dataset_kompas_4k_cleaned.xlsx", "dataset_tempo_6k_cleaned.xlsx"]:
        d = pd.read_excel(p / f)
        parts.append(pd.DataFrame({"text": d["Title"].fillna("") + ". " + d["FullText"].fillna(""),
                                   "label": 0, "source": "linkgish_" + f.split("_")[1]}))
    # Hoaks: pakai Judul + Narasi (isi hoaks aslinya), BUKAN FullText (itu artikel bantahan)
    d = pd.read_excel(p / "dataset_turnbackhoax_10_cleaned.xlsx")
    parts.append(pd.DataFrame({"text": d["Title"].fillna("") + ". " + d["Narasi"].fillna(""),
                               "label": 1, "source": "linkgish_turnbackhoax"}))
    return pd.concat(parts)


def load_azis():  # Kaggle: Deteksi Berita Hoaks Indo (Mochamad Abdul Azis)
    p = DL / "deteksi-berita-hoaks-indo-dataset" / "Cleaned Dataset V3"
    parts = []
    for name in ["antaranews", "detik", "kompas"]:
        d = pd.read_csv(p / f"{name}_cleaned_v3.csv")
        parts.append(pd.DataFrame({"text": d["judul"].fillna("") + ". " + d["narasi"].fillna(""),
                                   "label": 0, "source": f"azis_{name}"}))
    d = pd.read_csv(p / "tbh_cleaned_v3.csv")
    parts.append(pd.DataFrame({"text": d["judul"].fillna("") + ". " + d["narasi"].fillna(""),
                               "label": 1, "source": "azis_turnbackhoax"}))
    return pd.concat(parts)


def load_komdigi():  # Kaggle: klarifikasi hoaks Komdigi -> semua hoaks.
    # Hanya judul yang dipakai: isi artikel adalah penjelasan/bantahan (bocor label).
    d = pd.read_csv(DL / "indonesian-hoax-news-dataset" / "komdigi_hoaks.csv")
    return pd.DataFrame({"text": d["title"], "label": 1, "source": "komdigi_judul"})


def load_mafindo():  # HF nlp-brin-id/fakenews-mafindo (gated) - dipakai jika sudah diunduh
    p = DL / "fakenews-mafindo"
    if not p.exists():
        print("[lewati] fakenews-mafindo belum diunduh (perlu izin akses di Hugging Face)")
        return pd.DataFrame(columns=["text", "label", "source"])
    d = pd.concat([pd.read_csv(f) for f in sorted(p.glob("*.csv"))])
    text_col = next(c for c in d.columns if c.lower() in ("text", "content", "narasi", "fact", "body"))
    label_col = next(c for c in d.columns if c.lower() in ("label_id", "label"))
    lab = d[label_col]
    if lab.dtype == object:
        lab = lab.str.upper().eq("HOAX").astype(int)
    return pd.DataFrame({"text": d[text_col], "label": lab, "source": "mafindo_brin"})


# ---------------------------------------------------------------- pembersihan
MEDIA = r"(cnn\s*indonesia|cnnindonesia(\.com)?|kompas(\.com|\.id|\s*tv)?|tempo(\.co)?|antara(news)?(\.com)?|" \
        r"detik(\.com|com|news|health|finance|inet)?|liputan\s*6(\.com)?|tribun(news)?(\.com)?|okezone|" \
        r"kumparan|tirto(\.id)?|turn\s*back\s*hoa[xk]s?(\.id)?|mafindo|kominfo|komdigi|snopes(\.com)?)"
DATELINE = re.compile(r"^\s*[A-Za-z .]{0,30}[,(]\s*" + MEDIA + r"\s*\)?\s*[-–—]+\s*", re.I)
# dateline di awal isi berita (setelah judul): "Jakarta, CNN Indonesia --", "TEMPO.CO, Jakarta -",
# "Jakarta (ANTARA) -", "KOMPAS.com -", "Jakarta -"
BODY_DATELINE = re.compile(r"(^|\.\s+)(?:[A-Z][\w.]{1,15}\s*,\s*)?[A-Z][A-Za-z ]{1,25}"
                           r"(?:\s*,\s*[A-Za-z][\w .]{1,20}?|\s*\([^)]{1,25}\))?\s*[-–—]{1,3}\s+")
# kata label di awal judul cek fakta: "Keliru, ...", "FITNAH: ...", "Salah: ..."
LEAD_LABEL = re.compile(r"(^|\.\s+)\s*(keliru|fitnah|salah|sesat|menyesatkan|hoaks?|hoax|disinformasi|"
                        r"misinformasi|benar|cek fakta)\s*[:,!\-–]\s*", re.I)
TAG = re.compile(r"\[(salah|hoaks?|hoax|disinformasi|misinformasi|klarifikasi|fitnah|benar|valid)[^\]]*\]", re.I)
# kalimat bernada pemeriksa fakta -> dibuang dari SEMUA teks (adil untuk dua label)
FACTCHECK = re.compile(r"(hoaks|hoax|hoaxs|disinformasi|misinformasi|cek\s*fakta|periksa\s*fakta|faktanya|"
                       r"penelusuran|menyesatkan|tidak\s+benar|klaim\s+tersebut|klarifikasi|"
                       r"dilansir|baca\s+juga|simak\s+juga|referensi\s*:)", re.I)
URL = re.compile(r"https?://\S+|www\.\S+|\S+\.(com|id|co|net|org)\S*", re.I)
ID_WORDS = {"yang", "dan", "di", "ini", "itu", "dengan", "untuk", "tidak", "dari", "dalam", "akan", "ada", "kita",
            "saya", "pada", "juga", "karena", "sudah", "bisa", "atau", "mereka", "kami", "jadi", "oleh", "tersebut"}
EN_WORDS = {"the", "and", "is", "are", "of", "to", "in", "that", "this", "it", "for", "with", "was", "they",
            "have", "be", "on", "not", "you", "from", "by", "an", "what", "their", "all"}


def clean(t: str) -> str:
    t = str(t)
    t = TAG.sub(" ", t)
    t = DATELINE.sub("", t)
    t = BODY_DATELINE.sub(r"\1", t)
    t = LEAD_LABEL.sub(r"\1", t)
    t = re.sub(r"gambas\s*:?\s*video\s*\w*", " ", t, flags=re.I)
    t = URL.sub(" ", t)
    t = re.sub(r"[@#]\w+", " ", t)
    sents = re.split(r"(?<=[.!?\n])\s+", t)
    t = " ".join(s for s in sents if not FACTCHECK.search(s))
    t = re.sub(MEDIA, " ", t, flags=re.I)
    t = t.lower()
    t = re.sub(r"[^\w\s.,!?'\"%-]", " ", t)
    t = re.sub(r"\s+", " ", t).strip(" .")
    return t


def is_indonesian(t: str) -> bool:
    w = t.split()
    return sum(x in ID_WORDS for x in w) >= sum(x in EN_WORDS for x in w)


# ---------------------------------------------------------------- main
def main():
    loaders = [load_satria_data, load_idnhoaxcorpus, load_irfanqs, load_political_local,
               load_linkgish, load_azis, load_komdigi, load_mafindo]
    df = pd.concat([f() for f in loaders], ignore_index=True)
    log = [f"Data mentah (berlabel): {len(df)}"]

    df["label"] = df["label"].astype(int)
    df["text"] = df["text"].fillna("").map(clean)

    n = len(df); df = df[df["text"].str.split().str.len() >= 5]
    log.append(f"Dibuang teks < 5 kata: {n - len(df)}")
    n = len(df); df = df[df["text"].map(is_indonesian)]
    log.append(f"Dibuang teks bukan bahasa Indonesia: {n - len(df)}")

    # Deduplikasi: kunci = 300 karakter pertama (alfanumerik saja)
    df["key"] = df["text"].str.replace(r"[^a-z0-9]", "", regex=True).str[:300]
    conflict = df.groupby("key")["label"].nunique()
    conflict = conflict[conflict > 1].index
    n = len(df); df = df[~df["key"].isin(conflict)]
    log.append(f"Dibuang duplikat dengan label bertentangan: {n - len(df)}")
    n = len(df); df = df.drop_duplicates("key")
    log.append(f"Dibuang duplikat: {n - len(df)}")
    df = df.drop(columns="key").sample(frac=1, random_state=SEED).reset_index(drop=True)

    OUT.mkdir(exist_ok=True)
    df.to_csv(OUT / "dataset_gabungan.csv", index=False)

    # Split 80/10/10 terstratifikasi per (label, source)
    df["_r"] = df.groupby(["label", "source"]).cumcount() / df.groupby(["label", "source"])["label"].transform("size")
    splits = {"train": df[df._r < .8], "val": df[(df._r >= .8) & (df._r < .9)], "test": df[df._r >= .9]}
    for k, v in splits.items():
        v.drop(columns="_r").to_csv(OUT / f"{k}.csv", index=False)

    log.append(f"\nTOTAL BERSIH: {len(df)}  (hoaks=1: {(df.label == 1).sum()}, valid=0: {(df.label == 0).sum()})")
    log.append("Split: " + ", ".join(f"{k}={len(v)}" for k, v in splits.items()))
    log.append("\nPer sumber:\n" + df.pivot_table(index="source", columns="label", values="text",
                                                  aggfunc="count", fill_value=0).to_string())
    txt = "\n".join(log)
    (OUT / "ringkasan.txt").write_text(txt, encoding="utf-8")
    print(txt)


if __name__ == "__main__":
    main()
