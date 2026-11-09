"""
Training and Evaluation Script for IndoBERT + BiLSTM + Attention
Can be executed locally or on Google Colab / Kaggle Notebook with GPU.
"""

import os
import argparse
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, classification_report
from transformers import AutoTokenizer

from src.model import HybridIndoBERTBiLSTMAttention
from src.dataset import IndonesianHoaxDataset

def evaluate(model, dataloader, criterion, device):
    model.eval()
    total_loss = 0.0
    all_preds = []
    all_labels = []

    with torch.no_grad():
        for batch in dataloader:
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["label"].to(device)

            logits, _ = model(input_ids=input_ids, attention_mask=attention_mask)
            loss = criterion(logits, labels)
            total_loss += loss.item()

            preds = torch.argmax(logits, dim=1).cpu().numpy()
            all_preds.extend(preds)
            all_labels.extend(labels.cpu().numpy())

    avg_loss = total_loss / max(len(dataloader), 1)
    acc = accuracy_score(all_labels, all_preds)
    precision, recall, f1, _ = precision_recall_fscore_support(all_labels, all_preds, average="binary")
    
    return avg_loss, acc, precision, recall, f1, all_preds, all_labels

def train(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device yang digunakan: {device}")

    # Load data
    train_df = pd.read_csv(os.path.join(args.data_dir, "train.csv"))
    val_df = pd.read_csv(os.path.join(args.data_dir, "val.csv"))
    test_df = pd.read_csv(os.path.join(args.data_dir, "test.csv"))

    if args.debug:
        train_df = train_df.head(200)
        val_df = val_df.head(50)
        test_df = test_df.head(50)
        print("[DEBUG MODE] Menggunakan subset data kecil untuk verifikasi pipeline.")

    # Tokenizer
    print(f"Memuat tokenizer: {args.bert_model}")
    tokenizer = AutoTokenizer.from_pretrained(args.bert_model)

    train_dataset = IndonesianHoaxDataset(train_df, tokenizer, max_length=args.max_length)
    val_dataset = IndonesianHoaxDataset(val_df, tokenizer, max_length=args.max_length)
    test_dataset = IndonesianHoaxDataset(test_df, tokenizer, max_length=args.max_length)

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False)

    # Inisialisasi model
    model = HybridIndoBERTBiLSTMAttention(
        bert_model_name=args.bert_model,
        lstm_hidden_dim=args.lstm_hidden_dim,
        dense_dim=args.dense_dim,
        dropout=args.dropout,
        freeze_bert=args.freeze_bert
    ).to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    os.makedirs(args.output_dir, exist_ok=True)
    best_f1 = 0.0
    best_model_path = os.path.join(args.output_dir, "best_hybrid_model.pt")

    print("\n--- Memulai Training ---")
    for epoch in range(1, args.epochs + 1):
        model.train()
        total_train_loss = 0.0

        for step, batch in enumerate(train_loader):
            optimizer.zero_grad()
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["label"].to(device)

            logits, _ = model(input_ids=input_ids, attention_mask=attention_mask)
            loss = criterion(logits, labels)
            loss.backward()

            nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            total_train_loss += loss.item()

            if (step + 1) % 50 == 0:
                print(f"Epoch {epoch}/{args.epochs} | Step {step + 1}/{len(train_loader)} | Batch Loss: {loss.item():.4f}")

        avg_train_loss = total_train_loss / len(train_loader)
        val_loss, val_acc, val_p, val_r, val_f1, _, _ = evaluate(model, val_loader, criterion, device)

        print(f"\n[Epoch {epoch} Selesai]")
        print(f"  Train Loss: {avg_train_loss:.4f} | Val Loss: {val_loss:.4f}")
        print(f"  Val Acc: {val_acc:.4f} | Val Precision: {val_p:.4f} | Val Recall: {val_r:.4f} | Val F1: {val_f1:.4f}")

        if val_f1 > best_f1:
            best_f1 = val_f1
            torch.save(model.state_dict(), best_model_path)
            print(f"  >>> Model terbaik baru disimpan dengan Val F1: {best_f1:.4f} di {best_model_path}")

    # Evaluasi akhir pada Test set
    print("\n--- Evaluasi Akhir pada Test Set ---")
    if os.path.exists(best_model_path):
        model.load_state_dict(torch.load(best_model_path))
    test_loss, test_acc, test_p, test_r, test_f1, test_preds, test_labels = evaluate(model, test_loader, criterion, device)
    print(f"Test Loss: {test_loss:.4f} | Test Acc: {test_acc:.4f} | Test F1: {test_f1:.4f}")
    print("\nClassification Report (Test Set):")
    print(classification_report(test_labels, test_preds, target_names=["Fakta (0)", "Hoaks (1)"]))

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", type=str, default="data/processed")
    parser.add_argument("--output_dir", type=str, default="models")
    parser.add_argument("--bert_model", type=str, default="indobenchmark/indobert-base-p1")
    parser.add_argument("--max_length", type=int, default=128)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--lr", type=float, default=2e-5)
    parser.add_argument("--weight_decay", type=float, default=0.01)
    parser.add_argument("--lstm_hidden_dim", type=int, default=128)
    parser.add_argument("--dense_dim", type=int, default=64)
    parser.add_argument("--dropout", type=float, default=0.3)
    parser.add_argument("--freeze_bert", action="store_true", help="Bekukan bobot IndoBERT untuk percepat training")
    parser.add_argument("--debug", action="store_true", help="Jalankan subset kecil untuk debugging")
    
    args = parser.parse_args()
    train(args)

