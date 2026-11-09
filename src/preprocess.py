"""
Data Preprocessing Script for Indonesian Fake News & Hoax Detection
Menggabungkan data dari sumber publik CekFakta/TurnBackHoax dan Media Terverifikasi (CNN Indonesia, dsb)
menjadi format terstandarisasi untuk training model.
"""

import os
import re
import pandas as pd
from sklearn.model_selection import train_test_split

def clean_text(text: str) -> str:
    """Membersihkan teks dari URL, mention, karakter aneh, dan spasi berlebih."""
    if not isinstance(text, str):
        return ""
    
    # Hapus URL
    text = re.sub(r"https?://\S+|www\.\S+", " ", text)
    # Hapus tag HTML jika ada
    text = re.sub(r"<.*?>", " ", text)
    # Hapus mention Twitter / username
    text = re.sub(r"@\w+", " ", text)
    # Hapus karakter non-ASCII dan simbol berlebih kecuali tanda baca standar
    text = re.sub(r"[^\w\s.,!?-]", " ", text)
    # Normalisasi spasi berlebih
    text = re.sub(r"\s+", " ", text).strip()
    return text

def build_dataset(
    output_dir: str = "data/processed",
    max_samples_per_class: int = 10000,
    random_state: int = 42
):
    os.makedirs(output_dir, exist_ok=True)
    
    print("[1/4] Memuat data Hoaks (TurnBackHoax)...")
    tbh_path = "data/raw/kaggle_political/Cleaned/dataset_turnbackhoax_10_cleaned.xlsx"
    df_tbh = pd.read_excel(tbh_path)
    
    df_tbh_clean = pd.DataFrame({
        "title": df_tbh["Title"].apply(clean_text),
        "text": df_tbh["Clean Narasi"].fillna(df_tbh["FullText"]).apply(clean_text),
        "label": 1,  # 1 = Hoaks / Berita Palsu
        "source": "TurnBackHoax"
    })
    
    print("[2/4] Memuat data Fakta / Berita Valid (CNN Indonesia)...")
    cnn_path = "data/raw/kaggle_political/Cleaned/dataset_cnn_10k_cleaned.xlsx"
    df_cnn = pd.read_excel(cnn_path)
    
    df_cnn_clean = pd.DataFrame({
        "title": df_cnn["Title"].apply(clean_text),
        "text": df_cnn["text_new"].fillna(df_cnn["FullText"]).apply(clean_text),
        "label": 0,  # 0 = Fakta / Berita Valid
        "source": "CNN Indonesia"
    })

    # Filter teks kosong atau terlalu pendek (< 20 karakter)
    df_tbh_clean = df_tbh_clean[df_tbh_clean["text"].str.len() > 20]
    df_cnn_clean = df_cnn_clean[df_cnn_clean["text"].str.len() > 20]
    
    if max_samples_per_class:
        df_tbh_clean = df_tbh_clean.head(max_samples_per_class)
        df_cnn_clean = df_cnn_clean.head(max_samples_per_class)

    print(f"Data Hoaks valid: {len(df_tbh_clean)}")
    print(f"Data Fakta valid: {len(df_cnn_clean)}")

    # Gabung
    combined_df = pd.concat([df_tbh_clean, df_cnn_clean], ignore_index=True)
    combined_df = combined_df.sample(frac=1.0, random_state=random_state).reset_index(drop=True)

    # Gabungkan title + text sebagai input kalimat utama untuk klasifikasi
    combined_df["full_content"] = combined_df["title"] + " " + combined_df["text"]

    # Simpan dataset lengkap
    full_path = os.path.join(output_dir, "dataset_hoax_clean.csv")
    combined_df.to_csv(full_path, index=False)
    print(f"[3/4] Dataset lengkap tersimpan di: {full_path} ({len(combined_df)} baris)")

    # Split: 80% Train, 10% Validation, 10% Test
    train_df, temp_df = train_test_split(
        combined_df, test_size=0.2, random_state=random_state, stratify=combined_df["label"]
    )
    val_df, test_df = train_test_split(
        temp_df, test_size=0.5, random_state=random_state, stratify=temp_df["label"]
    )

    train_path = os.path.join(output_dir, "train.csv")
    val_path = os.path.join(output_dir, "val.csv")
    test_path = os.path.join(output_dir, "test.csv")

    train_df.to_csv(train_path, index=False)
    val_df.to_csv(val_path, index=False)
    test_df.to_csv(test_path, index=False)

    print("[4/4] Dataset split berhasil:")
    print(f"  - Train: {len(train_df)} baris (Tersimpan di: {train_path})")
    print(f"  - Val:   {len(val_df)} baris (Tersimpan di: {val_path})")
    print(f"  - Test:  {len(test_df)} baris (Tersimpan di: {test_path})")
    print("\nDistribusi Label pada Train:")
    print(train_df["label"].value_counts(normalize=True).rename({0: "0 (Fakta)", 1: "1 (Hoaks)"}))

if __name__ == "__main__":
    build_dataset()

