# AI untuk Deteksi Misinformasi dan Berita Hoaks Berbahasa Indonesia

Proyek Mata Kuliah: **AI untuk Deteksi Misinformasi dan Deepfake**  
Dosen Pengampu: **Pak RCH**

Arsitektur Model: **Hybrid IndoBERT + Bi-directional LSTM with Attention Mechanism**

---

## 📌 Ringkasan Progres 1
- **Topik:** Deteksi Misinformasi & Berita Palsu Berbahasa Indonesia.
- **Dataset:** 19.630 berita seimbang (10.000 Hoaks dari TurnBackHoax/CekFakta dan 9.630 Berita Valid dari CNN Indonesia).
- **Pembagian Data:** 80% Train (15.704 data), 10% Val (1.963 data), 10% Test (1.963 data).
- **Model:** `IndoBERT` sebagai *feature extractor*, `BiLSTM` untuk pemodelan sekuensial, dan `Bahdanau Attention` untuk menyoroti frasa provokatif.
- **Dokumen Progres 1:** Tersedia lengkap di [`docs/LAPORAN_PROGRESS_1.md`](docs/LAPORAN_PROGRESS_1.md).

---

## 🚀 Panduan Memulai Cepat (Quickstart)

### 1. Instalasi Dependensi
```bash
pip install -r requirements.txt
```

### 2. Preprocessing Data (Opsional jika ingin rebuild ulang dataset)
```bash
python src/preprocess.py
```
Data hasil pembersihan akan otomatis tersimpan di folder `data/processed/` (`train.csv`, `val.csv`, `test.csv`).

### 3. Menjalankan Training Model
Untuk melatih model Hybrid (disarankan menggunakan GPU/Google Colab):
```bash
python src/train.py --epochs 3 --batch_size 16 --lr 2e-5
```

Untuk mode uji coba cepat (*debugging subset kecil*):
```bash
python src/train.py --debug --epochs 1 --batch_size 8
```

### 4. Menggunakan Google Colab
Buka notebook di [`notebooks/01_eda_and_hybrid_model.ipynb`](notebooks/01_eda_and_hybrid_model.ipynb), lalu upload ke Google Colab dengan Runtime GPU (T4) untuk pelatihan yang lebih cepat.

---

## 📂 Struktur Direktori
```
DeepFake Detection/
├── data/
│   ├── raw/                  # Data mentah
│   └── processed/            # train.csv, val.csv, test.csv
├── docs/
│   └── LAPORAN_PROGRESS_1.md # Dokumen resmi Progres 1
├── notebooks/
│   └── 01_eda_and_hybrid_model.ipynb
├── src/
│   ├── preprocess.py
│   ├── dataset.py
│   ├── model.py
│   └── train.py
├── requirements.txt
└── README.md
```

