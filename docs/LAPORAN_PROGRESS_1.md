# DOKUMEN PROGRES 1: USULAN PROYEK PENELITIAN
**Mata Kuliah:** AI untuk Deteksi Misinformasi dan Deepfake  
**Dosen Pengampu:** Pak RCH  
**Topik:** AI untuk Deteksi Misinformasi dan Berita Palsu Berbahasa Indonesia  
**Judul Usulan:** *"Klasifikasi Misinformasi dan Berita Hoaks Berbahasa Indonesia Menggunakan Arsitektur Hybrid IndoBERT-BiLSTM dengan Mekanisme Attention"*

---

## 1. Latar Belakang & Urgensi Penelitian
Di era digital, penyebaran berita palsu (*fake news*) dan disinformasi di Indonesia telah menjadi ancaman serius bagi kohesi sosial, stabilitas politik, dan pengambilan keputusan publik. Platform pesan instan (seperti WhatsApp) dan media sosial (X/Twitter, Facebook, TikTok) menjadi saluran utama penyebaran narasi fabrikasi yang dikemas secara emosional dan sensasional.

Tantangan utama dalam mendeteksi misinformasi berbahasa Indonesia meliputi:
1. **Kekayaan Morfologis & Ragam Bahasa:** Bahasa Indonesia di media sosial kerap memadukan bahasa baku, bahasa gaul (*slang*), singkatan, dan istilah lokal.
2. **Keterbatasan Model Konvensional:** Pendekatan berbasis *bag-of-words* atau TF-IDF dengan algoritma Machine Learning klasik (SVM, Naive Bayes) gagal menangkap dependensi kontekstual dan semantik kalimat panjang.
3. **Kelemahan Model Sekuensial Murni:** Model RNN/LSTM murni rentan kehilangan pemahaman konteks semantik mendalam yang dapat disediakan oleh model berbasis Transformer (*Pretrained Language Models*).

Untuk mengatasi tantangan ini, penelitian ini mengusulkan pendekatan **Hybrid Deep Learning** yang memadukan pemahaman semantik kontekstual tingkat tinggi dari **IndoBERT**, pemodelan dependensi sekuensial dua arah dari **Bidirectional LSTM (BiLSTM)**, serta **Attention Mechanism** untuk menitikberatkan bobot pada kata-kata/frasa provokatif yang menjadi ciri khas fabrikasi narasi hoaks.

---

## 2. Rumusan Masalah & Tujuan

### Rumusan Masalah:
1. Bagaimana mengintegrasikan model representasi bahasa *IndoBERT* dengan *BiLSTM* dan *Attention Mechanism* untuk mendeteksi berita hoaks berbahasa Indonesia?
2. Bagaimana kinerja performa arsitektur hybrid dibandingkan dengan baseline model (*Single IndoBERT* dan *Machine Learning konvensional*)?
3. Sejauh mana mekanisme *attention* mampu mengidentifikasi kata kunci/klaim provokatif pembentuk misinformasi?

### Tujuan Penelitian:
1. Membangun dataset terstandarisasi dan seimbang dari sumber cek fakta terpercaya di Indonesia (TurnBackHoax/Mafindo dan CNN Indonesia).
2. Merancang, mengimplementasikan, dan melatih model hybrid *IndoBERT + BiLSTM + Attention*.
3. Mengevaluasi keandalan model menggunakan metrik *Accuracy*, *Precision*, *Recall*, dan *F1-Score*.
4. Menyiapkan fondasi untuk prototipe sistem *fact-checking* berbasis AI yang dapat digunakan masyarakat umum.

---

## 3. Dataset & Karakteristik Data

### 3.1 Sumber Data
Dataset yang digunakan dalam penelitian ini merupakan hasil kurasi dan integrasi dari dataset publik terpercaya yang telah diajukan:
- **TurnBackHoax (Mafindo / CekFakta):** Kumpulan klaim dan narasi yang telah terverifikasi sebagai **Hoaks (Label: 1)** oleh pemeriksa fakta profesional.
- **CNN Indonesia (Media Berita Terverifikasi):** Kumpulan artikel berita faktual dan telah melalui proses editorial sebagai representasi data **Fakta/Valid (Label: 0)**.
- **Referensi Pembanding:** IDNHoaxCorpus dan Dataset Berita Palsu BRIN (*Badan Riset dan Inovasi Nasional*).

### 3.2 Distribusi & Pembagian Data (Data Splitting)
Data telah dibersihkan (menghapus tag HTML, URL, mention, normalisasi spasi) dan dibagi menggunakan metode **Stratified Splitting** untuk menjamin representasi kelas yang seimbang:

| Subset | Jumlah Sampel | Kelas 0 (Fakta) | Kelas 1 (Hoaks) | Persentase |
| :--- | :---: | :---: | :---: | :---: |
| **Train Set** | 15.704 | 7.704 (49.1%) | 8.000 (50.9%) | 80% |
| **Validation Set** | 1.963 | 963 (49.1%) | 1.000 (50.9%) | 10% |
| **Test Set** | 1.963 | 963 (49.1%) | 1.000 (50.9%) | 10% |
| **Total** | **19.630** | **9.630** | **10.000** | **100%** |

> **Catatan Kritis:** Dataset memiliki rasio kelas mendekati **1:1 (Balanced)**, sehingga model terhindar dari permasalahan *class imbalance bias*.

---

## 4. Desain Arsitektur Model Hybrid

Model yang diusulkan menggabungkan 4 komponen utama:

```
[Input Text: Berita / Klaim]
           │
           ▼
┌──────────────────────────────────────┐
│  IndoBERT Tokenizer & Embedding      │  --> Konversi teks ke representasi token (seq_len = 128)
└──────────────────────────────────────┘
           │
           ▼
┌──────────────────────────────────────┐
│  IndoBERT Base (Transformer Encoder) │  --> Ekstraksi fitur semantik kontekstual (dim = 768)
└──────────────────────────────────────┘
           │
           ▼
┌──────────────────────────────────────┐
│  Bidirectional LSTM (BiLSTM)         │  --> Menangkap dependensi sekuensial maju & mundur (dim = 256)
└──────────────────────────────────────┘
           │
           ▼
┌──────────────────────────────────────┐
│  Bahdanau Attention Mechanism        │  --> Menghitung bobot pentingnya tiap kata provokatif
└──────────────────────────────────────┘
           │
           ▼
┌──────────────────────────────────────┐
│  Classification Head (Dense + Drop)  │  --> Fully-Connected + Dropout + Softmax
└──────────────────────────────────────┘
           │
           ▼
[Output: Prediksi Probabilitas Fakta vs Hoaks]
```

### Justifikasi Arsitektur:
1. **IndoBERT (`indobenchmark/indobert-base-p1`):** Memberikan pemahaman morfologis dan semantik yang sangat tinggi terhadap leksikon bahasa Indonesia modern.
2. **BiLSTM:** Menganalisis alur retorika berita. Berita bohong sering kali memiliki pola kalimat berulang atau struktur narasi yang memancing kepanikan.
3. **Bahdanau Attention:** Memberikan interpretasi visual (*explainability*) terhadap kata-kata yang memicu klasifikasi hoaks (misal: "sebarkan sebelum dihapus", "terbukti konspirasi").

---

## 5. Rencana Evaluasi & Baseline

### 5.1 Model Pembanding (Baseline)
Untuk membuktikan keunggulan model hybrid yang diusulkan, penelitian akan menguji 3 baseline:
1. **Baseline 1:** TF-IDF + Logistic Regression / Support Vector Machine (Machine Learning Klasik).
2. **Baseline 2:** BiLSTM murni dengan Word2Vec/FastText embedding.
3. **Baseline 3:** Fine-tuned IndoBERT murni (Single Transformer).

### 5.2 Metrik Evaluasi
- **Accuracy:** Mengukur ketepatan prediksi keseluruhan.
- **Precision & Recall:** Mengukur kemampuan model meminimalkan *False Positive* (berita valid dianggap hoaks) dan *False Negative* (hoaks lolos sebagai berita valid).
- **F1-Score (Macro & Binary):** Metrik utama penentu keandalan model pada kedua kelas.
- **Confusion Matrix:** Analisis detail sebaran prediksi.

---

## 6. Jadwal & Rencana Kerja (Roadmap)

| Tahapan Proyek | Aktivitas Utama | Output | Status |
| :--- | :--- | :--- | :---: |
| **Progress 1 (Saat Ini)** | - Perumusan topik & latar belakang<br>- Kurasi & pembersihan dataset (19.630 data)<br>- Desain arsitektur Hybrid IndoBERT-BiLSTM-Attention<br>- Setup environment & repository | Dokumen Progres 1, Script Preprocessing, Baseline Dataset | **SELESAI (100%)** |
| **Progress 2** | - Implementasi training pipeline di PyTorch/Colab<br>- Hyperparameter tuning (Learning rate, batch size, dropout)<br>- Training baseline model (TF-IDF vs IndoBERT vs Hybrid) | Log hasil eksperimen & komparasi metrik awal | *Tahap Selanjutnya* |
| **Progress 3 / Final** | - Evaluasi akhir pada Test Set & Error Analysis<br>- Visualisasi Attention Weights (kata kunci pemicu hoaks)<br>- Pembuatan antarmuka demo interaktif (Streamlit/FastAPI)<br>- Penyusunan Laporan Akhir & Slide Presentasi | Aplikasi Demo & Laporan Akhir Lengkap | *Tahap Selanjutnya* |

---

## 7. Struktur Repository Proyek
```
DeepFake Detection/
├── data/
│   ├── raw/                  # Data mentah (TurnBackHoax, CNN, IDNHoaxCorpus, BRIN)
│   └── processed/            # Data bersih: train.csv, val.csv, test.csv
├── docs/
│   └── LAPORAN_PROGRESS_1.md # Dokumen laporan kemajuan Progres 1
├── notebooks/
│   └── 01_eda_and_hybrid_model.ipynb # Notebook siap pakai untuk Google Colab / GPU
├── src/
│   ├── preprocess.py         # Skrip otomatisasi kurasi data
│   ├── dataset.py            # PyTorch Dataset Loader
│   ├── model.py              # Arsitektur Hybrid IndoBERT + BiLSTM + Attention
│   └── train.py              # Training, evaluasi, dan simpan model terbaik
├── requirements.txt          # Dependensi pustaka Python
└── README.md                 # Panduan eksekusi proyek
```

