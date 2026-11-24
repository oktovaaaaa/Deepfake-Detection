# Deteksi Berita Hoaks Berbahasa Indonesia

Proyek mata kuliah **AI untuk Deteksi Misinformasi dan Deepfake**.
Klasifikasi teks berita/unggahan media sosial berbahasa Indonesia menjadi **hoaks (1)** atau **valid (0)**,
dengan membandingkan model klasik, IndoBERT, dan model hybrid **IndoBERT + BiLSTM**.

## Struktur

```
├── training/
│   ├── gabung_dataset.py            # menggabungkan & membersihkan dataset mentah -> merged/
│   ├── training_deteksi_hoaks.py    # training & evaluasi semua model (Python)
│   └── training_deteksi_hoaks.ipynb # isi sama, versi notebook
├── Dataset Certan Fake News/
│   └── merged/                      # train.csv, val.csv, test.csv, ringkasan.txt
└── requirements.txt
```

## Dataset

76.134 data hasil gabungan 8 dataset publik hoaks Indonesia
(TurnBackHoax, Komdigi, CNN/Kompas/Tempo/Antara/Detik, IDNHoaxCorpus, Indonesia Political Hoax, dll.).

| Split | Jumlah | Valid (0) | Hoaks (1) |
|---|---|---|---|
| Train (80%) | 60.914 | 31.276 | 29.638 |
| Validation (10%) | 7.616 | 3.911 | 3.705 |
| Test (10%) | 7.604 | 3.903 | 3.701 |

Kolom: `text`, `label`, `source`. Split terstratifikasi per label dan sumber.
Rincian pembersihan dan jumlah per sumber ada di `Dataset Certan Fake News/merged/ringkasan.txt`.

Data mentah tidak disimpan di repo karena ukurannya besar. Untuk membuat ulang `merged/`,
letakkan data mentah di `Dataset Certan Fake News/` lalu jalankan `python training/gabung_dataset.py`.

## Model

| Model | Keterangan |
|---|---|
| Baseline panjang teks | Hanya jumlah kata, untuk mengukur bias dataset |
| TF-IDF + Logistic Regression / SVM | Baseline klasik |
| IndoBERT | `indobenchmark/indobert-base-p1`, fine-tuning |
| IndoBERT + BiLSTM | Model usulan (hybrid) |

Evaluasi pada data test: accuracy, precision, recall, F1, ROC-AUC, confusion matrix, dan akurasi per sumber.

## Menjalankan

```bash
pip install -r requirements.txt
python training/training_deteksi_hoaks.py
```

Disarankan memakai GPU. Hasil (tabel perbandingan, grafik, checkpoint model) tersimpan di `training/hasil_training/`.

**Server (berjalan di background):**
```bash
tmux new -s hoaks
python training/training_deteksi_hoaks.py 2>&1 | tee log_training.txt
```

**Google Colab:** upload folder `merged` ke `MyDrive/certan/merged`, buka `training/training_deteksi_hoaks.ipynb`,
pilih runtime GPU, lalu *Run all*.
