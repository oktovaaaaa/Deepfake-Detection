# Validasi Ringkasan dan Pembagian Dataset — Progress 3

**Proyek:** Deteksi Berita Hoaks Berbahasa Indonesia Menggunakan Model Hibrida IndoBERT–BiLSTM dan Evaluasi Generalisasi Lintas Sumber  
**Kelompok:** 07 — D3 Teknologi Informasi, Institut Teknologi Del  
**Penyusun pemeriksaan:** Laura Aurelia (42324027)  
**Dasar pemeriksaan:** ringkasan hasil penggabungan data pada commit `03861173bfc90056851eb6c7807d39fd980d40a5` yang diunggah oleh anggota kelompok (Oktova Samosir).  
**Ruang lingkup:** verifikasi aritmetika dan konsistensi statistik yang *tertulis* dalam `merged/ringkasan.txt`, bukan audit isi CSV baris per baris.

## 1. Tujuan

Memeriksa konsistensi jumlah data setelah pembersihan, distribusi label, statistik per sumber, dan proporsi pembagian data latih (*train*), validasi (*validation*), serta uji (*test*). Pemeriksaan dilakukan terhadap angka yang dipublikasikan pada `ringkasan.txt` dan nama berkas pembagian yang tampil pada commit.

## 2. Sumber bukti

- `Dataset Certan Fake News/merged/ringkasan.txt` — sumber angka pada bagian 3–5.
- `Dataset Certan Fake News/merged/train.csv`, `val.csv`, dan `test.csv` — berkas keluaran pembagian data yang terlihat pada commit.
- Commit GitHub: https://github.com/oktovaaaaa/Deepfake-Detection/commit/03861173bfc90056851eb6c7807d39fd980d40a5

**Catatan:** Jumlah baris pada file CSV belum dihitung ulang secara mandiri dalam pemeriksaan ini. Angka pembagian di bawah berasal dari `ringkasan.txt`, bukan dari pembacaan langsung seluruh CSV.

## 3. Pemeriksaan jumlah data

| Tahapan | Jumlah baris |
|---|---:|
| Data mentah berlabel | 86.086 |
| Dibuang: teks kurang dari 5 kata | 3.092 |
| Dibuang: teks bukan bahasa Indonesia | 966 |
| Dibuang: duplikat dengan label bertentangan | 228 |
| Dibuang: duplikat | 5.666 |
| **Total data dibuang** | **9.952** |
| **Total bersih** | **76.134** |

**Pemeriksaan aritmetika:**

- Data dibuang = 3.092 + 966 + 228 + 5.666 = **9.952**.
- Data bersih = 86.086 − 9.952 = **76.134**.
- **Hasil:** jumlah tahapan pembersihan konsisten secara aritmetika dengan total bersih yang dilaporkan.

## 4. Pemeriksaan distribusi label

| Label | Jumlah | Persentase dari data bersih |
|---|---:|---:|
| Fakta (`0`) | 39.090 | 51,34% |
| Hoaks (`1`) | 37.044 | 48,66% |
| **Total** | **76.134** | **100,00%** |

**Pemeriksaan:** 39.090 + 37.044 = **76.134**. Kedua kelas tidak sama besar persis; selisihnya **2.046** baris. Maka, data tidak boleh disebut seimbang tepat 1:1.

### Verifikasi jumlah berdasarkan sumber

| Sumber (`source`) | Fakta (0) | Hoaks (1) |
|---|---:|---:|
| `azis_antaranews` | 4.066 | 0 |
| `azis_detik` | 3.575 | 0 |
| `azis_kompas` | 4.498 | 0 |
| `azis_turnbackhoax` | 0 | 11.606 |
| `hoaxnews_irfanqs` | 147 | 108 |
| `idnhoaxcorpus_twitter` | 633 | 2.712 |
| `komdigi_judul` | 0 | 4.068 |
| `lingkish_cnn` | 7.226 | 0 |
| `lingkish_kompas` | 3.828 | 0 |
| `lingkish_tempo` | 6.489 | 0 |
| `lingkish_turnbackhoax` | 0 | 8.457 |
| `political_hoax_local` | 8.030 | 6.801 |
| `satria_data_2020` | 598 | 3.292 |
| **Total** | **39.090** | **37.044** |

Jumlah per sumber cocok dengan total setiap label dan total data bersih. Ini memeriksa **konsistensi penjumlahan**, bukan validitas label setiap artikel.

## 5. Pemeriksaan proporsi train/validation/test

| Subset | Jumlah tercatat | Persentase aktual | Target |
|---|---:|---:|---:|
| Train | 60.914 | 80,01% | 80% |
| Validation | 7.616 | 10,00% | 10% |
| Test | 7.604 | 9,99% | 10% |
| **Total** | **76.134** | **100,00%** | **100%** |

**Pemeriksaan:** 60.914 + 7.616 + 7.604 = **76.134**. Selisih kecil terhadap proporsi 80:10:10 dapat terjadi karena jumlah baris harus berupa bilangan bulat. **Secara jumlah dan rasio, pembagian data konsisten dengan target 80:10:10.**

## 6. Batasan pemeriksaan dan tindak lanjut

Pemeriksaan ini **belum** membuktikan:

1. Ketiadaan teks identik atau hampir identik di antara `train.csv`, `val.csv`, dan `test.csv` (*data leakage*).
2. Konsistensi distribusi label dan sumber **di masing-masing** subset; `ringkasan.txt` hanya memuat jumlah keseluruhan per sumber.
3. Tidak adanya sampel dari data uji eksternal yang turut masuk pelatihan. Pada ringkasan terbaru, `hoaxnews_irfanqs` tercantum dalam gabungan, berbeda dari rencana Progress 2 yang memisahkannya sebagai data uji eksternal.
4. Validitas klasifikasi bahasa Indonesia serta ketepatan label setiap sampel.

Untuk memverifikasi poin-poin tersebut, seluruh CSV perlu dibaca dan diuji dengan skrip audit terpisah; hasilnya harus didokumentasikan sebelum dinyatakan selesai. Selain itu, perubahan dari split 70:15:15 pada Progress 2 menjadi 80:10:10 pada Progress 3 perlu dijelaskan dalam laporan.

## 7. Kesimpulan

Berdasarkan pemeriksaan independen terhadap angka pada `ringkasan.txt`, jumlah data dibuang, total data bersih, total label, total berdasarkan sumber, dan pembagian data **konsisten secara aritmetika**. Dataset terbaru memuat **76.134** baris dengan **39.090** fakta dan **37.044** hoaks, serta pembagian **60.914 / 7.616 / 7.604** yang mendekati **80:10:10**. Kesimpulan ini terbatas pada konsistensi statistik tertulis dan **tidak** berarti dataset sudah bebas kebocoran antarsplit.

## 8. Bukti kontribusi untuk laporan

**Pekerjaan Laura Aurelia:** melakukan verifikasi independen konsistensi statistik hasil penggabungan dataset, pemeriksaan total per label dan per sumber, perhitungan ulang proporsi train/validation/test, serta mendokumentasikan batasan audit.

**Berkas hasil:** `docs/validasi_pembagian_dataset.md` (isi dokumen ini).  
**Bukti commit Laura:** isi tautan commit setelah file dibuat, diperiksa, dan benar-benar di-*push* dari akun Laura sendiri.
