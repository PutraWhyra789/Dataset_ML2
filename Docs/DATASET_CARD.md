---
language:
- id
- en
license: unknown
task_categories:
- image-classification
task_ids:
- handwritten-digit-recognition
- optical-character-recognition
tags:
- handwritten-digits
- computer-vision
- pytorch
- small-dataset-benchmark
- stratified-cross-validation
size_categories:
- n<1K
dataset_info:
  features:
  - name: image
    dtype: image
  - name: label
    dtype:
      class_label:
        names:
          '0': '0'
          '1': '1'
          '2': '2'
          '3': '3'
          '4': '4'
          '5': '5'
          '6': '6'
          '7': '7'
          '8': '8'
          '9': '9'
  splits:
  - name: full_dataset
    num_bytes: 17646592
    num_examples: 100
  - name: 5_fold_stratified_cv
    num_folds: 5
    examples_per_fold: 20
    train_per_fold: 80
    val_per_fold: 20
---

# Dataset Card: Citra Digit Tulisan Tangan (0–9)

## 1. Dataset Summary & Overview

Dataset ini merupakan kumpulan **100 citra digital asli angka tulisan tangan (0 sampai 9)** yang dikumpulkan secara mandiri (*primary data acquisition*) dari dokumen fisik kertas HVS menggunakan kamera ponsel pintar beresolusi tinggi. Dataset ini dirancang untuk penelitian komparatif pemodelan visi komputer (*Computer Vision*) berskala data kecil (*small-sample learning*), membandingkan algoritma *Machine Learning* klasik (SVM, Random Forest) dengan arsitektur *Deep Learning* (Custom CNN, ResNet18, MobileNetV2) menggunakan skema validasi **5-Fold Stratified Cross-Validation**.

> **Catatan Pembaruan Rencana (150 → 100 Citra):**
> Pada rancangan proposal awal (*Tugas 1 Data Acquisition*), kuota teoretis direncanakan berjumlah 150 data (15 citra per kelas). Pada pelaksanaan aktual dan fase penjaminan mutu (*Quality Assurance*), dataset difinalisasi menjadi **100 citra terpilih dan tervalidasi penuh (tepat 10 citra per kelas digit 0–9)**. Penyesuaian ini mengeliminasi sampel berkualitas rendah/kabur serta menghasilkan partisi yang tepat dan seimbang untuk skema *5-Fold Stratified Cross-Validation* (20 sampel per fold: 16 data latih + 4 data uji per kelas proporsional).

---

## 2. Dataset Structure

### 2.1 File Directory Organization
Dataset tersusun dalam hierarki partisi folder berbasis kelas (*PyTorch ImageFolder format*):

```text
Dataset/
├── Labeled/            # 100 citra asli terlabel (10 per kelas) = dataset utama
│   ├── 0/              # 10 citra digit 0 (0_1.jpg s.d. 0_10.jpg)
│   ├── 1/              # 10 citra digit 1 (1_1.jpg s.d. 1_10.jpg)
│   ├── 2/              # 10 citra digit 2 (2_1.jpg s.d. 2_10.jpg)
│   ├── 3/              # 10 citra digit 3 (3_1.jpg s.d. 3_10.jpg)
│   ├── 4/              # 10 citra digit 4 (4_1.jpg s.d. 4_10.jpg)
│   ├── 5/              # 10 citra digit 5 (5_1.jpg s.d. 5_10.jpg)
│   ├── 6/              # 10 citra digit 6 (6_1.jpg s.d. 6_10.jpg)
│   ├── 7/              # 10 citra digit 7 (7_1.jpg s.d. 7_10.jpg)
│   ├── 8/              # 10 citra digit 8 (8_1.jpg s.d. 8_10.jpg)
│   └── 9/              # 10 citra digit 9 (9_1.jpg s.d. 9_10.jpg)
├── Raw/                # 100 citra mentah sebelum pelabelan/pemotongan
├── Docs/               # dokumentasi dataset (berkas ini)
├── augmented dataset/  # 500 citra hasil augmentasi (50 per kelas) + metadata
└── code/               # generator augmentasi + Docker Compose (uv di dalam image)
```

### 2.2 Class Distribution
Distribusi dataset seimbang sempurna (*perfect class balance*):

| Label Numerik | Kategori Digit | Rentang File | Jumlah Sampel | Persentase |
|:---:|:---|:---|:---:|:---:|
| `0` | Digit Nol (0) | `Labeled/0/0_1.jpg` s.d. `0_10.jpg` | 10 Citra | 10.0% |
| `1` | Digit Satu (1) | `Labeled/1/1_1.jpg` s.d. `1_10.jpg` | 10 Citra | 10.0% |
| `2` | Digit Dua (2) | `Labeled/2/2_1.jpg` s.d. `2_10.jpg` | 10 Citra | 10.0% |
| `3` | Digit Tiga (3) | `Labeled/3/3_1.jpg` s.d. `3_10.jpg` | 10 Citra | 10.0% |
| `4` | Digit Empat (4) | `Labeled/4/4_1.jpg` s.d. `4_10.jpg` | 10 Citra | 10.0% |
| `5` | Digit Lima (5) | `Labeled/5/5_1.jpg` s.d. `5_10.jpg` | 10 Citra | 10.0% |
| `6` | Digit Enam (6) | `Labeled/6/6_1.jpg` s.d. `6_10.jpg` | 10 Citra | 10.0% |
| `7` | Digit Tujuh (7) | `Labeled/7/7_1.jpg` s.d. `7_10.jpg` | 10 Citra | 10.0% |
| `8` | Digit Delapan (8) | `Labeled/8/8_1.jpg` s.d. `8_10.jpg` | 10 Citra | 10.0% |
| `9` | Digit Sembilan (9) | `Labeled/9/9_1.jpg` s.d. `9_10.jpg` | 10 Citra | 10.0% |
| **Total** | **10 Kelas Digit** | **100 Berkas JPEG** | **100 Citra** | **100.0%** |

---

## 3. Technical Specifications & Data Fields

### 3.1 Parameter Teknis Citra Mentah
- **Format Berkas**: JPEG (`.jpg` / JFIF standard compression).
- **Ruang Warna**: RGB (3 kanal, kedalaman warna 24-bit).
- **Dimensi Asli**: Rata-rata 2093.2 × 3000 piksel (Lebar: 2036–2161 px, Tinggi: 3000 px).
- **Ukuran File**: 129.1 KB – 262.3 KB (rata-rata 168.3 KB per citra, total dataset ~16.8 MB).
- **Media Penulisan**: Kertas HVS putih polos 80 gsm.
- **Instrumen Penulisan**: Spidol hitam pekat (*marker* 2.0 mm).
- **Kondisi Cahaya**: Cahaya alami ruangan difus (500–800 Lux), tanpa lampu kilat (*no flash*).
- **Sudut & Jarak Pemotretan**: 90° tegak lurus (*top-down*), jarak 20–25 cm.

### 3.2 Data Fields & Metadata Schema
Setiap record citra dapat direpresentasikan dalam skema metadata berikut:

| Nama Kolom | Tipe Data | Deskripsi | Contoh Nilai |
|---|---|---|---|
| `sample_id` | String | Pengidentifikasi unik sampel | `DIGIT_5_10` |
| `class_label` | Integer | Label kelas target (0–9) | `5` |
| `class_name` | String | Nama semantik kelas digit | `Digit Lima (5)` |
| `file_name` | String | Nama berkas citra asli | `5_10.jpg` |
| `relative_path`| String | Lokasi relatif berkas dalam repositori | `Labeled/5/5_10.jpg` |
| `width` | Integer | Lebar spasial piksel asli | `2093` |
| `height` | Integer | Tinggi spasial piksel asli | `3000` |
| `channels` | Integer | Jumlah kanal warna | `3` (RGB) |
| `file_size_kb` | Float | Ukuran berkas dalam Kilobytes | `155.7` |
| `qa_status` | String | Status verifikasi mutu | `PASSED_QA_VERIFIED` |

---

## 4. Data Acquisition & Quality Assurance Protocol

Akuisisi data dilakukan melalui 5 fase terstandarisasi:
1. **Perencanaan Kisi (Grid Planning)**: Pembuatan area kisi $5 \times 5$ cm pada kertas HVS putih 80 gsm untuk memusatkan posisi penulisan karakter.
2. **Protokol Penulisan**: Penulisan angka 0–9 dengan spidol hitam 2.0 mm dengan variasi goresan wajar manusia (inklinasi wajar, gaya penulisan alternatif untuk angka 1, 4, 7, 9).
3. **Digitalisasi Sensor**: Menggunakan kamera smartphone 12 MP ($f/1.8$) dengan sudut 90° tegak lurus (*top-down*) untuk meminimalkan distorsi perspektif.
4. **Penjaminan Mutu (QA)**:
   - *Kriteria Inklusi*: Satu karakter utuh terpusat, kontras tajam (rasio kontras Weber > 0.70), terbaca jelas tanpa ambiguitas.
   - *Kriteria Eksklusi*: Mengeliminasi citra kabur (*blur*), terpotong, atau berbayangan tebal.
5. **Katalogisasi & Strukturisasi**: Penamaan terstandarisasi `{digit}_{index}.jpg` dan pengelompokan ke subfolder masing-masing digit.

---

## 5. Machine Learning Readiness & Benchmarks

Dataset telah diuji secara komprehensif menggunakan protokol **5-Fold Stratified Cross-Validation** dengan augmentasi data offline pada fold pelatihan (rotasi $\pm 15^\circ$, translasi 8%, zoom 92%–108%).

## 6. Metadata & Creator Information

- **Penyusun**: Putra Whyra Pratama S. (NIM: 2310817210029)
- **Program Studi**: S1 Teknologi Informasi, Fakultas Teknik, Universitas Lambung Mangkurat
- **Mata Kuliah**: Machine Learning
- **Dosen Pengampu**: Achmad Mujaddid Islami, S.Kom, M.Kom
- **Tahun Akademik**: 2025/2026
- **Lokasi Pengambilan**: Banjarmasin, Kalimantan Selatan, Indonesia
