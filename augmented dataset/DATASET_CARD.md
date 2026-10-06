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
- data-augmentation
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
  - name: augmented_full
    num_bytes: 5000000
    num_examples: 500
  - name: 5_fold_stratified_cv
    num_folds: 5
    examples_per_fold: 100
    train_per_fold: 400
    val_per_fold: 100
---

# Dataset Card: Augmented Digit Tulisan Tangan (0–9)

## 1. Ringkasan

Dataset ini merupakan **hasil augmentasi terkendali (label-preserving) dari 100
citra digit tulisan tangan asli** menjadi **500 citra siap latih** dengan
komposisi seimbang sempurna, yaitu **50 citra untuk setiap kelas digit 0–9**.

Augmentasi dibatasi hanya pada transformasi yang relevan dengan domain
tulisan tangan: variasi geometri kecil (rotasi, translasi, skala, shear,
deformasi elastis) dan variasi fotometri kamera (brightness, kontras, gradien
pencahayaan, blur, noise, rekompresi JPEG). Transformasi yang mengubah makna
digit (flip horizontal/vertikal, rotasi besar, inversi warna) **tidak digunakan**
karena akan menghasilkan label yang salah atau keluar dari domain data.

Tujuan utama dataset ini adalah mengatasi keterbatasan *small-sample learning*
pada data asli (10 citra per kelas) sehingga pelatihan model Machine Learning
klasik maupun Deep Learning menjadi lebih stabil dan dapat divalidasi dengan
skema **5-Fold Stratified Cross-Validation**.

## 2. Hubungan dengan Dataset Asli

| Aspek | Dataset Asli | Augmented Dataset |
|---|---|---|
| Jumlah citra | 100 | 500 |
| Citra per kelas | 10 | 50 |
| Resolusi | ± 2.100 × 3.000 px (RGB) | 128 × 128 px (grayscale) |
| Format | JPEG (foto kamera) | JPEG quality 95 |
| Lokasi | `Dataset/Labeled/` | `Dataset/augmented dataset/` |
| Citra novel | 100 citra tulisan berbeda | 500 citra (100 rupa unik + 400 varian) |
| Rasio augmentasi | — | ×5 (1 kanonik + 4 varian per citra) |

**Catatan penting:** setiap kelas memiliki tepat **10 rupa tulisan unik**
(berasal dari 10 citra sumber), masing-masing direplikasi menjadi 5 berkas
(1 kanonik + 4 varian augmentasi). Augmentasi menambah *variasi permukaan*,
bukan menambah *keragaman tulisan baru*.

## 3. Struktur dan Distribusi

```text
Dataset/
├── Labeled/               # 100 citra asli (10 per kelas) - sumber augmentasi
├── augmented dataset/     # <- dataset ini
│   ├── 0/ … 9/            # 50 berkas .jpg per kelas (ImageFolder format)
│   ├── _metadata/
│   │   ├── metadata.csv   # jejak tiap berkas ke citra sumber + parameter augmentasi
│   │   ├── manifest.json  # konfigurasi, seed, sidik jari dataset sumber (SHA-256)
│   │   └── folds.json     # partisi 5-fold stratified deterministik
│   ├── DATASET_CARD.md
│   └── README.md
├── Raw/
├── Docs/
└── code/                  # generator + Docker Compose
```

| Label | Digit | Berkas | Jumlah | Persentase |
|:---:|:---|:---|:---:|:---:|
| `0` | Nol | `0/0_001.jpg` – `0_050.jpg` | 50 | 10% |
| `1` | Satu | `1/1_001.jpg` – `1_050.jpg` | 50 | 10% |
| `2` | Dua | `2/2_001.jpg` – `2_050.jpg` | 50 | 10% |
| `3` | Tiga | `3/3_001.jpg` – `3_050.jpg` | 50 | 10% |
| `4` | Empat | `4/4_001.jpg` – `4_050.jpg` | 50 | 10% |
| `5` | Lima | `5/5_001.jpg` – `5_050.jpg` | 50 | 10% |
| `6` | Enam | `6/6_001.jpg` – `6_050.jpg` | 50 | 10% |
| `7` | Tujuh | `7/7_001.jpg` – `7_050.jpg` | 50 | 10% |
| `8` | Delapan | `8/8_001.jpg` – `8_050.jpg` | 50 | 10% |
| `9` | Sembilan | `9/9_001.jpg` – `9_050.jpg` | 50 | 10% |
| **Total** | **10 Kelas** | **500 Berkas JPEG** | **500** | **100%** |

## 4. Konvensi Penamaan dan Asal Berkas

Pola nama: `<label>_<index>.jpg` dengan `index` 001–050.

- `index = 001` → citra **kanonik asli** (hasil pra-pemrosesan, tanpa augmentasi).
- `index ≥ 002` → **varian augmentasi**, dipetakan bergiliran (round-robin) ke
  salah satu dari 10 citra sumber kelas tersebut.

Hubungan berkas → sumber dan parameter augmentasi tercatat lengkap di
`_metadata/metadata.csv` dengan kolom:

| Kolom | Deskripsi | Contoh |
|---|---|---|
| `file_name` | Nama berkas keluaran | `4_017.jpg` |
| `relative_path` | Jalur relatif dari root dataset | `4/4_017.jpg` |
| `label` | Label kelas | `4` |
| `class_name` | Nama semantik kelas | `Digit 4` |
| `type` | `original` atau daftar parameter augmentasi | `rot_deg=8.49;…;jpeg_quality=88` |
| `source_file` | Berkas asli di `Dataset/Labeled/` | `4_3.jpg` |
| `variant_order` | Urutan varian dari sumber tersebut | `2` |
| `rng_seed` | Seed RNG deterministik varian ini | `42000129` |
| `width`, `height`, `mode` | Dimensi & mode citra | `128`, `128`, `L` |

## 5. Protokol Augmentasi

### 5.1 Pra-pemrosesan kanonik
Grayscale → perataan pencahayaan (estimasi level kertas lokal) → auto-crop
digit dengan **ambang relatif terhadap level kertas** (kertas × 0,85) →
margin 12% pada kanvas persegi putih → resize 128 × 128 (LANCZOS) →
normalisasi kontras dua titik (kertas → 255, tinta → gelap).

### 5.2 Transformasi augmentasi

| Transformasi | Rentang | Probabilitas | Relevansi domain |
|---|---|---|---|
| Rotasi | ±10° | 100% | kemiringan tulisan/sudut kamera |
| Translasi | ±8% | 100% | posisi digit dalam bidang pandang |
| Skala | 0,92–1,08 | 100% | jarak pemotretan |
| Shear | ±5° | 100% | gaya goresan pena |
| Deformasi elastis | α 0,75; σ 6 | 25% | variasi halus bentuk goresan |
| Brightness | ±0,10 | 100% | pencahayaan ruangan |
| Contrast | ±0,12 | 100% | kontras kertas–tinta |
| Gradien iluminasi | ≤ 0,08 | 20% | cahaya tidak rata |
| Gaussian blur | σ 0,35–0,9 | 30% | fokus kamera |
| Gaussian noise | σ 1,5–6 | 30% | noise sensor |
| JPEG rekompresi | q 72–96 | 35% | kompresi berkas |

### 5.3 Transformasi yang dilarang (pelanggaran label)

Horizontal flip, vertical flip (`6`↔`9`, `2`↔`5`), rotasi > 15°,
crop/warp agresif, dan inversi warna. Daftar ini tersimpan di
`_metadata/manifest.json` pada kunci `forbidden_transforms`.

## 6. Reproduksibilitas

| Parameter | Nilai |
|---|---|
| Generator | `Dataset/code/` (Python 3.12, Pillow + NumPy + SciPy) |
| Versi generator | 1.0.0 |
| Seed | 42 |
| Target per kelas | 50 |
| Kualitas JPEG | 95 |
| Dependensi | terkunci di `Dataset/code/uv.lock`, dijalankan via Docker Compose |

Karena setiap varian memakai seed turunan deterministik
(`seed × 1.000.003 + label × 10.007 + index`), regenerasi menghasilkan berkas
**identik byte-for-byte** dan tidak bergantung pada jumlah thread. Sidik jari
dataset sumber (SHA-256 dari nama, ukuran, dan waktu ubah berkas) disimpan di
`manifest.json` sebagai bukti versi data yang dipakai.

## 7. Kesiapan Machine Learning

- Format `ImageFolder` → langsung dapat dipakai `torchvision.datasets.ImageFolder`.
- Kelas seimbang sempurna (50/50/…/50) → metrik akurasi tidak bias.
- `folds.json` menyediakan partisi 5-fold stratified (100 validasi, 400 latih per fold).
- Sudah diuji dengan verifikasi otomatis: distribusi kelas, integritas berkas,
  keberadaan goresan digit, dan tidak adanya duplikat persis.

## 8. Batasan dan Etika Penggunaan

1. **Bukan data baru.** Hanya 100 rupa tulisan unik yang mendasari 500 berkas.
   Mengklaim akurasi pada 500 citra ini sebagai generalisasi terhadap penulis
   baru adalah **tidak valid**; gunakan pengujian pada citra asli/penulis lain.
2. **Risiko kebocoran data.** Varian dari satu citra sumber dapat tersebar di
   fold berbeda. Untuk evaluasi paling ketat, split pada tingkat `source_file`.
3. **Domain terbatas.** Spidol hitam 2,0 mm di kertas HVS putih, pemotretan
   tegak lurus. Pensil, pulpen tipis, atau kertas berwarna berada di luar domain.
4. **Tidak digunakan untuk identifikasi seseorang.** Dataset hanya berisi digit
   tulisan tangan, bukan data pribadi.
5. **Metadata wajib disertakan** ketika dataset ini dibagikan atau dilaporkan,
   agar asal-usul setiap berkas dapat ditelusuri.

## 9. Metadata & Creator

- **Penyusun**: Putra Whyra Pratama S. (NIM: 2310817210029)
- **Program Studi**: S1 Teknologi Informasi, Fakultas Teknik, Universitas Lambung Mangkurat
- **Mata Kuliah**: Machine Learning
- **Dosen Pengampu**: Achmad Mujaddid Islami, S.Kom, M.Kom
- **Tahun Akademik**: 2025/2026
- **Lokasi Pengambilan Data**: Banjarmasin, Kalimantan Selatan, Indonesia
