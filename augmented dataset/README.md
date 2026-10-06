# Augmented Dataset — Digit Tulisan Tangan 0–9

Dataset augmentasi hasil pengembangan dari `Dataset/Labeled` (100 citra asli, 10 per kelas)
menjadi **500 citra siap latih (50 citra per kelas, 10 kelas)**.

| Atribut | Nilai |
|---|---|
| Jumlah citra | 500 (50 per kelas 0–9) |
| Resolusi | 128 × 128 piksel |
| Mode warna | Grayscale (`L`) |
| Format | JPEG, quality 95 |
| Rasio augmentasi | 1 asli kanonik + 4 varian per citra sumber |
| Seed | 42 (deterministik, byte-for-byte) |
| Struktur | `ImageFolder`: `<label>/<label>_<index>.jpg` |

## Struktur Folder

Folder ini berada di dalam `Dataset/`, bersebelahan dengan data asli `Labeled/`
dan folder kode `code/`:

```text
Dataset/
├── Labeled/                  # 100 citra asli (10 per kelas) - sumber augmentasi
├── augmented dataset/        # <- folder ini
│   ├── 0/  0_001.jpg … 0_050.jpg   # 50 citra digit 0 (0_001 = citra kanonik asli)
│   ├── 1/  1_001.jpg … 1_050.jpg
│   ├── …
│   ├── 9/  9_001.jpg … 9_050.jpg
│   ├── _metadata/
│   │   ├── metadata.csv    # 500 baris: sumber, jenis, seed, dan parameter augmentasi tiap berkas
│   │   ├── manifest.json   # konfigurasi lengkap + sidik jari dataset sumber (SHA-256)
│   │   └── folds.json      # partisi 5-Fold Stratified CV siap pakai (4 berkas uji per kelas per fold)
│   ├── DATASET_CARD.md
│   └── README.md
├── Raw/                      # 100 citra mentah (sebelum pelabelan)
├── Docs/
└── code/                     # generator augmentasi + Docker Compose
```

> Setiap berkas `<label>_001.jpg` adalah citra kanonik **tanpa augmentasi**
> (hanya pra-pemrosesan geometri/fotometri), sehingga dataset tetap memuat
> rupa asli tulisan. Berkasi `_002` sampai `_050` adalah varian augmentasi.

## Pra-pemrosesan Kanonik (dilakukan sebelum augmentasi)

Foto kamera ponsel (≈ 2100 × 3000 px, RGB) diubah ke kanvas seragam agar
pelatihan stabil dan perbandingan antar model adil:

1. **Grayscale** — warna tidak membawa informasi kelas.
2. **Perataan pencahayaan** — citra dibagi dengan estimasi level kertas lokal
   (persentil tinggi lokal), menghapus gradien cahaya, vignette, dan bayangan.
3. **Auto-crop digit** — ambang tinta *relatif terhadap level kertas*
   (kertas × 0,85). Ambang Otsu global tidak dipakai karena histogram foto
   didominasi kertas sehingga gagal memisahkan digit.
4. **Margin 12% + kanvas persegi putih** — rasio aspek digit dipertahankan.
5. **Resize 128 × 128 (LANCZOS)**.
6. **Normalisasi kontras dua titik** — kertas menjadi putih murni (255), tinta
   menjadi gelap, dengan penguatan terbatas agar kedalaman goresan asli terjaga.

## Kebijakan Augmentasi (hanya transformasi yang relevan)

Transformasi dipilih agar **label tetap valid** untuk digit tulisan tangan dan
meniru variasi nyata saat pengambilan data ulang dengan kamera ponsel.

| Transformasi | Rentang | Probabilitas | Alasan relevansi |
|---|---|---|---|
| Rotasi | ±10° | 100% | kemiringan tulisan & sudut kamera |
| Translasi | ±8% sisi | 100% | posisi digit di bidang pandang |
| Skala | 0,92–1,08 | 100% | variasi jarak pemotretan |
| Shear | ±5° | 100% | variasi gaya goresan pena |
| Deformasi elastis | α=0,75; σ=6 | 25% | variasi halus bentuk goresan antar penulisan |
| Brightness | ±0,10 | 100% | variasi pencahayaan ruangan |
| Contrast | ±0,12 | 100% | variasi kontras kertas-tinta |
| Gradien iluminasi | ≤0,08 | 20% | pencahayaan tidak rata di atas kertas |
| Gaussian blur | σ 0,35–0,9 | 30% | variasi fokus kamera ponsel |
| Gaussian noise | σ 1,5–6 | 30% | noise sensor kamera ponsel |
| JPEG rekompresi | quality 72–96 | 35% | variasi kompresi berkas |

### Transformasi yang sengaja DILARANG

| Transformasi | Alasan |
|---|---|
| Horizontal flip | `2`, `3`, `4`, `5`, `7` berubah menjadi bentuk yang tidak ada di domain sekaligus menukar identitas digit |
| Vertical flip | `6` ↔ `9` dan `2` ↔ `5` bertukar label |
| Rotasi > 15° | `6`/`9` berpotensi terbaca sebagai digit lain |
| Crop ekstrem / warp berat | dapat memotong atau merusak struktur goresan |
| Inversi warna | menghasilkan domain yang tidak pernah muncul di data nyata |

Semua batas di atas tercatat di `_metadata/manifest.json` (kunci `policy` dan
`forbidden_transforms`) sehingga dapat diaudit.

## Cara Memakai untuk Pelatihan

Struktur folder sudah kompatibel langsung dengan `torchvision.datasets.ImageFolder`:

```python
from torchvision import datasets, transforms

train_tf = transforms.Compose([
    transforms.Resize((128, 128)),
    transforms.ToTensor(),
    transforms.Normalize([0.5], [0.5]),
])

dataset = datasets.ImageFolder("Dataset/augmented dataset/", transform=train_tf)
print(len(dataset), dataset.class_to_idx)   # 500 {'0': 0, '1': 1, ..., '9': 9}
```

### Validasi 5-Fold Stratified CV

`_metadata/folds.json` sudah berisi partisi stratified yang siap dipakai
(seed 42, 4 berkas validasi per kelas per fold = 40 citra validasi, 460 latih):

```python
import json

folds = json.load(open("Dataset/augmented dataset/_metadata/folds.json", encoding="utf-8"))
fold1 = folds["folds"][0]
print(len(fold1["train"]), len(fold1["val"]))   # 460 40
```

> **Catatan metodologis:** berkas asli (`_001`) dan variannya berasal dari citra
> sumber yang sama. Untuk klaim generalisasi yang paling ketat, lakukan split
> pada tingkat **citra sumber** (lihat kolom `source_file` di `metadata.csv`) agar
> varian dari satu tulisan tidak muncul sekaligus di data latih dan data uji.

## Regenerasi Dataset

Dataset ini dihasilkan oleh proyek di folder [`code/`](../code/README.md)
(berada di dalam `Dataset/` yang sama).
Untuk membangkitkan ulang (butuh Docker, uv tidak perlu dipasang):

```bash
cd Dataset/code
docker compose run --rm augmentasi
```

Bila folder hasil berada di lokasi lain, atur lewat berkas `.env`
(salin dari `.env.example`) tanpa mengubah `compose.yaml`.

Hasil regenerasi identik byte-for-byte selama `--seed 42`, `--per-class 50`,
dan dataset sumber tidak berubah (dibuktikan oleh `source_fingerprint_sha256`).

## Sitasi Data Sumber

- **Penyusun**: Putra Whyra Pratama S. (NIM: 2310817210029)
- **Program Studi**: S1 Teknologi Informasi, Fakultas Teknik, Universitas Lambung Mangkurat
- **Mata Kuliah**: Machine Learning — Semester 2025/2026
- **Dosen Pengampu**: Achmad Mujaddid Islami, S.Kom, M.Kom
- **Lokasi Pengambilan**: Banjarmasin, Kalimantan Selatan, Indonesia
- **Data asli**: 100 citra foto kamera ponsel, kertas HVS 80 gsm, spidol hitam 2,0 mm,
  cahaya ruangan difus, pemotretan tegak lurus 90°.
