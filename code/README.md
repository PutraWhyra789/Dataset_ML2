# Generator Dataset Augmentasi — Digit Tulisan Tangan 0–9

Kode untuk membangkitkan folder [`../augmented dataset/`](../augmented%20dataset/README.md)
dari dataset asli [`../Labeled/`](../Labeled).

Lokasi: **`Dataset/code/`** — bersebelahan dengan `Dataset/Labeled/` (data asli)
dan `Dataset/augmented dataset/` (hasil).

Dibuat agar **mudah direproduksi di komputer lain tanpa memasang uv**: seluruh
perintah dijalankan melalui Docker Compose, dan `uv` sudah tersedia di dalam
image (diambil dari image resmi `ghcr.io/astral-sh/uv`).

## Prasyarat

| Cara | Yang dibutuhkan |
|---|---|
| **Cara utama (Docker)** | Docker Desktop / Docker Engine + Docker Compose v2 |
| Cara alternatif (lokal) | [uv](https://docs.astral.sh/uv/) — opsional, untuk pengembangan |

Tidak perlu memasang Python, NumPy, Pillow, atau SciPy secara manual.

## Struktur Proyek

```text
Dataset/
├── Labeled/                  # 100 citra asli (10 per kelas) = input
├── augmented dataset/        # 500 citra hasil = output
└── code/
    ├── compose.yaml          # service: augmentasi, verifikasi, test, shell
    ├── .env.example          # contoh override lokasi folder dataset
    ├── Dockerfile            # image Python 3.12 + uv (uv tidak perlu di host)
    ├── pyproject.toml        # dependensi terkunci (versi persis)
    ├── uv.lock               # lockfile lintas-platform untuk instalasi reproducible
    ├── digit_aug/
    │   ├── augment.py        # kebijakan & pipeline transformasi
    │   ├── generate.py       # CLI pembangkit dataset
    │   └── verify.py         # CLI pemeriksaan mutu dataset hasil
    └── tests/test_augment.py # unit test pipeline
```

## Cara Menjalankan (Docker Compose)

Jalankan dari dalam folder `Dataset/code/`:

```bash
# 1) Bangun image (sekali saja; uv ikut terpasang di dalam image)
docker compose build

# 2) Bangkitkan dataset augmentasi: 50 citra/kelas -> 500 citra
docker compose run --rm augmentasi

# 3) Verifikasi hasil (distribusi kelas, integritas berkas, duplikat)
docker compose run --rm verifikasi

# 4) Jalankan unit test pipeline
docker compose run --rm test
```

Setelah langkah 2 selesai, folder `Dataset/augmented dataset/` akan terisi
500 citra `128 × 128` grayscale (50 per kelas) beserta folder `_metadata/`.

### Mengubah parameter tanpa mengedit berkas

Semua argumen CLI dapat ditambahkan setelah nama service:

```bash
# 100 citra per kelas (total 1000) dengan seed berbeda
docker compose run --rm augmentasi --per-class 100 --seed 2025

# Resolusi keluaran lebih besar dan kualitas JPEG lebih tinggi
docker compose run --rm augmentasi --per-class 50 --size 224 --jpeg-quality 98

# Uji cepat: hanya 2 citra sumber per kelas, 12 citra keluaran per kelas
docker compose run --rm augmentasi --limit-sources 2 --per-class 12

# Lihat semua opsi
docker compose run --rm augmentasi --help
```

> Perintah `verifikasi` juga menerima `--expect-per-class N` agar sesuai bila
> Anda mengubah target jumlah citra:

```bash
docker compose run --rm verifikasi --expect-per-class 100
```

### Lapisan data (volume)

`compose.yaml` memasang dua folder dari host. Bawaannya relatif terhadap
`Dataset/code/`:

| Host | Di dalam container | Akses |
|---|---|---|
| `../Labeled` | `/data/Labeled` | read-only (data asli tidak pernah diubah) |
| `../augmented dataset` | `/data/augmented dataset` | read-write (hasil langsung muncul di host) |

Bila lokasi folder berbeda, salin `.env.example` menjadi `.env` lalu ubah
`DATASET_SOURCE` / `DATASET_OUTPUT` — tidak perlu menyentuh `compose.yaml`:

```bash
cp .env.example .env
# DATASET_SOURCE=../Labeled
# DATASET_OUTPUT=../augmented dataset
```

Karena hasil ditulis ke folder host, hasilnya langsung dapat dibuka,
diperiksa, dan dipakai untuk pelatihan tanpa menyalin berkas keluar container.

### Shell interaktif

Bila perlu bereksperimen dengan `uv` di dalam container:

```bash
docker compose run --rm shell
# di dalam container:
#   uv run --frozen python -c "import digit_aug; print(digit_aug.__version__)"
#   uv run --frozen python -m digit_aug.generate --help
```

## Cara Menjalankan Tanpa Docker (opsional, butuh uv)

```bash
cd Dataset/code
uv sync --frozen                                    # instalasi sesuai uv.lock
uv run --frozen python -m digit_aug.generate \
    --source ../Labeled \
    --output "../augmented dataset" \
    --per-class 50 --seed 42
uv run --frozen python -m digit_aug.verify --dataset "../augmented dataset"
uv run --frozen pytest -q
```

Paket `digit_aug` berada tepat di root proyek, sehingga perintah `uv run`
di atas bekerja tanpa mengatur `PYTHONPATH` (perilaku sama persis dengan di
dalam container).

## Opsi CLI `digit_aug.generate`

| Opsi | Default | Keterangan |
|---|---|---|
| `--source` | `../Labeled` | Folder dataset asli berisi subfolder kelas `0`–`9` |
| `--output` | `../augmented dataset` | Folder keluaran (isi lama pada folder kelas dibersihkan otomatis) |
| `--per-class` | `50` | Target jumlah citra akhir per kelas |
| `--seed` | `42` | Seed RNG; menentukan seluruh varian augmentasi |
| `--size` | `128` | Sisi citra kanonik keluaran (piksel) |
| `--jpeg-quality` | `95` | Kualitas JPEG berkas keluaran |
| `--workers` | `0` (otomatis) | Jumlah thread paralel |
| `--limit-sources` | `0` (semua) | Batasi jumlah citra sumber per kelas untuk uji cepat |

Default `--source` dan `--output` mengasumsikan perintah dijalankan dari
`Dataset/code/`, sehingga mengarah ke `Dataset/Labeled/` dan
`Dataset/augmented dataset/`.

## Kebijakan Augmentasi

Ringkasan lengkap beserta alasan relevansinya ada di
[`../augmented dataset/README.md`](../augmented%20dataset/README.md).
Inti kebijakannya:

- **Dipakai:** rotasi ±10°, translasi ±8%, skala 0,92–1,08, shear ±5°,
  deformasi elastis (25%), brightness ±0,10, kontras ±0,12, gradien cahaya (20%),
  blur Gauss σ 0,35–0,9 (30%), noise Gauss σ 1,5–6 (30%), rekompresi JPEG (35%).
- **Dilarang:** flip horizontal/vertikal, rotasi > 15°, crop/warp agresif,
  inversi warna — semuanya mengubah makna digit atau keluar dari domain data.

## Reproduksibilitas dan Audit

- `uv.lock` mengunci versi dependensi persis (`numpy==2.4.6`, `pillow==12.2.0`,
  `scipy==1.17.1`, `pytest==9.0.2`), sehingga hasil sama di mesin berbeda.
- Setiap varian memakai seed turunan `seed × 1.000.003 + label × 10.007 + index`,
  sehingga hasil **identik byte-for-byte** dan tidak bergantung urutan eksekusi
  atau jumlah thread.
- `_metadata/metadata.csv` mencatat parameter augmentasi **yang benar-benar
  diterapkan** pada setiap berkas (bukan sekadar rentang kebijakan).
- `_metadata/manifest.json` menyimpan kebijakan, daftar transformasi terlarang,
  versi Python, dan sidik jari SHA-256 dataset sumber.
- `_metadata/folds.json` menyediakan partisi 5-fold stratified deterministik.

## Pemecahan Masalah

| Gejala | Penyebab & solusi |
|---|---|
| `Folder sumber tidak ditemukan` | Jalankan perintah dari dalam folder `Dataset/code/`, atau arahkan `--source` ke folder yang benar. |
| `--per-class` lebih kecil dari jumlah citra sumber | Turunkan `--limit-sources` atau naikkan `--per-class`. |
| Container menulis berkas dengan pemilik `root` (Linux) | Jalankan `docker compose run --rm --user "$(id -u):$(id -g)" augmentasi`. |
| Ingin hasil berbeda tanpa mengubah data asli | Ubah `--seed`; seed berbeda menghasilkan varian berbeda. |
| `docker compose` tidak dikenali | Pastikan Docker Compose **v2** (`docker compose version`), bukan `docker-compose` v1. |
| Ingin hasil di folder lain | Salin `.env.example` ke `.env` dan ubah `DATASET_OUTPUT`. |
