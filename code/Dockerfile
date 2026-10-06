# syntax=docker/dockerfile:1
#
# Image reproducible untuk generator dataset augmentasi digit tulisan tangan.
# uv TIDAK perlu terpasang di komputer pengguna: image ini membawa uv sendiri
# sehingga `docker compose run --rm augmentasi` langsung dapat dijalankan.
#
# uv diambil dari image resmi (distroless) sesuai dokumentasi resmi uv:
#   https://docs.astral.sh/uv/guides/integration/docker/#installing-uv
# Tag versi penuh dikunci agar build reproducible. Ganti tag ini (mis. ke
# 0.12) hanya bila memang ingin memutakhirkan uv.

FROM ghcr.io/astral-sh/uv:0.11.21 AS uv

FROM python:3.12-slim-bookworm

LABEL org.opencontainers.image.title="digit-augmentation" \
      org.opencontainers.image.description="Generator augmentasi dataset digit tulisan tangan 0-9 (UAS Machine Learning)" \
      org.opencontainers.image.source="https://github.com/astral-sh/uv"

# Salin binary uv + uvx dari image resmi (tanpa perlu install apa pun).
COPY --from=uv /uv /uvx /bin/

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    UV_LINK_MODE=copy \
    UV_PROJECT_ENVIRONMENT=/opt/venv \
    UV_COMPILE_BYTECODE=1 \
    UV_PYTHON_DOWNLOADS=never \
    UV_CACHE_DIR=/opt/uv-cache \
    PATH="/opt/venv/bin:$PATH"

WORKDIR /app

# 1) Lapisan dependensi: hanya ter-invalidate saat pyproject/uv.lock berubah.
COPY pyproject.toml uv.lock ./
RUN uv sync --frozen --no-install-project

# 2) Lapisan kode: perubahan kode tidak mengulang instalasi dependensi.
#    Paket `digit_aug` berada tepat di root proyek sehingga dapat diimpor
#    langsung oleh `uv run python -m digit_aug.generate` (sama seperti di host).
COPY digit_aug ./digit_aug
COPY tests ./tests

# Smoke test saat build: modul harus bisa diimpor di lingkungan terkunci.
RUN python -c "import digit_aug, numpy, PIL, scipy; print('digit_aug', digit_aug.__version__, 'siap')"
RUN uv run --frozen pytest -q -p no:cacheprovider

# Entry point default: bangkitkan dataset augmentasi 50 citra/kelas.
# Folder dataset di-mount lewat compose.yaml: ../Labeled -> /data/Labeled
# dan ../augmented dataset -> /data/augmented dataset (/data setara root dataset).
ENTRYPOINT ["uv", "run", "--frozen", "python", "-m", "digit_aug.generate"]
CMD ["--source", "/data/Labeled", "--output", "/data/augmented dataset", "--per-class", "50", "--seed", "42"]
