"""Modul augmentasi citra digit tulisan tangan.

Semua transformasi di sini dipilih agar *label tetap valid* (relevan) untuk
digit 0-9 hasil tulisan tangan, yaitu meniru variasi nyata saat pengambilan
data ulang dengan kamera ponsel:

    - rotasi kecil          -> kemiringan tulisan / sudut kamera
    - translasi & skala     -> posisi & jarak pemotretan
    - shear & elastic warp  -> variasi goresan pena antar penulisan
    - brightness/contrast   -> variasi pencahayaan ruangan
    - blur & noise          -> variasi fokus kamera & noise sensor
    - JPEG recompress       -> variasi kompresi berkas

Transformasi yang DILARANG (merusak makna digit):
    - horizontal flip  (2 -> bentuk tak dikenal, 4/5/7 jadi salah)
    - vertical flip    (6 <-> 9, 2 <-> 5)
    - rotasi besar > 15 derajat
    - crop agresif / warp berat
    - inversi warna (background jadi hitam, di luar domain data)
"""

from __future__ import annotations

import io
import math
from dataclasses import dataclass

import numpy as np
import scipy.ndimage as ndi
from PIL import Image, ImageFilter

# ---------------------------------------------------------------------------
# Konfigurasi kanonik dataset
# ---------------------------------------------------------------------------

CANONICAL_SIZE = 128  # piksel keluaran (persegi)
CLASSES = tuple(str(d) for d in range(10))

# Level kertas setelah perataan pencahayaan; di atas 255 karena piksel di atas
# ambang ini nantinya dipangkas menjadi putih murni oleh _normalize_contrast.
_FLAT_PAPER_LEVEL = 258.0


@dataclass(frozen=True)
class AugmentPolicy:
    """Batas rentang tiap transformasi (konservatif, label-preserving)."""

    max_rotation_deg: float = 10.0
    max_translate: float = 0.08  # fraksi dari sisi citra
    min_scale: float = 0.92
    max_scale: float = 1.08
    max_shear_deg: float = 5.0
    brightness_delta: float = 0.10
    contrast_delta: float = 0.12
    blur_prob: float = 0.30
    blur_sigma_max: float = 0.9
    noise_prob: float = 0.30
    noise_sigma_max: float = 6.0
    jpeg_prob: float = 0.35
    jpeg_quality_min: int = 72
    jpeg_quality_max: int = 96
    elastic_prob: float = 0.25
    elastic_alpha: float = 0.75  # kekuatan deformasi (piksel pada kanvas 128)
    elastic_sigma: float = 6.0
    gradient_prob: float = 0.20
    gradient_strength: float = 0.08


DEFAULT_POLICY = AugmentPolicy()


# ---------------------------------------------------------------------------
# Pra-pemrosesan (dari citra mentah kamera -> kanvas kanonik)
# ---------------------------------------------------------------------------


def _otsu_threshold(gray: np.ndarray) -> int:
    """Threshold Otsu pada histogram 256 bin tanpa dependensi tambahan."""
    hist = np.bincount(gray.ravel(), minlength=256).astype(np.float64)
    total = hist.sum()
    if total == 0:
        return 127
    levels = np.arange(256, dtype=np.float64)
    omega = np.cumsum(hist)
    mu = np.cumsum(hist * levels)
    mu_t = mu[-1]
    denom = omega * (total - omega)
    denom[denom == 0] = np.nan
    sigma_b = (mu_t * omega - total * mu) ** 2 / denom
    sigma_b = np.nan_to_num(sigma_b, nan=-1.0)
    return int(np.argmax(sigma_b))


def _paper_level(gray: np.ndarray) -> float:
    """Estimasi level kertas (persentil tinggi) dari histogram citra."""
    return float(np.percentile(gray, 85.0))


def _ink_threshold(gray: np.ndarray, ratio: float = 0.85) -> int:
    """Threshold tinta relatif terhadap level kertas.

    Otsu global gagal pada foto kamera karena histogram didominasi kertas
    (satu puncak besar) sehingga threshold-nya bisa jatuh di dalam puncak
    kertas dan menandai seluruh lembar sebagai "tinta". Ambang relatif
    (kertas x `ratio`) jauh lebih stabil untuk citra seperti ini.
    """
    paper = _paper_level(gray)
    return int(max(10.0, min(paper * ratio, paper - 12.0)))


def _ink_bbox(
    gray: np.ndarray,
    thresh: int,
    min_fraction: float = 0.0004,
) -> tuple[int, int, int, int] | None:
    """Bounding box piksel tinta (lebih gelap dari threshold).

    Bila ambang terlalu ketat sehingga tinta hampir tak terdeteksi, ambang
    dilonggarkan bertahap sampai batas minimum piksel terpenuhi.
    """
    total = gray.size
    for factor in (1.0, 1.10, 1.22, 1.4):
        ink = gray < min(250.0, thresh * factor)
        if ink.sum() >= max(16, total * min_fraction):
            ys, xs = np.nonzero(ink)
            return int(ys.min()), int(ys.max()) + 1, int(xs.min()), int(xs.max()) + 1
    ys, xs = np.nonzero(gray < _otsu_threshold(gray))
    if ys.size == 0:
        return None
    return int(ys.min()), int(ys.max()) + 1, int(xs.min()), int(xs.max()) + 1


def _local_high_percentile(
    small: np.ndarray,
    win: int,
    alpha: float = 0.35,
) -> np.ndarray:
    """Estimasi lokal persentil tinggi via kombinasi dilasi & erosi (O(n)).

    Persentil tinggi (`dilate*alpha + erode*(1-alpha)`) memberi perkiraan level
    kertas yang tahan terhadap goresan tinta maupun bayangan, tanpa memerlukan
    perhitungan persentil per blok yang mahal.
    """
    dil = ndi.maximum_filter(small, size=win, mode="nearest")
    ero = ndi.minimum_filter(small, size=win, mode="nearest")
    return alpha * dil + (1.0 - alpha) * ero


def _flatten_illumination(
    gray: np.ndarray,
    work: int = 512,
) -> np.ndarray:
    """Ratukan pencahayaan: bagi citra dengan estimasi level kertas lokal.

    Foto kamera ponsel punya gradien cahaya, vignette, dan bayangan. Membagi
    citra dengan estimasi level kertas lokal membuat kertas menjadi rata tanpa
    menghapus goresan tinta. Estimasi memakai persentil tinggi lokal sehingga
    tetap benar walaupun digit mengisi hampir seluruh bidang pandang.
    """
    h, w = gray.shape
    scale = min(1.0, work / max(h, w))
    small_size = (max(1, int(round(w * scale))), max(1, int(round(h * scale))))
    small = np.asarray(
        Image.fromarray(gray).resize(small_size, Image.Resampling.BILINEAR),
        dtype=np.float32,
    )

    # Jendela ~8% sisi terpanjang, cukup lebar untuk melompati goresan pena.
    win = int(max(21, round(0.08 * max(small.shape)))) | 1  # harus ganjil
    paper = _local_high_percentile(small, win)
    # Haluskan agar tidak muncul artefak blok pada latar.
    paper = ndi.median_filter(paper, size=9, mode="nearest")

    if paper.shape != (h, w):
        paper = np.asarray(
            Image.fromarray(paper.astype(np.float32)).resize(
                (w, h), Image.Resampling.BILINEAR
            ),
            dtype=np.float32,
        )

    flat = gray.astype(np.float32) / np.maximum(paper, 1.0)
    # Rasio latar ~1.0 dipetakan ke PAPER_LEVEL (258); tinta tetap jauh lebih
    # gelap. Nilai di atas 255 dipangkas oleh _normalize_contrast.
    return np.clip(flat * _FLAT_PAPER_LEVEL, 0, 255).astype(np.uint8)


def canonicalize(
    img: Image.Image,
    size: int = CANONICAL_SIZE,
    pad_ratio: float = 0.12,
) -> Image.Image:
    """Ubah foto kamera menjadi kanvas kanonik persegi berisi satu digit.

    Langkah: grayscale -> ratakan pencahayaan -> auto-crop ke bounding box tinta
    (ambang relatif terhadap level kertas) -> normalisasi kontras dua titik
    (kertas menjadi putih, tinta menjadi gelap) -> tempel di kanvas persegi putih
    -> resize ke `size` x `size`.
    """
    gray = _flatten_illumination(np.asarray(img.convert("L"), dtype=np.uint8))
    box = _ink_bbox(gray, _ink_threshold(gray))
    if box is None:
        return Image.new("L", (size, size), color=255)

    y0, y1, x0, x1 = box
    h, w = y1 - y0, x1 - x0
    py = max(1, int(round(h * pad_ratio)))
    px = max(1, int(round(w * pad_ratio)))
    y0 = max(0, y0 - py)
    y1 = min(gray.shape[0], y1 + py)
    x0 = max(0, x0 - px)
    x1 = min(gray.shape[1], x1 + px)
    cropped = Image.fromarray(_normalize_contrast(gray[y0:y1, x0:x1]))

    # Tempel di tengah kanvas persegi putih (mempertahankan rasio aspek digit).
    w, h = cropped.size
    side = max(w, h)
    canvas = Image.new("L", (side, side), color=255)
    canvas.paste(cropped, ((side - w) // 2, (side - h) // 2))

    return canvas.resize((size, size), Image.Resampling.LANCZOS)


def _normalize_contrast(
    patch: np.ndarray,
    paper_cut: float = 236.0,
    ink_floor: float = 30.0,
) -> np.ndarray:
    """Petakan kertas menjadi putih murni dan tinta menjadi gelap.

    Setelah pencahayaan diratakan, level kertas berada pada `_FLAT_PAPER_LEVEL`.
    Semua piksel seterang `paper_cut` atau lebih dipaksa menjadi 255 sehingga
    latar halaman bersih tanpa kotak abu-abu, sementara level tinta (persentil
    rendah patch) dipetakan ke `ink_floor` dengan penguatan terbatas agar
    kedalaman goresan asli tetap terjaga.
    """
    arr = patch.astype(np.float32)
    ink = min(float(np.percentile(arr, 1.0)), paper_cut * 0.85)
    if paper_cut - ink < 3.0:
        return np.full_like(arr, 255, dtype=np.uint8)
    gain = min(3.0, (255.0 - ink_floor) / (paper_cut - ink))
    out = (arr - ink) * gain + ink_floor
    out = np.where(arr >= paper_cut, 255.0, out)
    return np.clip(out, 0, 255).astype(np.uint8)


# ---------------------------------------------------------------------------
# Transformasi individual
# ---------------------------------------------------------------------------


def _rotate(img: Image.Image, deg: float, fill: int = 255) -> Image.Image:
    return img.rotate(deg, resample=Image.Resampling.BICUBIC, fillcolor=fill)


def spec_string(trace: dict[str, float | int]) -> str:
    """Ringkas jejak parameter augmentasi menjadi satu string ringkas."""
    if trace.get("original"):
        return "original"
    return ";".join(f"{k}={v}" for k, v in trace.items())


def _affine_from(
    img: Image.Image,
    tx: float,
    ty: float,
    scale: float,
    shear_deg: float,
    side: int,
    fill: int = 255,
) -> Image.Image:
    """Translasi + skala + shear dalam satu operasi (pusat citra sebagai pivot).

    `tx`/`ty` dalam fraksi sisi citra, `shear_deg` dalam derajat.
    """
    tx_px = tx * side
    ty_px = ty * side
    shear = math.tan(math.radians(shear_deg))

    # Matriks PIL: (a, b, c, d, e, f) memetakan (x, y) -> (a x + b y + c, ...)
    a, b = scale, shear * scale
    d, e = 0.0, scale
    cx = cy = (side - 1) / 2.0
    c = cx - (a * cx + b * cy) + tx_px
    f = cy - (d * cx + e * cy) + ty_px
    return img.transform(
        img.size,
        Image.Transform.AFFINE,
        (a, b, c, d, e, f),
        resample=Image.Resampling.BICUBIC,
        fillcolor=fill,
    )


def _elastic(
    img: Image.Image,
    rng: np.random.Generator,
    policy: AugmentPolicy,
) -> Image.Image:
    """Deformasi elastis halus -> meniru variasi goresan pena."""
    arr = np.asarray(img, dtype=np.float32)
    shape = arr.shape
    dx = ndi.gaussian_filter(
        rng.uniform(-1.0, 1.0, shape), policy.elastic_sigma, mode="reflect"
    )
    dy = ndi.gaussian_filter(
        rng.uniform(-1.0, 1.0, shape), policy.elastic_sigma, mode="reflect"
    )
    dx *= policy.elastic_alpha / (np.abs(dx).max() + 1e-6)
    dy *= policy.elastic_alpha / (np.abs(dy).max() + 1e-6)
    gy, gx = np.meshgrid(
        np.arange(shape[0], dtype=np.float32),
        np.arange(shape[1], dtype=np.float32),
        indexing="ij",
    )
    warped = ndi.map_coordinates(
        arr, [gy + dy, gx + dx], order=1, mode="reflect"
    )
    return Image.fromarray(np.clip(warped, 0, 255).astype(np.uint8))


def _photometric_from(
    img: Image.Image,
    rng: np.random.Generator,
    policy: AugmentPolicy,
    brightness: float,
    contrast: float,
    gradient: bool,
    rec: dict[str, float | int],
) -> Image.Image:
    """Brightness, contrast, dan (opsional) gradien pencahayaan ruangan."""
    arr = np.asarray(img, dtype=np.float32)

    # Brightness: geser level abu-abu.
    arr = arr + brightness * 255.0

    # Contrast: regangkan di sekitar rata-rata citra.
    factor = 1.0 + contrast
    mean = float(arr.mean())
    arr = (arr - mean) * factor + mean

    # Illumination gradient (pencahayaan ruangan tidak rata).
    if gradient:
        side = arr.shape[0]
        gx = np.linspace(-1.0, 1.0, side, dtype=np.float32)
        gy = np.linspace(-1.0, 1.0, side, dtype=np.float32)
        angle_deg = float(rng.uniform(0.0, 360.0))
        strength = float(rng.uniform(0.5, 1.0) * policy.gradient_strength)
        angle = math.radians(angle_deg)
        plane = np.add.outer(gy * math.sin(angle), gx * math.cos(angle))
        arr = arr + plane * strength * 255.0
        rec["grad_angle_deg"] = round(angle_deg, 1)
        rec["grad_strength"] = round(strength, 3)

    return Image.fromarray(np.clip(arr, 0, 255).astype(np.uint8))


def _blur(
    img: Image.Image,
    rng: np.random.Generator,
    policy: AugmentPolicy,
) -> Image.Image:
    if rng.random() >= policy.blur_prob:
        return img
    sigma = rng.uniform(0.35, policy.blur_sigma_max)
    return img.filter(ImageFilter.GaussianBlur(radius=sigma))


def _noise(
    img: Image.Image,
    rng: np.random.Generator,
    policy: AugmentPolicy,
) -> Image.Image:
    if rng.random() >= policy.noise_prob:
        return img
    arr = np.asarray(img, dtype=np.float32)
    sigma = rng.uniform(1.5, policy.noise_sigma_max)
    arr = arr + rng.normal(0.0, sigma, arr.shape)
    return Image.fromarray(np.clip(arr, 0, 255).astype(np.uint8))


def _jpeg_recompress(
    img: Image.Image,
    rng: np.random.Generator,
    policy: AugmentPolicy,
) -> Image.Image:
    if rng.random() >= policy.jpeg_prob:
        return img
    quality = int(rng.integers(policy.jpeg_quality_min, policy.jpeg_quality_max + 1))
    buffer = io.BytesIO()
    img.save(buffer, format="JPEG", quality=quality, optimize=True)
    buffer.seek(0)
    return Image.open(buffer).convert("L")


# ---------------------------------------------------------------------------
# Pipeline lengkap
# ---------------------------------------------------------------------------


def sample_params(rng: np.random.Generator, policy: AugmentPolicy) -> dict[str, float | int]:
    """Undi seluruh parameter augmentasi untuk satu varian (deterministik).

    Satu-satunya tempat angka acak dikonsumsi, sehingga urutan pengundian
    terjamin stabil dan dapat diaudit.
    """
    params: dict[str, float | int] = {
        "rot_deg": round(float(rng.uniform(-policy.max_rotation_deg, policy.max_rotation_deg)), 2),
        "tx": round(float(rng.uniform(-policy.max_translate, policy.max_translate)), 3),
        "ty": round(float(rng.uniform(-policy.max_translate, policy.max_translate)), 3),
        "scale": round(float(rng.uniform(policy.min_scale, policy.max_scale)), 3),
        "shear_deg": round(float(rng.uniform(-policy.max_shear_deg, policy.max_shear_deg)), 2),
    }
    params["elastic_alpha"] = (
        policy.elastic_alpha if rng.random() < policy.elastic_prob else 0
    )
    params["brightness"] = round(
        float(rng.uniform(-policy.brightness_delta, policy.brightness_delta)), 3
    )
    params["contrast"] = round(
        float(rng.uniform(-policy.contrast_delta, policy.contrast_delta)), 3
    )
    gradient = rng.random() < policy.gradient_prob
    params["gradient"] = int(gradient)
    params["grad_angle_deg"] = round(float(rng.uniform(0.0, 360.0)), 1) if gradient else 0
    params["grad_strength"] = (
        round(float(rng.uniform(0.5, 1.0) * policy.gradient_strength), 3) if gradient else 0
    )
    params["blur_sigma"] = (
        round(float(rng.uniform(0.35, policy.blur_sigma_max)), 2)
        if rng.random() < policy.blur_prob
        else 0
    )
    params["noise_sigma"] = (
        round(float(rng.uniform(1.5, policy.noise_sigma_max)), 2)
        if rng.random() < policy.noise_prob
        else 0
    )
    params["jpeg_quality"] = (
        int(rng.integers(policy.jpeg_quality_min, policy.jpeg_quality_max + 1))
        if rng.random() < policy.jpeg_prob
        else 0
    )
    return params


def apply_params(
    img: Image.Image,
    params: dict[str, float | int],
    rng: np.random.Generator,
    policy: AugmentPolicy = DEFAULT_POLICY,
) -> Image.Image:
    """Terapkan parameter hasil `sample_params` ke citra (urutan tetap)."""
    out = _rotate(img, float(params["rot_deg"]))
    out = _affine_from(
        out,
        float(params["tx"]),
        float(params["ty"]),
        float(params["scale"]),
        float(params["shear_deg"]),
        img.size[0],
    )

    if params["elastic_alpha"]:
        out = _elastic(out, rng, policy)

    out = _photometric_from(
        out,
        rng,
        policy,
        float(params["brightness"]),
        float(params["contrast"]),
        bool(params["gradient"]),
        {},
    )

    if params["blur_sigma"]:
        out = out.filter(ImageFilter.GaussianBlur(radius=float(params["blur_sigma"])))

    if params["noise_sigma"]:
        sigma = float(params["noise_sigma"])
        arr = np.asarray(out, dtype=np.float32) + rng.normal(0.0, sigma, out.size[::-1])
        out = Image.fromarray(np.clip(arr, 0, 255).astype(np.uint8))

    if params["jpeg_quality"]:
        buffer = io.BytesIO()
        out.save(buffer, format="JPEG", quality=int(params["jpeg_quality"]), optimize=True)
        buffer.seek(0)
        out = Image.open(buffer).convert("L")

    return out.convert("L")


def augment_image(
    img: Image.Image,
    rng: np.random.Generator,
    policy: AugmentPolicy = DEFAULT_POLICY,
) -> Image.Image:
    """Terapkan satu kombinasi augmentasi acak (label-preserving) pada citra."""
    return apply_params(img, sample_params(rng, policy), rng, policy)


def make_variants(
    canonical: Image.Image,
    n_variants: int,
    seed: int,
    policy: AugmentPolicy = DEFAULT_POLICY,
) -> list[tuple[Image.Image, dict[str, float | int]]]:
    """Buat `n_variants` varian deterministik dari satu citra kanonik.

    Mengembalikan pasangan (citra, jejak parameter) agar setiap berkas keluaran
    dapat dilacak kembali ke parameter augmentasinya.
    """
    rng = np.random.default_rng(seed)
    results: list[tuple[Image.Image, dict[str, float | int]]] = []
    for _ in range(n_variants):
        params = sample_params(rng, policy)
        results.append((apply_params(canonical, params, rng, policy), params))
    return results
