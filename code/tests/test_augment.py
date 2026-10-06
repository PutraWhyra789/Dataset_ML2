"""Unit test untuk pipeline augmentasi (tanpa dependensi dataset asli)."""

from __future__ import annotations

import numpy as np
import pytest
from PIL import Image, ImageDraw

from digit_aug.augment import (
    CANONICAL_SIZE,
    DEFAULT_POLICY,
    canonicalize,
    make_variants,
    spec_string,
)


@pytest.fixture()
def fake_photo() -> Image.Image:
    """Tiruan foto kamera: kertas putih besar dengan digit gelap di tengah."""
    img = Image.new("RGB", (400, 600), color=(245, 245, 240))
    draw = ImageDraw.Draw(img)
    draw.ellipse((150, 200, 250, 400), outline=(20, 20, 20), width=12)
    return img


def test_canonicalize_shape_and_background(fake_photo: Image.Image) -> None:
    out = canonicalize(fake_photo)
    assert out.size == (CANONICAL_SIZE, CANONICAL_SIZE)
    assert out.mode == "L"
    arr = np.asarray(out)
    # Sudut kanvas harus tetap bernuansa kertas (terang, bukan hitam).
    assert arr[0, 0] > 150


def test_canonicalize_handles_blank_image() -> None:
    blank = Image.new("RGB", (64, 64), color=(255, 255, 255))
    out = canonicalize(blank)
    assert out.size == (CANONICAL_SIZE, CANONICAL_SIZE)


def test_augmentation_is_deterministic(fake_photo: Image.Image) -> None:
    canon = canonicalize(fake_photo)
    first, params_a = make_variants(canon, 3, seed=42)[-1]
    second, params_b = make_variants(canon, 3, seed=42)[-1]
    assert params_a == params_b
    assert np.array_equal(np.asarray(first), np.asarray(second))


def test_different_seed_gives_different_image(fake_photo: Image.Image) -> None:
    canon = canonicalize(fake_photo)
    a, _ = make_variants(canon, 1, seed=1)[0]
    b, _ = make_variants(canon, 1, seed=2)[0]
    assert not np.array_equal(np.asarray(a), np.asarray(b))


def test_augmentation_keeps_label_identity(fake_photo: Image.Image) -> None:
    """Rotasi/shear/blur ringan tidak boleh menghapus struktur digit."""
    canon = canonicalize(fake_photo)
    ink_before = float((np.asarray(canon) < 200).mean())
    for img, _ in make_variants(canon, 12, seed=7):
        arr = np.asarray(img)
        assert img.size == (CANONICAL_SIZE, CANONICAL_SIZE)
        assert img.mode == "L"
        ink_after = float((arr < 200).mean())
        # Massa tinta tidak berubah drastis -> bentuk digit tetap ada.
        assert ink_after > ink_before * 0.5, "digit terlihat hilang"
        assert arr.min() < 200, "tidak ada piksel tinta sama sekali"


def test_no_flip_or_large_rotation_policy() -> None:
    """Transformasi perusak label harus tetap dilarang oleh kebijakan."""
    p = DEFAULT_POLICY
    assert p.max_rotation_deg <= 15
    assert p.max_shear_deg <= 8
    assert p.min_scale >= 0.85
    assert p.gradient_strength <= 0.15


def test_spec_string_marks_original() -> None:
    assert spec_string({"original": 1}) == "original"
    assert "rot_deg=" in spec_string({"rot_deg": 1.0})


def test_output_pixels_in_valid_range(fake_photo: Image.Image) -> None:
    canon = canonicalize(fake_photo)
    for img, _ in make_variants(canon, 6, seed=99):
        arr = np.asarray(img)
        assert arr.dtype == np.uint8
        assert arr.min() >= 0 and arr.max() <= 255
