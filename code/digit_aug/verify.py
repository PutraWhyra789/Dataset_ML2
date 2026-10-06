"""Verifikasi dataset augmentasi: distribusi kelas, integritas berkas, duplikat.

Contoh pakai:
    uv run python -m digit_aug.verify --dataset "../augmented dataset"
"""

from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter
from pathlib import Path

import numpy as np
from PIL import Image

from .augment import CLASSES


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(prog="digit_aug.verify")
    parser.add_argument(
        "--dataset",
        type=Path,
        default=Path("../augmented dataset"),
        help=(
            "Folder dataset augmentasi "
            "(default: ../augmented dataset, yaitu Dataset/augmented dataset)."
        ),
    )
    parser.add_argument(
        "--expect-per-class",
        type=int,
        default=50,
        help="Jumlah citra yang diharapkan per kelas (0 = tidak dicek).",
    )
    return parser.parse_args(argv)


def file_hash(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    root = args.dataset.resolve()
    if not root.is_dir():
        raise SystemExit(f"[ERROR] Folder dataset tidak ditemukan: {root}")

    problems: list[str] = []
    counts: dict[str, int] = {}
    hashes: dict[str, list[str]] = {}
    sizes: Counter[tuple[int, int]] = Counter()
    modes: Counter[str] = Counter()
    pixel_stats: list[float] = []
    ink_ratios: list[float] = []

    for label in CLASSES:
        files = sorted((root / label).glob("*.jpg"))
        counts[label] = len(files)
        hashes[label] = []
        if args.expect_per_class and len(files) != args.expect_per_class:
            problems.append(
                f"kelas {label}: {len(files)} berkas (harusnya {args.expect_per_class})"
            )
        for path in files:
            try:
                with Image.open(path) as img:
                    img.load()
                    sizes[img.size] += 1
                    modes[img.mode] += 1
                    arr = np.asarray(img, dtype=np.float32)
            except Exception as exc:  # pragma: no cover - jalur galat I/O
                problems.append(f"{path.name}: gagal dibuka ({exc})")
                continue
            # Citra kosong/rusak: rentang piksel sangat sempit.
            if float(arr.max() - arr.min()) < 20:
                problems.append(f"kelas {label}/{path.name}: kontras hampir nol")
            # Kehilangan tinta: hampir seluruh kanvas putih.
            # Digit bergores tipis (1-2% luas citra), sehingga ambang bawah
            # dibuat longgar; kegagalan nyata ditandai oleh |min| yang tinggi.
            ink_ratio = float((arr < 215).mean())
            ink_ratios.append(ink_ratio)
            if ink_ratio < 0.004 or float(arr.min()) > 150:
                problems.append(
                    f"kelas {label}/{path.name}: goresan digit hilang "
                    f"(tinta {ink_ratio:.3%}, min {arr.min():.0f})"
                )
            if ink_ratio > 0.60:
                problems.append(f"kelas {label}/{path.name}: tinta > 60% (citra rusak)")
            pixel_stats.append(float(arr.mean()))
            hashes[label].append(file_hash(path))

    # Duplikat persis di dalam satu kelas (indikasi augmentasi tidak bervariasi).
    for label in CLASSES:
        dupes = [h for h, n in Counter(hashes[label]).items() if n > 1]
        if dupes:
            problems.append(f"kelas {label}: {len(dupes)} berkas duplikat persis")

    total = sum(counts.values())
    print("=== Verifikasi dataset augmentasi ===")
    print(f"Folder      : {root}")
    print(f"Total citra : {total}")
    print("Distribusi  :")
    for label in CLASSES:
        print(f"  kelas {label}: {counts[label]:>4} citra")
    print(f"Mode warna  : {dict(modes)}")
    print(f"Ukuran      : {dict(sizes)}")
    if pixel_stats:
        print(f"Intensitas  : rata-rata {np.mean(pixel_stats):.1f} (0=hitam, 255=putih)")
    if ink_ratios:
        arr_ink = np.array(ink_ratios)
        print(
            f"Massa tinta : rata-rata {arr_ink.mean():.3%} | "
            f"min {arr_ink.min():.3%} | maks {arr_ink.max():.3%}"
        )

    manifest_path = root / "_metadata" / "manifest.json"
    if manifest_path.is_file():
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        print(
            f"Manifest    : v{manifest.get('generator_version')} | seed={manifest.get('seed')} "
            f"| dibuat {manifest.get('generated_at_utc')}"
        )
    else:
        problems.append("_metadata/manifest.json tidak ditemukan")

    print()
    if problems:
        print(f"[GAGAL] {len(problems)} masalah ditemukan:")
        for p in problems:
            print(f"  - {p}")
        return 1

    print("[OK] Distribusi seimbang, semua berkas valid, tidak ada duplikat.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
