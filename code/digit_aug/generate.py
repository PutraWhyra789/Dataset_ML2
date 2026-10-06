"""Generator dataset augmentasi digit tulisan tangan 0-9.

Membaca dataset asli (folder per kelas 0-9), melakukan pra-pemrosesan kanonik
(auto-crop digit, kanvas persegi 128x128) lalu membangkitkan augmentasi
label-preserving sampai jumlah citra per kelas terpenuhi.

Contoh pakai:
    uv run python -m digit_aug.generate --source ../Dataset/Labeled \
        --output "../augmented dataset" --per-class 50 --seed 42

Semua proses deterministik: seed yang sama + dataset sumber yang sama akan
menghasilkan berkas yang identik (byte-for-byte untuk JPEG quality tetap).
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import platform
import sys
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
from PIL import Image

from . import __version__
from .augment import (
    CANONICAL_SIZE,
    CLASSES,
    DEFAULT_POLICY,
    canonicalize,
    make_variants,
    spec_string,
)

SOURCE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        prog="digit_aug.generate",
        description="Bangkitkan dataset augmentasi digit tulisan tangan (ImageFolder).",
    )
    parser.add_argument(
        "--source",
        type=Path,
        default=Path("../Labeled"),
        help=(
            "Folder dataset asli berisi subfolder kelas 0-9 "
            "(default: ../Labeled, yaitu Dataset/Labeled bila dijalankan dari Dataset/code)."
        ),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("../augmented dataset"),
        help=(
            "Folder keluaran dataset augmentasi "
            "(default: ../augmented dataset, yaitu Dataset/augmented dataset)."
        ),
    )
    parser.add_argument(
        "--per-class",
        type=int,
        default=50,
        help="Target jumlah citra akhir per kelas (default: 50).",
    )
    parser.add_argument("--seed", type=int, default=42, help="Random seed (default: 42).")
    parser.add_argument(
        "--jpeg-quality",
        type=int,
        default=95,
        help="Kualitas JPEG berkas keluaran (default: 95).",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=0,
        help="Jumlah thread (0 = otomatis).",
    )
    parser.add_argument(
        "--size",
        type=int,
        default=CANONICAL_SIZE,
        help=f"Sisi citra kanonik dalam piksel (default: {CANONICAL_SIZE}).",
    )
    parser.add_argument(
        "--limit-sources",
        type=int,
        default=0,
        help="Batasi jumlah citra sumber per kelas (0 = semua). Untuk uji cepat.",
    )
    return parser.parse_args(argv)


def collect_sources(source: Path, limit: int = 0) -> dict[str, list[Path]]:
    """Kumpulkan berkas citra sumber per kelas, urut dan tervalidasi."""
    if not source.is_dir():
        raise SystemExit(f"[ERROR] Folder sumber tidak ditemukan: {source.resolve()}")

    found: dict[str, list[Path]] = {}
    for label in CLASSES:
        class_dir = source / label
        if not class_dir.is_dir():
            raise SystemExit(f"[ERROR] Subfolder kelas tidak ditemukan: {class_dir.resolve()}")
        files = sorted(
            (p for p in class_dir.iterdir() if p.suffix.lower() in SOURCE_EXTS),
            key=lambda p: (len(p.stem), p.stem),
        )
        if not files:
            raise SystemExit(f"[ERROR] Tidak ada citra di {class_dir.resolve()}")
        found[label] = files[:limit] if limit > 0 else files

    counts = {k: len(v) for k, v in found.items()}
    if len(set(counts.values())) != 1:
        print(f"[WARN] Jumlah citra sumber per kelas tidak seimbang: {counts}")
    return found


def _display_path(path: Path, base: Path) -> str:
    """Tampilkan jalur relatif terhadap root dataset agar manifest portabel.

    Manifest tidak boleh menyimpan jalur absolut mesin pembuat, dan juga tidak
    boleh bergantung pada direktori kerja saat perintah dijalankan (host vs
    container). Dengan basis folder induk keluaran, nilainya konsisten:
    `Labeled` dan `augmented dataset`.
    """
    return os.path.relpath(path, base)


def source_fingerprint(files_by_class: dict[str, list[Path]]) -> str:
    """Hash SHA-256 dari daftar berkas + ukuran + mtime (jejak dataset sumber)."""
    digest = hashlib.sha256()
    for label in sorted(files_by_class):
        for path in files_by_class[label]:
            stat = path.stat()
            digest.update(f"{label}/{path.name}|{stat.st_size}|{int(stat.st_mtime)}".encode())
    return digest.hexdigest()


def build_plan(
    files_by_class: dict[str, list[Path]], per_class: int, seed: int
) -> list[tuple[str, int, Path, int, bool]]:
    """Susun rencana keluaran: (label, index, sumber, varian_ke, is_original).

    Citra #1 setiap kelas = citra kanonik asli (tanpa augmentasi). Sisanya
    didistribusikan merata (round-robin) ke seluruh citra sumber kelas tersebut
    agar setiap tulisan asli mendapat porsi varian yang setara.
    """
    plan: list[tuple[str, int, Path, int, bool]] = []
    for label in CLASSES:
        files = files_by_class[label]
        if per_class < len(files):
            raise SystemExit(
                f"[ERROR] --per-class={per_class} lebih kecil dari jumlah citra sumber "
                f"kelas {label} ({len(files)}). Naikkan nilainya."
            )
        plan.append((label, 1, files[0], 0, True))
        cursor = 0
        for idx in range(2, per_class + 1):
            src = files[cursor % len(files)]
            variant_order = idx // len(files) + 1
            plan.append((label, idx, src, variant_order, False))
            cursor += 1
    return plan


def output_name(label: str, index: int, suffix: str = ".jpg") -> str:
    return f"{label}_{index:03d}{suffix}"


def reset_output(output: Path) -> None:
    """Bersihkan berkas hasil lama agar regenerasi bersifat idempoten.

    Hanya berkas di dalam folder kelas 0-9 dan folder `_metadata` yang dihapus,
    sehingga jalur lain di root keluaran tidak tersentuh.
    """
    removed = 0
    for label in CLASSES:
        class_dir = output / label
        class_dir.mkdir(parents=True, exist_ok=True)
        for stale in class_dir.glob("*.jpg"):
            stale.unlink()
            removed += 1
    metadata_dir = output / "_metadata"
    if metadata_dir.is_dir():
        for stale in metadata_dir.iterdir():
            if stale.is_file():
                stale.unlink()
                removed += 1
    if removed:
        print(f"[INFO] Membersihkan {removed} berkas hasil lama di {output}")


def generate_one(
    item: tuple[str, int, Path, int, bool],
    cache: dict[Path, Image.Image],
    cache_lock,
    seed: int,
    size: int,
) -> dict[str, object]:
    label, index, src, variant_order, is_original = item

    with cache_lock:
        canonical = cache.get(src)
    if canonical is None:
        with Image.open(src) as raw:
            canonical = canonicalize(raw, size=size)
        with cache_lock:
            cache[src] = canonical

    if is_original:
        out_img = canonical
        params: dict[str, float | int] = {"original": 1}
        rng_seed = 0
    else:
        # Seed deterministik per (kelas, indeks keluaran) -> tidak bergantung
        # pada urutan eksekusi maupun jumlah thread.
        rng_seed = seed * 1_000_003 + int(label) * 10_007 + index
        (out_img, params), = make_variants(canonical, 1, rng_seed, DEFAULT_POLICY)

    return {
        "label": label,
        "index": index,
        "file_name": output_name(label, index),
        "source_file": src.name,
        "type": spec_string(params),
        "variant_order": variant_order,
        "rng_seed": rng_seed,
        "image": out_img,
    }


def make_folds(records: list[dict[str, object]], n_folds: int = 5, seed: int = 42) -> dict:
    """Partisi stratified k-fold deterministik atas dataset augmentasi."""
    rng = np.random.default_rng(seed)
    per_class_folds: dict[str, list[str]] = {}
    for label in CLASSES:
        names = [str(r["file_name"]) for r in records if r["label"] == label]
        names = sorted(names)
        rng.shuffle(names)
        per_class_folds[label] = [names[i::n_folds] for i in range(n_folds)]

    folds = []
    for k in range(n_folds):
        val = sorted(n for label in CLASSES for n in per_class_folds[label][k])
        train = sorted(
            str(r["file_name"])
            for r in records
            if r["file_name"] not in set(val)
        )
        folds.append({"fold": k + 1, "train": train, "val": val})
    return {
        "scheme": f"{n_folds}-fold stratified cross-validation",
        "seed": seed,
        "n_folds": n_folds,
        "note": "Indeks berkas relatif terhadap root dataset augmentasi ({label}/{file_name}).",
        "folds": folds,
    }


def write_csv(path: Path, records: list[dict[str, object]]) -> None:
    fields = [
        "file_name",
        "relative_path",
        "label",
        "class_name",
        "type",
        "source_file",
        "variant_order",
        "rng_seed",
        "width",
        "height",
        "mode",
    ]
    with path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=fields)
        writer.writeheader()
        for r in sorted(records, key=lambda r: (int(r["label"]), int(r["index"]))):
            label = str(r["label"])
            writer.writerow(
                {
                    "file_name": r["file_name"],
                    "relative_path": f"{label}/{r['file_name']}",
                    "label": label,
                    "class_name": f"Digit {label}",
                    "type": r["type"],
                    "source_file": r["source_file"],
                    "variant_order": r["variant_order"],
                    "rng_seed": r["rng_seed"],
                    "width": r["width"],
                    "height": r["height"],
                    "mode": r["mode"],
                }
            )


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    source = args.source.resolve()
    output = args.output.resolve()

    files_by_class = collect_sources(source, limit=args.limit_sources)
    n_sources = sum(len(v) for v in files_by_class.values())
    print(f"[INFO] Sumber   : {source}")
    print(f"[INFO] Citra asli: {n_sources} ({len(files_by_class)} kelas)")
    print(f"[INFO] Target    : {args.per_class} citra/kelas -> {args.per_class * len(CLASSES)} citra")
    print(f"[INFO] Keluaran  : {output}")

    plan = build_plan(files_by_class, args.per_class, args.seed)
    workers = args.workers if args.workers > 0 else 4

    output.mkdir(parents=True, exist_ok=True)
    reset_output(output)

    records: list[dict[str, object]] = []
    cache: dict[Path, Image.Image] = {}
    import threading

    cache_lock = threading.Lock()

    def work(item):
        rec = generate_one(item, cache, cache_lock, args.seed, args.size)
        label = str(rec["label"])
        dest = output / label / str(rec["file_name"])
        img: Image.Image = rec["image"]  # type: ignore[assignment]
        img.save(dest, format="JPEG", quality=args.jpeg_quality, optimize=True)
        rec["width"], rec["height"] = img.size
        rec["mode"] = img.mode
        return rec

    with ThreadPoolExecutor(max_workers=workers) as pool:
        for i, rec in enumerate(pool.map(work, plan), start=1):
            records.append(rec)
            if i % 50 == 0 or i == len(plan):
                print(f"       ... {i}/{len(plan)} citra", flush=True)

    metadata_dir = output / "_metadata"
    metadata_dir.mkdir(exist_ok=True)
    write_csv(metadata_dir / "metadata.csv", records)

    folds = make_folds(records, n_folds=5, seed=args.seed)
    (metadata_dir / "folds.json").write_text(
        json.dumps(folds, indent=2), encoding="utf-8"
    )

    per_class_counts = {
        label: sum(1 for r in records if r["label"] == label) for label in CLASSES
    }
    manifest = {
        "generator": "digit-augmentation",
        "generator_version": __version__,
        "generated_at_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "seed": args.seed,
        "per_class_target": args.per_class,
        "total_images": len(records),
        "per_class_counts": per_class_counts,
        "canonical_size": args.size,
        "output_mode": "L (grayscale)",
        "jpeg_quality": args.jpeg_quality,
        "source_dir": _display_path(source, output.parent),
        "output_dir": _display_path(output, output.parent),
        "source_image_count": n_sources,
        "source_fingerprint_sha256": source_fingerprint(files_by_class),
        "policy": DEFAULT_POLICY.__dict__,
        "forbidden_transforms": [
            "horizontal_flip",
            "vertical_flip",
            "rotation_larger_than_15_deg",
            "aggressive_crop_or_warp",
            "color_inversion",
        ],
        "environment": {
            "python": sys.version.split()[0],
            "platform": platform.platform(),
        },
    }
    (metadata_dir / "manifest.json").write_text(
        json.dumps(manifest, indent=2), encoding="utf-8"
    )

    # Tiruan split ImageFolder standar agar siap dipakai pelatihan.
    print("\n[SELESAI] Distribusi akhir:")
    for label in CLASSES:
        print(f"  kelas {label}: {per_class_counts[label]} citra")
    print(f"  TOTAL  : {len(records)} citra")
    print(f"[INFO] Metadata: {metadata_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
