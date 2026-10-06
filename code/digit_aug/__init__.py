"""Paket generator augmentasi dataset digit tulisan tangan."""

from .augment import (
    CANONICAL_SIZE,
    CLASSES,
    DEFAULT_POLICY,
    AugmentPolicy,
    apply_params,
    augment_image,
    canonicalize,
    make_variants,
    sample_params,
    spec_string,
)

__all__ = [
    "CANONICAL_SIZE",
    "CLASSES",
    "DEFAULT_POLICY",
    "AugmentPolicy",
    "apply_params",
    "augment_image",
    "canonicalize",
    "make_variants",
    "sample_params",
    "spec_string",
]
__version__ = "1.0.0"
