"""Tumor class labels.

Order matches the model's output logits and the standard 4-class
Brain Tumor MRI dataset on Kaggle.
"""

from __future__ import annotations

CLASS_NAMES: tuple[str, ...] = (
    "glioma",
    "meningioma",
    "notumor",
    "pituitary",
)

CLASS_DESCRIPTIONS: dict[str, str] = {
    "glioma": "Glioma tumor — originates in glial cells of the brain or spine.",
    "meningioma": "Meningioma tumor — arises from the meninges surrounding the brain.",
    "notumor": "No tumor detected in the MRI scan.",
    "pituitary": "Pituitary tumor — develops in the pituitary gland.",
}


def num_classes() -> int:
    return len(CLASS_NAMES)
