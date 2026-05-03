"""
MMLUDataset — wraps the MMLU HuggingFace dataset for use in the LabAI framework.

This bridges the existing mmlu_benchmark.py loading logic into the BaseDataset ABC,
so the same eval runner can evaluate agents on MMLU questions.

Registered as: "mmlu"
"""

from __future__ import annotations

import math
import uuid
from collections import defaultdict

from labai.core.base import BaseDataset
from labai.core.registry import Registry
from labai.core.types import EvalItem

_CHOICES = ["A", "B", "C", "D"]


@Registry.dataset("mmlu")
class MMLUDataset(BaseDataset):
    """
    Loads questions from the MMLU benchmark (HuggingFace: cais/mmlu).

    Args:
        split: HuggingFace split to use (default: "test").

    Usage:
        ds = MMLUDataset()
        items = ds.load(subjects=["high_school_mathematics"], n_samples=50)
    """

    name     = "mmlu"
    domain   = "general"
    language = "en"

    def __init__(self, split: str = "test") -> None:
        self.split = split

    def load(
        self,
        subjects:  list[str] | None = None,
        n_samples: int | None       = None,
    ) -> list[EvalItem]:
        """
        Load MMLU questions as EvalItems.

        Args:
            subjects:  Optional subject keyword filters (e.g. ['math', 'physics']).
            n_samples: Max questions. Sampled evenly across subjects.

        Returns:
            List of EvalItem where:
              - input    = formatted multiple-choice question
              - expected = correct letter (A/B/C/D)
              - metadata = {subject, difficulty="N/A", category=subject}
        """
        from datasets import load_dataset  # lazy import

        raw = load_dataset("cais/mmlu", "all", split=self.split)

        # Build unique list of MMLU subjects
        all_subjects = sorted(set(raw["subject"]))

        # Filter subjects
        if subjects:
            keywords      = [s.lower() for s in subjects]
            all_subjects  = [s for s in all_subjects if any(kw in s for kw in keywords)]
            rows          = [r for r in raw if r["subject"] in set(all_subjects)]
        else:
            rows = list(raw)

        # Even sampling across subjects
        if n_samples is not None and n_samples < len(rows):
            rows = _sample_evenly_mmlu(rows, n_samples)

        items: list[EvalItem] = []
        for row in rows:
            choices_text = "\n".join(
                f"  {letter}) {choice}"
                for letter, choice in zip(_CHOICES, row["choices"])
            )
            input_text = (
                f"{row['question']}\n\n"
                f"Choices:\n{choices_text}\n\n"
                "Think step by step, then end with: ANSWER: <letter>"
            )
            correct_letter = _CHOICES[row["answer"]]
            items.append(
                EvalItem(
                    id       = str(uuid.uuid4())[:8],
                    input    = input_text,
                    expected = correct_letter,
                    metadata = {
                        "subject":    row["subject"],
                        "category":   row["subject"],
                        "difficulty": "N/A",
                    },
                )
            )

        return items


# ── Helpers ────────────────────────────────────────────────────────────────────

def _sample_evenly_mmlu(rows: list, n: int) -> list:
    """Sample n rows evenly across subjects."""
    groups: dict[str, list] = defaultdict(list)
    for row in rows:
        groups[row["subject"]].append(row)

    per_subject = math.ceil(n / max(len(groups), 1))
    sampled: list = []
    for grp in groups.values():
        sampled.extend(grp[:per_subject])

    return sampled[:n]
