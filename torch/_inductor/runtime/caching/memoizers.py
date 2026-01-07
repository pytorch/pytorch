from pathlib import Path

from .interfaces import PersistentMemoizer


# Memoizer for _should_pad in pad_mm
should_pad_memoizer = PersistentMemoizer(sub_dir=Path("should_pad"))

# Memoizer for tuned_mm in kernel/mm
tuned_mm_memoizer = PersistentMemoizer(sub_dir=Path("tuned_mm"))

# Memoizer for tuned_addmm in kernel/mm
tuned_addmm_memoizer = PersistentMemoizer(sub_dir=Path("tuned_addmm"))
