from pathlib import Path

from .interfaces import PersistentMemoizer


# Memoizer for _should_pad in pad_mm
should_pad_memoizer = PersistentMemoizer(sub_dir=Path("should_pad"))

# Memoizer for tuned_mm in kernel/mm
tuned_mm_memoizer = PersistentMemoizer(sub_dir=Path("tuned_mm"))

# Memoizer for tuned_addmm in kernel/mm
tuned_addmm_memoizer = PersistentMemoizer(sub_dir=Path("tuned_addmm"))

# Memoizer for tuned_bmm in kernel/bmm
tuned_bmm_memoizer = PersistentMemoizer(sub_dir=Path("tuned_bmm"))

# Memoizer for tuned_baddbmm in kernel/bmm
tuned_baddbmm_memoizer = PersistentMemoizer(sub_dir=Path("tuned_baddbmm"))

# Memoizer for tuned_mm_plus_mm in kernel/mm_plus_mm
tuned_mm_plus_mm_memoizer = PersistentMemoizer(sub_dir=Path("tuned_mm_plus_mm"))

# Memoizer for tuned_int_mm in kernel/mm
tuned_int_mm_memoizer = PersistentMemoizer(sub_dir=Path("tuned_int_mm"))

# Memoizer for tuned_scaled_mm in kernel/mm
tuned_scaled_mm_memoizer = PersistentMemoizer(sub_dir=Path("tuned_scaled_mm"))
