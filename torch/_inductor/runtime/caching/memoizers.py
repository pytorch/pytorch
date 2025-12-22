from pathlib import Path

from .interfaces import PersistentMemoizer


# Memoizer for _should_pad in pad_mm
should_pad_memoizer = PersistentMemoizer(sub_dir=Path("should_pad"))
