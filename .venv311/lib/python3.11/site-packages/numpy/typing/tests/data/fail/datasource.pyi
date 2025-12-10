from pathlib import Path
import numpy as np

path: Path
d1: np.lib.npyio.DataSource

d1.abspath(path)  # type: ignore[arg-type]
d1.abspath(b"...")  # type: ignore[arg-type]

d1.exists(path)  # type: ignore[arg-type]
d1.exists(b"...")  # type: ignore[arg-type]

d1.open(path, "r")  # type: ignore[arg-type]
d1.open(b"...", encoding="utf8")  # type: ignore[arg-type]
d1.open(None, newline="/n")  # type: ignore[arg-type]
