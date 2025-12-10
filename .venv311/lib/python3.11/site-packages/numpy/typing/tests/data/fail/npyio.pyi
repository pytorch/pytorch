import pathlib
from typing import IO

import numpy.typing as npt
import numpy as np

str_path: str
bytes_path: bytes
pathlib_path: pathlib.Path
str_file: IO[str]
AR_i8: npt.NDArray[np.int64]

np.load(str_file)  # type: ignore[arg-type]

np.save(bytes_path, AR_i8)  # type: ignore[call-overload]
np.save(str_path, AR_i8, fix_imports=True)  # type: ignore[deprecated]  # pyright: ignore[reportDeprecated]

np.savez(bytes_path, AR_i8)  # type: ignore[arg-type]

np.savez_compressed(bytes_path, AR_i8)  # type: ignore[arg-type]

np.loadtxt(bytes_path)  # type: ignore[arg-type]

np.fromregex(bytes_path, ".", np.int64)  # type: ignore[call-overload]
