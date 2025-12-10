from typing import Any
import numpy as np

AR_U: np.char.chararray[tuple[Any, ...], np.dtype[np.str_]]
AR_S: np.char.chararray[tuple[Any, ...], np.dtype[np.bytes_]]

AR_S.encode()  # type: ignore[misc]
AR_U.decode()  # type: ignore[misc]

AR_U.join(b"_")  # type: ignore[arg-type]
AR_S.join("_")  # type: ignore[arg-type]

AR_U.ljust(5, fillchar=b"a")  # type: ignore[arg-type]
AR_S.ljust(5, fillchar="a")  # type: ignore[arg-type]
AR_U.rjust(5, fillchar=b"a")  # type: ignore[arg-type]
AR_S.rjust(5, fillchar="a")  # type: ignore[arg-type]

AR_U.lstrip(chars=b"a")  # type: ignore[arg-type]
AR_S.lstrip(chars="a")  # type: ignore[arg-type]
AR_U.strip(chars=b"a")  # type: ignore[arg-type]
AR_S.strip(chars="a")  # type: ignore[arg-type]
AR_U.rstrip(chars=b"a")  # type: ignore[arg-type]
AR_S.rstrip(chars="a")  # type: ignore[arg-type]

AR_U.partition(b"a")  # type: ignore[arg-type]
AR_S.partition("a")  # type: ignore[arg-type]
AR_U.rpartition(b"a")  # type: ignore[arg-type]
AR_S.rpartition("a")  # type: ignore[arg-type]

AR_U.replace(b"_", b"-")  # type: ignore[arg-type]
AR_S.replace("_", "-")  # type: ignore[arg-type]

AR_U.split(b"_")  # type: ignore[arg-type]
AR_S.split("_")  # type: ignore[arg-type]
AR_S.split(1)  # type: ignore[arg-type]
AR_U.rsplit(b"_")  # type: ignore[arg-type]
AR_S.rsplit("_")  # type: ignore[arg-type]

AR_U.count(b"a", start=[1, 2, 3])  # type: ignore[arg-type]
AR_S.count("a", end=9)  # type: ignore[arg-type]

AR_U.endswith(b"a", start=[1, 2, 3])  # type: ignore[arg-type]
AR_S.endswith("a", end=9)  # type: ignore[arg-type]
AR_U.startswith(b"a", start=[1, 2, 3])  # type: ignore[arg-type]
AR_S.startswith("a", end=9)  # type: ignore[arg-type]

AR_U.find(b"a", start=[1, 2, 3])  # type: ignore[arg-type]
AR_S.find("a", end=9)  # type: ignore[arg-type]
AR_U.rfind(b"a", start=[1, 2, 3])  # type: ignore[arg-type]
AR_S.rfind("a", end=9)  # type: ignore[arg-type]

AR_U.index(b"a", start=[1, 2, 3])  # type: ignore[arg-type]
AR_S.index("a", end=9)  # type: ignore[arg-type]
AR_U.rindex(b"a", start=[1, 2, 3])  # type: ignore[arg-type]
AR_S.rindex("a", end=9)  # type: ignore[arg-type]

AR_U == AR_S  # type: ignore[operator]
AR_U != AR_S  # type: ignore[operator]
AR_U >= AR_S  # type: ignore[operator]
AR_U <= AR_S  # type: ignore[operator]
AR_U > AR_S  # type: ignore[operator]
AR_U < AR_S  # type: ignore[operator]
