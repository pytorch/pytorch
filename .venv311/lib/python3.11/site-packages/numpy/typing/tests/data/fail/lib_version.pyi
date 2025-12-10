from numpy.lib import NumpyVersion

version: NumpyVersion

NumpyVersion(b"1.8.0")  # type: ignore[arg-type]
version >= b"1.8.0"  # type: ignore[operator]
