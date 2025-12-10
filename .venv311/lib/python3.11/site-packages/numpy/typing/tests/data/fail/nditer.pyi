import numpy as np

class Test(np.nditer): ...  # type: ignore[misc]

np.nditer([0, 1], flags=["test"])  # type: ignore[list-item]
np.nditer([0, 1], op_flags=[["test"]])  # type: ignore[list-item]
np.nditer([0, 1], itershape=(1.0,))  # type: ignore[arg-type]
np.nditer([0, 1], buffersize=1.0)  # type: ignore[arg-type]
