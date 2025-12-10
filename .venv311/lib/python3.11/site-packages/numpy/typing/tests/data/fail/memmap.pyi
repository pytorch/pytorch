import numpy as np

with open("file.txt", "r") as f:
    np.memmap(f)  # type: ignore[call-overload]
np.memmap("test.txt", shape=[10, 5])  # type: ignore[call-overload]
