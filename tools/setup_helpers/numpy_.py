"NumPy helper."

try:
    import numpy as np
except ImportError:
    USE_NUMPY = False
    NUMPY_INCLUDE_DIR = None
else:
    USE_NUMPY = True
    NUMPY_INCLUDE_DIR = np.get_include()
