import os
import sys

THIRD_PARTY_PATH = "../../../../third_party/"
CUTLASS_PY_PATH = "cutlass/tools/library/scripts/"

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), THIRD_PARTY_PATH, CUTLASS_PY_PATH)))
