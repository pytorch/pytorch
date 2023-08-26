import os
import sys

THIRD_PARTY_PATH = "../../../../third_party/"
CUTLASS_PY_PATH = "cutlass/tools/library/scripts/"

# Add CUTLASS Python scripts to Python search path.
# TODO(ipiszy): Update cutlass repo to export all submodules under "cutlass_scripts" to avoid
# potential naming conflicts.
sys.path.insert(
    0,
    os.path.abspath(
        os.path.join(os.path.dirname(__file__), THIRD_PARTY_PATH, CUTLASS_PY_PATH)
    ),
)
