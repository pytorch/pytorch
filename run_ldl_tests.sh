#!/usr/bin/env bash
# Runs every LDL test: the full test_ldl_custom.py (which also builds and uses
# ldl_panel_standalone.cu), and with SANITIZE=1 the compute-sanitizer passes.
# Run from an environment where torch (built from this checkout) is importable,
# e.g. `micromamba activate kernel`. Extra args are passed on to the test runner,
# e.g. `./run_ldl_tests.sh -k cooperative`.
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")"

echo "== compiling ldl_panel_standalone.cu"
python - <<'PY'
from torch.utils.cpp_extension import load
load("ldl_panel_standalone", ["ldl_panel_standalone.cu"], verbose=True)
PY

echo "== running test_ldl_custom.py"
python test_ldl_custom.py "$@"

if [[ "${SANITIZE:-0}" == "1" ]]; then
  ./run_ldl_panel_standalone.sh
fi
