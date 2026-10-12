#!/usr/bin/env bash
# Compiles ldl_panel_standalone.cu and runs the tests that use it.
# Run from an environment where torch (built from this checkout) is importable,
# e.g. `micromamba activate kernel`. Extra args are passed on to the test runner.
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")"

echo "== compiling ldl_panel_standalone.cu"
python - <<'PY'
from torch.utils.cpp_extension import load
load("ldl_panel_standalone", ["ldl_panel_standalone.cu"], verbose=True)
PY

echo "== running tests"
python test_ldl_custom.py -k standalone -k grid_size "$@"

if [[ "${SANITIZE:-0}" == "1" ]]; then
  # racecheck covers shared memory only, synccheck the barriers (including the
  # grid-wide ones). Slow, so only a few representative tests.
  for tool in racecheck synccheck; do
    echo "== compute-sanitizer --tool $tool"
    compute-sanitizer --tool "$tool" --error-exitcode 1 \
      python test_ldl_custom.py -k block_boundary -k far_blocks -k nonfinite_column
  done
fi
