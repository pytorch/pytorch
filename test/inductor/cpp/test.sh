#!/bin/bash
set -euo pipefail
IFS=$'\n\t'

cmake . -DATEN_INCLUDE:PATH=$(python -c "import torch; from torch.utils import cpp_extension; print(cpp_extension.include_paths()[0])")
make
./test/bin/test_cpp_prefix
