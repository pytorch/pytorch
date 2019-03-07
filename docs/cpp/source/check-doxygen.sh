#!/bin/bash

set -ex

ignore_warning() {
  # Invert match to filter out $1.
  set +e
  grep -v "$1" doxygen-log.txt > temp.txt
  set -e
  mv temp.txt doxygen-log.txt
}

pushd "$(dirname "$0")/../../.."

cp aten/src/ATen/common_with_cwrap.py tools/shared/cwrap_common.py
cp torch/_utils_internal.py tools/shared

python aten/src/ATen/gen.py \
  -s aten/src/ATen \
  -d build/aten/src/ATen \
  aten/src/ATen/Declarations.cwrap \
  aten/src/THNN/generic/THNN.h \
  aten/src/THCUNN/generic/THCUNN.h \
  aten/src/ATen/nn.yaml \
  aten/src/ATen/native/native_functions.yaml

python tools/setup_helpers/generate_code.py                 \
  --declarations-path build/aten/src/ATen/Declarations.yaml \
  --nn-path aten/src

popd

# Run doxygen and log all output.
doxygen 2> original-doxygen-log.txt
cp original-doxygen-log.txt doxygen-log.txt

echo "Original output"
cat original-doxygen-log.txt

# Filter out some warnings.
ignore_warning "warning: no uniquely matching class member found for"

# Count the number of remaining warnings.
warnings="$(grep 'warning:' doxygen-log.txt | wc -l)"

if [[ "$warnings" -ne "0" ]]; then
  echo "Filtered output"
  cat doxygen-log.txt
  rm -f doxygen-log.txt original-doxygen-log.txt
  exit 1
fi

rm -f doxygen-log.txt original-doxygen-log.txt
