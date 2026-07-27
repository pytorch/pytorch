#!/bin/bash

set -ex

ignore_warning() {
  # Invert match to filter out $1.
  set +e
  grep -v "$1" doxygen-log.txt > temp.txt
  set -e
  mv temp.txt doxygen-log.txt
}

command -v doxygen >/dev/null 2>&1 || { echo >&2 "doxygen is not supported. Aborting."; exit 1; }

pushd "$(dirname "$0")/../../.."

cp torch/_utils_internal.py tools/shared

python -m torchgen.gen --source-path aten/src/ATen

python tools/setup_helpers/generate_code.py                 \
  --native-functions-path aten/src/ATen/native/native_functions.yaml \
  --tags-path aten/src/ATen/native/tags.yaml
popd

# Run doxygen and log all output.
doxygen "$(dirname $0)" 2> original-doxygen-log.txt
cp original-doxygen-log.txt doxygen-log.txt

# Uncomment this if you need it for debugging; we're not printing this
# by default because it is confusing.
# echo "Original output"
# cat original-doxygen-log.txt

# Filter out some warnings.
ignore_warning "warning: no uniquely matching class member found for"
ignore_warning "warning: explicit link request to 'Item' could not be resolved"
ignore_warning "warning: Included by graph for 'types.h' not generated, too many nodes"

# Count the number of remaining warnings.
warnings="$(grep 'warning:' doxygen-log.txt | wc -l)"

echo "Treating all remaining warnings as errors"

if [[ "$warnings" -ne "0" ]]; then
  echo "Failing Doxygen test because the following warnings were treated fatally:"
  cat doxygen-log.txt
  echo "Please fix these warnings.  To run this test locally, use docs/cpp/source/check-doxygen.sh"
  rm -f doxygen-log.txt original-doxygen-log.txt
  exit 1
fi

rm -f doxygen-log.txt original-doxygen-log.txt
