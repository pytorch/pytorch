#!/bin/bash

set -ex

clang-format --version

# Run clang-format.
# Exit with non-zero status if output is non-empty.
[ -z "$(python tools/clang_format.py)" ]
