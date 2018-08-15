#!/bin/bash

set -ex

LOCAL_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
ROOT_DIR=$(cd "$LOCAL_DIR"/../../../ && pwd)

cd "$ROOT_DIR"

echo "Running C++ tests.."

for file in $(find "${ROOT_DIR}/build_caffe2/bin" -executable -type f); do
	if [[ "$file" =~ "test" ]]; then
		$file
	fi
done
