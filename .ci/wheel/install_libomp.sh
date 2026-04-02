#!/bin/bash

# Function to retry functions that sometimes timeout or have flaky failures
retry () {
    $*  || (sleep 1 && $*) || (sleep 2 && $*) || (sleep 4 && $*) || (sleep 8 && $*)
}

# Install libomp 19.1.7 (LLVM 19) from Homebrew's arm64_sonoma bottle
# Verified metadata:
#   - minos: 14.0
#   - sdk: 14.5
#   - built on: macOS 14.7
#   - homebrew-core commit: 3fca565
#   - bottle tag: 19.1.7.arm64_sonoma
#   - bottle blob sha256: df8bbd27e18c5206c5ca1b2d8858308f0a41c1c9f7a73f5d9593e606d37817e3
LIBOMP_BOTTLE="/tmp/libomp-19.1.7.arm64_sonoma.bottle.tar.gz"
EXPECTED_SHA="df8bbd27e18c5206c5ca1b2d8858308f0a41c1c9f7a73f5d9593e606d37817e3"
TOKEN=$(curl -s "https://ghcr.io/token?scope=repository:homebrew/core/libomp:pull" | grep -o '"token":"[^"]*"' | cut -d'"' -f4)
retry curl -sL "https://ghcr.io/v2/homebrew/core/libomp/blobs/sha256:${EXPECTED_SHA}" \
  -H "Authorization: Bearer ${TOKEN}" \
  -o "${LIBOMP_BOTTLE}"
echo "${EXPECTED_SHA}  ${LIBOMP_BOTTLE}" | shasum -a 256 -c -
brew install "${LIBOMP_BOTTLE}"
