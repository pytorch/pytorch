# Shared paths for SVE512 cross-build scripts (source from bash, do not execute).
sve512_paths() {
  SVE512_DIR="$(cd "$(dirname "${BASH_SOURCE[1]}")" && pwd)"
  PT="$(cd "$SVE512_DIR/../.." && pwd)"
  WORKSPACE="${PYTORCH_SVE512_WORKSPACE:-$(cd "$PT/.." && pwd)}"
  ROOTFS="${ARM64_ROOTFS:-$WORKSPACE/arm64-rootfs}"
}
