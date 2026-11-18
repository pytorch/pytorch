#!/usr/bin/env bash
# Regenerate the vendored Perfetto C++ Tracing SDK amalgamation.
#
# This is a maintainer step, run on a networked machine; it is not part of the
# build. It refreshes sdk/perfetto.{h,cc} and LICENSE for a chosen upstream
# version, then stamps the version into README.md from the regenerated files.
#
# Usage:
#   ./update.sh <tag>     # download the release zip for <tag>
#
# Example:
#   ./update.sh v56.1
set -euo pipefail

if [[ $# -ne 1 ]]; then
  # Print the leading comment block (usage), stopping at the first non-comment line.
  sed -n '2,/^[^#]/p' "$0" | sed -E '/^[^#]/d; s/^# ?//'
  exit 1
fi

TAG="$1"
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SDK_DIR="$HERE/sdk"
README="$HERE/README.md"

mkdir -p "$SDK_DIR"

# Download the prebuilt amalgamation release zip.
TMP="$(mktemp -d)"
trap 'rm -rf "$TMP"' EXIT
ZIP_URL="https://github.com/google/perfetto/releases/download/$TAG/perfetto-cpp-sdk-src.zip"
echo "Downloading $ZIP_URL"
curl -fsSL "$ZIP_URL" -o "$TMP/sdk.zip"
unzip -q "$TMP/sdk.zip" -d "$TMP/sdk"
H="$(find "$TMP/sdk" -name perfetto.h -print -quit)"
CC="$(find "$TMP/sdk" -name perfetto.cc -print -quit)"
if [[ -z "$H" || -z "$CC" ]]; then
  echo "could not find perfetto.h / perfetto.cc in release zip" >&2
  exit 1
fi
cp "$H" "$SDK_DIR/perfetto.h"
cp "$CC" "$SDK_DIR/perfetto.cc"
echo "Fetching LICENSE for $TAG"
curl -fsSL "https://raw.githubusercontent.com/google/perfetto/$TAG/LICENSE" -o "$HERE/LICENSE"

# Derive the exact version stamp from the regenerated amalgamation so the README
# always matches the vendored files rather than the requested tag.
VERSION_STRING="$(grep -oE 'PERFETTO_VERSION_STRING\(\) "[^"]+"' "$SDK_DIR/perfetto.cc" | head -1 | sed -E 's/.*"([^"]+)"/\1/')"
SCM_REVISION="$(grep -oE 'PERFETTO_VERSION_SCM_REVISION\(\) "[^"]+"' "$SDK_DIR/perfetto.cc" | head -1 | sed -E 's/.*"([^"]+)"/\1/')"
# "v56.1-c794fceab" -> "v56.1"
VERSION="${VERSION_STRING%%-*}"

if [[ -z "$VERSION_STRING" || -z "$SCM_REVISION" ]]; then
  echo "could not extract version stamp from $SDK_DIR/perfetto.cc" >&2
  exit 1
fi

python3 - "$README" "$VERSION" "$VERSION_STRING" "$SCM_REVISION" <<'PY'
import re, sys
readme, version, version_string, scm = sys.argv[1:5]
text = open(readme).read()
block = (
    "## Version\n\n"
    f"- Perfetto **{version}** (`PERFETTO_VERSION_STRING() == \"{version_string}\"`,\n"
    f"  upstream commit `{scm}`).\n"
)
text = re.sub(r"## Version\n\n- Perfetto .*?\)\.\n", block, text, count=1, flags=re.DOTALL)
open(readme, "w").write(text)
PY

echo
echo "Updated:"
echo "  $SDK_DIR/perfetto.h"
echo "  $SDK_DIR/perfetto.cc"
echo "  $HERE/LICENSE"
echo "  $README -> Perfetto $VERSION ($VERSION_STRING, $SCM_REVISION)"
echo
echo "Review the diff, then rebuild with:"
echo "  pip install -e . -v --no-build-isolation"
