#!/bin/bash

set -ex

# Install sccache from pre-compiled binary.
curl https://s3.amazonaws.com/ossci-linux/sccache -o /usr/local/bin/sccache
chmod a+x /usr/local/bin/sccache

# Setup SCCACHE
###############################################################################
SCCACHE="$(which sccache)"
if [ -z "${SCCACHE}" ]; then
  echo "Unable to find sccache..."
  exit 1
fi

# If rocm build, add hcc to sccache.
if [[ "${BUILD_ENVIRONMENT}" == *-rocm* ]]; then
  # HCC's symlink path: /opt/rocm/hcc/bin/hcc -> /opt/rocm/hcc/bin/clang -> /opt/rocm/hcc/bin/clang-7.0
  HCC_DEST_PATH="$(readlink -f $(which hcc))"
  HCC_REAL_BINARY="$(dirname $HCC_DEST_PATH)/clang-7.0_original"
  mv "$HCC_DEST_PATH" "$HCC_REAL_BINARY"

  # Create sccache wrapper.
  (
    echo "#!/bin/sh"
    echo "exec $SCCACHE $HCC_REAL_BINARY \"\$@\""
  ) > "$HCC_DEST_PATH"
  chmod +x "$HCC_DEST_PATH"
fi
