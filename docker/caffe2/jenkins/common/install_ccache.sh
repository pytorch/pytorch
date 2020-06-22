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

# If rocm build, if hcc file exists then use hcc else clang(hip-clang) for sccache
if [[ "${BUILD_ENVIRONMENT}" == *-rocm* ]]; then
  if [[ -e "/opt/rocm/hcc/bin/hcc" ]]; then
    HIPCOM_DEST_PATH="$(readlink -f /opt/rocm/hcc/bin/hcc )"
  else
    HIPCOM_DEST_PATH="$(readlink -f /opt/rocm/llvm/bin/clang )"
  fi
  HIPCOM_REAL_BINARY="$(dirname $HIPCOM_DEST_PATH)/hipcompiler_original"
  mv "$HIPCOM_DEST_PATH" "$HIPCOM_REAL_BINARY"

  # Create sccache wrapper.
  (
    echo "#!/bin/sh"
    echo "exec $SCCACHE $HIPCOM_REAL_BINARY \"\$@\""
  ) > "$HIPCOM_DEST_PATH"
  chmod +x "$HIPCOM_DEST_PATH"
fi
