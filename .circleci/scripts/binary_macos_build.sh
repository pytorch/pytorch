#!/bin/bash
set -eux -o pipefail

source "/Users/distiller/project/env"
mkdir -p "$PYTORCH_FINAL_PACKAGE_DIR"

# For some reason `unbuffer` breaks if we change the PATH here, so we
# write a script with the PATH change in it and unbuffer the whole
# thing
build_script="$workdir/build_script.sh"
touch "$build_script"
chmod +x "$build_script"

# Build
cat >"$build_script" <<EOL
export PATH="$workdir/miniconda/bin:$PATH"
if [[ "$PACKAGE_TYPE" == conda ]]; then
  "$workdir/builder/conda/build_pytorch.sh"
else
  export TORCH_PACKAGE_NAME="$(echo $TORCH_PACKAGE_NAME | tr '-' '_')"
  "$workdir/builder/wheel/build_wheel.sh"
fi
EOL
unbuffer "$build_script" | ts
