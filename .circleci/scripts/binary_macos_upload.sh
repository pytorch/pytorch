#!/bin/bash
# Do NOT set -x
set -eu -o pipefail
set +x
export AWS_ACCESS_KEY_ID="${PYTORCH_BINARY_AWS_ACCESS_KEY_ID}"
export AWS_SECRET_ACCESS_KEY="${PYTORCH_BINARY_AWS_SECRET_ACCESS_KEY}"

#!#!#!#!#!#!#!#!#!#!#!#!#!#!#!#!#!#!#!#!#!#!#!#!
# DO NOT TURN -x ON BEFORE THIS LINE
#!#!#!#!#!#!#!#!#!#!#!#!#!#!#!#!#!#!#!#!#!#!#!#!
set -eux -o pipefail

source "/Users/distiller/project/env"
export "PATH=$workdir/miniconda/bin:$PATH"

# This gets set in binary_populate_env.sh, but lets have a sane default just in case
PIP_UPLOAD_FOLDER=${PIP_UPLOAD_FOLDER:-nightly}

pushd "$workdir/final_pkgs"
if [[ "$PACKAGE_TYPE" == conda ]]; then
  retry conda install -yq anaconda-client
  retry anaconda -t "${CONDA_PYTORCHBOT_TOKEN}" upload "$(ls)" -u "pytorch-${PIP_UPLOAD_FOLDER}" --label main --no-progress --force
elif [[ "$PACKAGE_TYPE" == libtorch ]]; then
  retry pip install -q awscli
  s3_dir="s3://pytorch/libtorch/${PIP_UPLOAD_FOLDER}${DESIRED_CUDA}/"
  for pkg in $(ls); do
    retry aws s3 cp "$pkg" "$s3_dir" --acl public-read
  done
else
  retry pip install -q awscli
  s3_dir="s3://pytorch/whl/${PIP_UPLOAD_FOLDER}${DESIRED_CUDA}/"
  retry aws s3 cp "$(ls)" "$s3_dir" --acl public-read
fi
