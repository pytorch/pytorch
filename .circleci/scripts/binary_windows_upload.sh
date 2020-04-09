#!/bin/bash
set -eu -o pipefail
set +x
declare -x "AWS_ACCESS_KEY_ID=${PYTORCH_BINARY_AWS_ACCESS_KEY_ID}"
declare -x "AWS_SECRET_ACCESS_KEY=${PYTORCH_BINARY_AWS_SECRET_ACCESS_KEY}"

#!#!#!#!#!#!#!#!#!#!#!#!#!#!#!#!#!#!#!#!#!#!#!#!
# DO NOT TURN -x ON BEFORE THIS LINE
#!#!#!#!#!#!#!#!#!#!#!#!#!#!#!#!#!#!#!#!#!#!#!#!
set -eux -o pipefail

configs=($BUILD_ENVIRONMENT)
export PACKAGE_TYPE="${configs[0]}"
export DESIRED_PYTHON="${configs[1]}"
export DESIRED_CUDA="${configs[2]}"
export LIBTORCH_CONFIG="${configs[3]}"

retry () {
    $*  || (sleep 1 && $*) || (sleep 2 && $*) || (sleep 4 && $*) || (sleep 8 && $*)
}

# This gets set in binary_populate_env.sh, but lets have a sane default just in case
PIP_UPLOAD_FOLDER=${PIP_UPLOAD_FOLDER:-nightly/}
# TODO: Combine CONDA_UPLOAD_CHANNEL and PIP_UPLOAD_FOLDER into one variable
#       The only difference is the trailing slash
# Strip trailing slashes if there
CONDA_UPLOAD_CHANNEL=$(echo "${PIP_UPLOAD_FOLDER}" | sed 's:/*$::')

# Upload the package to the final location
if [[ "$PACKAGE_TYPE" == conda ]]; then
  retry conda install -yq anaconda-client
  anaconda -t "${CONDA_PYTORCHBOT_TOKEN}" upload  "$(ls)" -u "pytorch-${CONDA_UPLOAD_CHANNEL}" --label main --no-progress --force
elif [[ "$PACKAGE_TYPE" == libtorch ]]; then
  retry conda install -c conda-forge -yq awscli
  s3_dir="s3://pytorch/libtorch/${PIP_UPLOAD_FOLDER}${DESIRED_CUDA}/"
  for pkg in $(ls); do
    retry aws s3 cp "$pkg" "$s3_dir" --acl public-read
  done
else
  retry conda install -c conda-forge -yq awscli
  s3_dir="s3://pytorch/whl/${PIP_UPLOAD_FOLDER}${DESIRED_CUDA}/"
  retry aws s3 cp "$(ls)" "$s3_dir" --acl public-read
fi

