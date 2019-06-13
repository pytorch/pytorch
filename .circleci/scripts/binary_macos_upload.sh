#!/bin/bash
# Do NOT set -x
set -eu -o pipefail
set +x
export AWS_ACCESS_KEY_ID="${PYTORCH_BINARY_AWS_ACCESS_KEY_ID}"
export AWS_SECRET_ACCESS_KEY="${PYTORCH_BINARY_AWS_SECRET_ACCESS_KEY}"
cat >/Users/distiller/project/login_to_anaconda.sh <<EOL
set +x
echo "Trying to login to Anaconda"
yes | anaconda login \
    --username "$PYTORCH_BINARY_PJH5_CONDA_USERNAME" \
    --password "$PYTORCH_BINARY_PJH5_CONDA_PASSWORD"
set -x
EOL
chmod +x /Users/distiller/project/login_to_anaconda.sh

#!#!#!#!#!#!#!#!#!#!#!#!#!#!#!#!#!#!#!#!#!#!#!#!
# DO NOT TURN -x ON BEFORE THIS LINE
#!#!#!#!#!#!#!#!#!#!#!#!#!#!#!#!#!#!#!#!#!#!#!#!
set -eux -o pipefail

source "/Users/distiller/project/env"
export "PATH=$workdir/miniconda/bin:$PATH"

pushd "$workdir/final_pkgs"
if [[ "$PACKAGE_TYPE" == conda ]]; then
  retry conda install -yq anaconda-client
  retry /Users/distiller/project/login_to_anaconda.sh
  retry anaconda upload "$(ls)" -u pytorch --label main --no-progress --force
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
