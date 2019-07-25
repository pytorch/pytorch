#!/bin/bash
# Do NOT set -x
source /home/circleci/project/env
set -eu -o pipefail
set +x
declare -x "AWS_ACCESS_KEY_ID=${PYTORCH_BINARY_AWS_ACCESS_KEY_ID}"
declare -x "AWS_SECRET_ACCESS_KEY=${PYTORCH_BINARY_AWS_SECRET_ACCESS_KEY}"
cat >/home/circleci/project/login_to_anaconda.sh <<EOL
set +x
echo "Trying to login to Anaconda"
yes | anaconda login \
    --username "$PYTORCH_BINARY_PJH5_CONDA_USERNAME" \
    --password "$PYTORCH_BINARY_PJH5_CONDA_PASSWORD"
set -x
EOL
chmod +x /home/circleci/project/login_to_anaconda.sh

#!#!#!#!#!#!#!#!#!#!#!#!#!#!#!#!#!#!#!#!#!#!#!#!
# DO NOT TURN -x ON BEFORE THIS LINE
#!#!#!#!#!#!#!#!#!#!#!#!#!#!#!#!#!#!#!#!#!#!#!#!
set -eux -o pipefail
export PATH="$MINICONDA_ROOT/bin:$PATH"

# Upload the package to the final location
pushd /home/circleci/project/final_pkgs
if [[ "$PACKAGE_TYPE" == conda ]]; then
  retry conda install -yq anaconda-client
  retry timeout 30 /home/circleci/project/login_to_anaconda.sh
  anaconda upload "$(ls)" -u pytorch --label main --no-progress --force
elif [[ "$PACKAGE_TYPE" == libtorch ]]; then
  retry pip install -q awscli
  s3_dir="s3://pytorch/libtorch/${PIP_UPLOAD_FOLDER}${DESIRED_CUDA}/"
  retry aws s3 cp "$(ls)" "$s3_dir" --acl public-read
else
  retry pip install -q awscli
  s3_dir="s3://pytorch/whl/${PIP_UPLOAD_FOLDER}${DESIRED_CUDA}/"
  retry aws s3 cp "$(ls)" "$s3_dir" --acl public-read
fi
