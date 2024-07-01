#! /bin/bash
set -e

if [ -z "${PYTORCH_REPO_PATH}" ]; then
  echo "Please specify \`PYTORCH_REPO_PATH\`"
  exit 1
fi

cd "${PYTORCH_REPO_PATH}"

# Helper function to manage conda environments.
make_clean_env(){
    ENV_NAME="$1"

    # Cleanup prior if it exists
    conda env remove --name "${ENV_NAME}" 2> /dev/null || true

    conda create -yn "${ENV_NAME}" python=3
    source activate "${ENV_NAME}"
    conda install -y numpy ninja pyyaml mkl mkl-include setuptools cmake hypothesis
    conda install -y -c pytorch magma-cuda102
    conda deactivate
}

make_clean_env ref_39850
make_clean_env pr_39850
make_clean_env ref_39967
make_clean_env pr_39967
make_clean_env ref_39744
make_clean_env pr_39744

git submodule update --init --recursive
wget https://github.com/pytorch/pytorch/pull/39850.diff
wget https://github.com/pytorch/pytorch/pull/39744.diff

source activate ref_39850
git checkout 766889b6bfeb9802f892fa0782d629554b2c71b4
python setup.py install
conda deactivate

source activate pr_39850
git apply 39850.diff
python setup.py install
git checkout .
conda deactivate

source activate ref_39967
git checkout c4fc278fa8cbd0fb45b8130a679c62f673087484
python setup.py install
conda deactivate

source activate pr_39967
git checkout 7a3c223bbb711c7a93910ce406a0126b8000b43b
python setup.py install
conda deactivate

source activate ref_39744
git checkout 541814f2b7eacabacdc87ccb1b4495bf486f501a
python setup.py install
conda deactivate

source activate pr_39744
git apply 39744.diff
python setup.py install
git checkout .
conda deactivate
