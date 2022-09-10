#!/usr/bin/env bash
set -x
set -e

unset PYTORCH_VERSION
# For unittest, nightly PyTorch is used as the following section,
# so no need to set PYTORCH_VERSION.
# In fact, keeping PYTORCH_VERSION forces us to hardcode PyTorch version in config.

set -e

eval "$(./conda/bin/conda shell.bash hook)"
conda activate ./env

# if [ "${CU_VERSION:-}" == cpu ] ; then
#     cudatoolkit="cpuonly"
# else
#     if [[ ${#CU_VERSION} -eq 4 ]]; then
#         CUDA_VERSION="${CU_VERSION:2:1}.${CU_VERSION:3:1}"
#     elif [[ ${#CU_VERSION} -eq 5 ]]; then
#         CUDA_VERSION="${CU_VERSION:2:2}.${CU_VERSION:4:1}"
#     fi
#     echo "Using CUDA $CUDA_VERSION as determined by CU_VERSION"
#     version="$(python -c "print('.'.join(\"${CUDA_VERSION}\".split('.')[:2]))")"
#     cudatoolkit="cudatoolkit=${version}"
# fi

WHEELS_FOLDER=${HOME}/project/wheels
mkdir -p $WHEELS_FOLDER

PYVSHORT=${PARAMETERS_PYTHON_VERSION:0:1}${PARAMETERS_PYTHON_VERSION:2:1}

if [[ "$PYVSHORT" == "38" ]] ; then
   PYVSHORT=cp${PYVSHORT}-cp${PYVSHORT}
else
   PYVSHORT=cp${PYVSHORT}-cp${PYVSHORT}m
fi

# if [ "${CU_VERSION:-}" == cpu ] ; then
#     pip install https://download.pytorch.org/whl/nightly/cpu/torch-1.9.0.dev20210427%2Bcpu-${PYVSHORT}-linux_x86_64.whl
#     pip install https://download.pytorch.org/whl/nightly/cpu/torchvision-0.10.0.dev20210427%2Bcpu-${PYVSHORT}-linux_x86_64.whl
#     USE_NINJA=1 python setup.py develop bdist_wheel -d $WHEELS_FOLDER
# else
#     pip install https://download.pytorch.org/whl/nightly/cu102/torch-1.9.0.dev20210427%2Bcu102-${PYVSHORT}-linux_x86_64.whl
#     pip install https://download.pytorch.org/whl/nightly/cu102/torchvision-0.10.0.dev20210427-${PYVSHORT}-linux_x86_64.whl
#     USE_NINJA=1 python setup.py develop bdist_wheel -d $WHEELS_FOLDER
# fi

gcc --version

# TODO: This should really be a part of environment.yml or the docker image.
# expecttest isn't on conda so it can't be a part of environment.yml :/
pip install expecttest

if [ "${CU_VERSION:-}" == cpu ] ; then
    conda install -y pytorch torchvision cpuonly -c pytorch-nightly
    PYTORCH_VERSION="$(python -c "import torch; print(torch.__version__)")" python setup.py develop bdist_wheel -d $WHEELS_FOLDER
else
    conda install -y pytorch torchvision cudatoolkit=10.2 -c pytorch-nightly
    PYTORCH_VERSION="$(python -c "import torch; print(torch.__version__)")" python setup.py develop bdist_wheel -d $WHEELS_FOLDER
fi
