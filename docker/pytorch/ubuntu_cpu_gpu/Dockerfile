# This is the Dockerfile for an image that is ready to build PyTorch from source.
# PyTorch is not yet downloaded nor installed.
#
# Available BASE_IMAGE options:
#   nvidia/cuda:11.2.1-cudnn8-devel-ubuntu18.04
#   nvidia/cuda:10.2-cudnn8-devel-ubuntu18.04
#   nvidia/cuda:10.1-cudnn7-devel-ubuntu18.04
#   nvidia/cuda:9.2-cudnn7-devel-ubuntu18.04
#
# Available MAGMA_CUDA_VERSION options (for GPU/CUDA builds):
#   magma-cuda112
#   magma-cuda111
#   magma-cuda102
#   magma-cuda101
#   magma-cuda92
#
# Available TORCH_CUDA_ARCH_LIST_VAR options (for GPU/CUDA builds):
#   "3.7+PTX;5.0;6.0;6.1;7.0;7.5;8.0;8.6" for CUDA 11.2/11.1
#   "3.7+PTX;5.0;6.0;6.1;7.0;7.5;8.0" for CUDA 11.0
#   "3.7+PTX;5.0;6.0;6.1;7.0;7.5" for CUDA 10.2/10.1
#   "3.7+PTX;5.0;6.0;6.1;7.0" for CUDA 9.2
#
# Build image with CPU or GPU support with the following command:
#   nvidia-docker build -t ${CONTAINER_TAG}
#   --build-arg BASE_IMAGE=${BASE_IMAGE_VER} \
#   --build-arg PYTHON_VERSION=${PYTHON_VER} \
#   --build-arg MAGMA_CUDA_VERSION=${MAGMA_CUDA_VER} \ #(for GPU/CUDA builds)
#   --build-arg TORCH_CUDA_ARCH_LIST_VAR=${TORCH_CUDA_ARCH_LIST} \ #(for GPU/CUDA builds):
#   .
#
# For example, for a CPU Ubuntu 18.04 and Python 3.7.6 build:
#   docker build -t ubuntu_1804_py_37_cpu_dev \
#   --build-arg BASE_IMAGE=ubuntu:18.04 \
#   --build-arg PYTHON_VERSION=3.7.6 .
#
# For example, for a CUDA 10.2 Ubuntu 18.04 and Python 3.9.1 build:
#   nvidia-docker build -t ubuntu_1804_py_39_cuda_102_cudnn_8_dev \
#   --build-arg BASE_IMAGE=nvidia/cuda:10.2-cudnn8-devel-ubuntu18.04 \
#   --build-arg PYTHON_VERSION=3.9.1 \
#   --build-arg MAGMA_CUDA_VERSION=magma-cuda102 \
#   --build-arg TORCH_CUDA_ARCH_LIST_VAR="3.7+PTX;5.0;6.0;6.1;7.0;7.5" .

ARG BASE_IMAGE
FROM ${BASE_IMAGE} as dev-base
RUN apt-get update && apt-get install -y --no-install-recommends \
        build-essential \
        ca-certificates \
        ccache \
        cmake \
        curl \
        git \
        libjpeg-dev \
        libpng-dev \
        wget && \
    rm -rf /var/lib/apt/lists/*
RUN /usr/sbin/update-ccache-symlinks
RUN mkdir /opt/ccache && ccache --set-config=cache_dir=/opt/ccache
ENV PATH /opt/conda/bin:$PATH

FROM dev-base as conda
ARG PYTHON_VERSION
ENV PYTHON_VER=$PYTHON_VERSION
RUN curl -fsSL -v -o ~/miniconda.sh -O  https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh  && \
    chmod +x ~/miniconda.sh && \
    ~/miniconda.sh -b -p /opt/conda && \
    rm ~/miniconda.sh && \
    /opt/conda/bin/conda install -y python=${PYTHON_VER} conda-build pyyaml numpy ipython cython typing typing_extensions mkl mkl-include ninja && \
    /opt/conda/bin/conda clean -ya

ARG MAGMA_CUDA_VERSION
RUN if [ -z "$MAGMA_CUDA_VERSION" ] ; then \
    echo "Building with CPU support ..."; \
  else \
    echo "Building with GPU/CUDA support ..."; \
    conda install -y -c pytorch ${MAGMA_CUDA_VERSION} && conda clean -ya; \
  fi

# Necessary step for Azure Pipelines Docker Build
# Docker image is build by root, but the build process
# is running from a non-priveledged user
RUN chmod -R ugo+rw /opt/conda/

WORKDIR /opt/pytorch
# Environment variables for PyTorch
ARG TORCH_CUDA_ARCH_LIST_VAR
RUN if [ -z "$TORCH_CUDA_ARCH_LIST_VAR" ] ; then \
    echo "Continuing CPU build ..."; \
  else \
    echo "Setting CUDA env vars ..."; \
  fi
# If the build argument TORCH_CUDA_ARCH_LIST_VAR is given, container will be
# set for GPU/CUDA build, else for CPU build.
ENV TORCH_CUDA_ARCH_LIST=${TORCH_CUDA_ARCH_LIST_VAR:+${TORCH_CUDA_ARCH_LIST_VAR}}
ENV TORCH_NVCC_FLAGS=${TORCH_CUDA_ARCH_LIST_VAR:+"-Xfatbin -compress-all"}
ENV CMAKE_PREFIX_PATH="$(dirname $(which conda))/../"

# Install Azure CLI and update its site packages
RUN curl -sL https://aka.ms/InstallAzureCLIDeb | bash
RUN pip install --upgrade pip --target /opt/az/lib/python3.6/site-packages/

# Install MKL
RUN wget https://raw.githubusercontent.com/pytorch/builder/f121b0919d799b5ea2030c92ca266cf4cddf6656/common/install_mkl.sh
RUN bash ./install_mkl.sh && rm install_mkl.sh
