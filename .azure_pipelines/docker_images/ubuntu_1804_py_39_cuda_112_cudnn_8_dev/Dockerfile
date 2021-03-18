# This is the Dockerfile for an image that is ready to build PyTorch from source.
# PyTorch is not yet downloaded nor installed.
# Specification: Ubuntu 18.04, CUDA 11.2, cuDNN 8.1.0
# The ARGs below are currently the only differences between other Ubuntu CUDA builds.
#   PYTHON_VERSION
#   TORCH_CUDA_ARCH_LIST_VAR

ARG BASE_IMAGE=nvidia/cuda:11.2.1-cudnn8-devel-ubuntu18.04
FROM ${BASE_IMAGE} as dev-base
RUN apt-get update && apt-get install -y --no-install-recommends \
        build-essential \
        ca-certificates \
        ccache \
        cmake \
        curl \
        git \
        libjpeg-dev \
        libpng-dev && \
    rm -rf /var/lib/apt/lists/*
RUN /usr/sbin/update-ccache-symlinks
RUN mkdir /opt/ccache && ccache --set-config=cache_dir=/opt/ccache
ENV PATH /opt/conda/bin:$PATH

FROM dev-base as conda
ARG PYTHON_VERSION=3.9.1
RUN curl -fsSL -v -o ~/miniconda.sh -O  https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh  && \
    chmod +x ~/miniconda.sh && \
    ~/miniconda.sh -b -p /opt/conda && \
    rm ~/miniconda.sh && \
    /opt/conda/bin/conda install -y python=${PYTHON_VERSION} conda-build pyyaml numpy ipython cython typing typing_extensions mkl mkl-include ninja && \
    /opt/conda/bin/conda clean -ya

RUN conda install -y -c pytorch magma-cuda112 \
 && conda clean -ya

# Necessary step for Azure Pipelines Docker Build
RUN chmod -R ugo+rw /opt/conda/
WORKDIR /opt/pytorch
# Environment variables for PyTorch
ARG TORCH_CUDA_ARCH_LIST_VAR="3.7+PTX;5.0;6.0;6.1;7.0;7.5;8.0;8.6"
ENV TORCH_CUDA_ARCH_LIST=${TORCH_CUDA_ARCH_LIST_VAR}
ENV TORCH_NVCC_FLAGS="-Xfatbin -compress-all"
ENV CMAKE_PREFIX_PATH="$(dirname $(which conda))/../"
ENV USE_CUDA="1"

# Install Azure CLI and update its site packages
RUN curl -sL https://aka.ms/InstallAzureCLIDeb | bash
RUN pip install --upgrade pip --target /opt/az/lib/python3.6/site-packages/

# Install MKL
ADD ./scripts/install_mkl.sh install_mkl.sh
RUN bash ./install_mkl.sh && rm install_mkl.sh
