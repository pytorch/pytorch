# This is the Dockerfile for an image that is ready to build PyTorch from source.
# PyTorch is not yet downloaded nor installed.
# Specification: Ubuntu 18.04, CUDA 10.1, cuDNN 7.6.5
# The ARGs below are currently the only differences between other Ubuntu CUDA builds.
ARG BASE_IMAGE=nvidia/cuda:10.1-cudnn7-devel-ubuntu18.04
ARG PYTHON_VERSION=3.7.6
ARG TORCH_CUDA_ARCH_LIST_VAR="3.7+PTX;5.0;6.0;6.1;7.0;7.5"

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
RUN curl -fsSL -v -o ~/miniconda.sh -O  https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh  && \
    chmod +x ~/miniconda.sh && \
    ~/miniconda.sh -b -p /opt/conda && \
    rm ~/miniconda.sh && \
    /opt/conda/bin/conda install -y python=${PYTHON_VERSION} conda-build pyyaml numpy ipython cython typing typing_extensions mkl mkl-include ninja && \
    /opt/conda/bin/conda clean -ya

RUN conda install -y -c pytorch magma-cuda101 \
 && conda clean -ya

# Necessary step for Azure Pipelines Docker Build
RUN chmod -R ugo+rw /opt/conda/
WORKDIR /opt/pytorch
# Environment variables for PyTorch
ENV TORCH_CUDA_ARCH_LIST=${TORCH_CUDA_ARCH_LIST_VAR}
ENV TORCH_NVCC_FLAGS="-Xfatbin -compress-all"
ENV CMAKE_PREFIX_PATH="$(dirname $(which conda))/../"
ENV USE_CUDA="1"

# Install Azure CLI and update its site packages
RUN curl -sL https://aka.ms/InstallAzureCLIDeb | bash
RUN pip install --upgrade pip --target /opt/az/lib/python3.6/site-packages/
