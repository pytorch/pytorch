# This is the Dockerfile for an image that is ready to build PyTorch from source.
# PyTorch is not yet downloaded nor installed.
# 
# Build image with the following command:
#   docker build -t ${CONTAINER_TAG} \
#   --build-arg BASE_IMAGE=${BASE_IMAGE_VER} \
#   --build-arg PYTHON_VERSION=${PYTHON_VER} .
#
# For example, with image Ubuntu 18.04 and Python 3.7.6:
#   docker build -t ubuntu_1804_py_37_cpu_dev \
#   --build-arg BASE_IMAGE=ubuntu:18.04 \
#   --build-arg PYTHON_VERSION=3.7.6 .


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

# Necessary step for Azure Pipelines Docker Build
RUN chmod -R ugo+rw /opt/conda/
WORKDIR /opt/pytorch
# Environment variables for PyTorch
ENV CMAKE_PREFIX_PATH="$(dirname $(which conda))/../"
ENV USE_CUDA="0"

# Install Azure CLI and update its site packages
RUN curl -sL https://aka.ms/InstallAzureCLIDeb | bash
RUN pip install --upgrade pip --target /opt/az/lib/python3.6/site-packages/

# Install MKL
RUN wget https://raw.githubusercontent.com/pytorch/builder/master/common/install_mkl.sh
RUN bash ./install_mkl.sh && rm install_mkl.sh
