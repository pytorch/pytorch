# syntax = docker/dockerfile:experimental
#
# NOTE: To build this you will need a docker version > 18.06 with
#       experimental enabled and DOCKER_BUILDKIT=1
#
#       For reference:
#           https://docs.docker.com/develop/develop-images/build_enhancements/
#
# This Dockerfile will build Docker Image with PyTorch + DNNL + AMD BLIS and Torchvision installed for CPU only
#
# Example commandline to build PyTorch with AMD BLIS:
# sudo DOCKER_BUILDKIT=1 docker build . -t docker-image-repo-name
# Example commandline to run the built docker container:
# sudo docker run --name container-name -it docker-image-repo-name

ARG BASE_IMAGE=ubuntu:18.04
ARG PYTHON_VERSION=3.8

FROM ${BASE_IMAGE} as dev-base
CMD echo "Welcome to the PyTorch Docker Container!" && \
    echo "Version of PyTorch Installed: " && python -c 'import torch; print(torch.__version__)' && \
    echo "Version of Torchvision Installed: " && python -c 'import torchvision; print(torchvision.__version__)' && \
    echo "LDD output showing successful linking with BLIS: " && ldd /opt/conda/lib/python3.8/site-packages/torch/_C.cpython-38-x86_64-linux-gnu.so && \
    /bin/bash
RUN --mount=type=cache,id=apt-dev,target=/var/cache/apt \
    apt-get update && apt-get install -y --no-install-recommends \
        build-essential \
        ca-certificates \
        ccache \
        cmake \
        curl \
        git \
        libjpeg-dev \
        libpng-dev \
        vim \
        wget && \
    rm -rf /var/lib/apt/lists/*
RUN /usr/sbin/update-ccache-symlinks
RUN mkdir /opt/ccache && ccache --set-config=cache_dir=/opt/ccache
ENV PATH /opt/conda/bin:$PATH

FROM dev-base as conda
RUN wget https://repo.continuum.io/miniconda/Miniconda3-latest-Linux-x86_64.sh -O ~/miniconda.sh && \
    chmod +x ~/miniconda.sh && \
    ~/miniconda.sh -b -p /opt/conda && \
    rm ~/miniconda.sh && \
    /opt/conda/bin/conda install -y python=${PYTHON_VERSION} conda-build && \
    /opt/conda/bin/conda install -y nomkl pyyaml numpy ipython ninja setuptools cmake cffi typing future && \
    /opt/conda/bin/conda clean -ya

RUN conda install typing_extensions

WORKDIR /root
ARG BLIS_URL=https://github.com/amd/blis.git
# Download, Build BLIS with multithreading support and place necessary library and include files at BLIS_HOME/lib and BLIS_HOME/include respectively
RUN git clone ${BLIS_URL} && cd blis && \
    ./configure --prefix=/root/BLISBuild --enable-cblas --enable-threading=openmp auto && make -j && make install && \
    if [ ! -e /root/BLISBuild/lib/libblis.so ] ; then cp /root/BLISBuild/lib/libblis*.so /root/BLISBuild/lib/libblis.so ; fi

# Build PyTorch with DNNL+BLIS
RUN git clone https://github.com/pytorch/pytorch.git && cd pytorch && \
    git submodule update --init --recursive && \
    export PATH=/root/BLISBuild/include/blis:$PATH LD_LIBRARY_PATH=/root/BLISBuild/lib:$LD_LIBRARY_PATH && \
    export BLIS_HOME=/root/BLISBuild BLAS=BLIS USE_MKLDNN_CBLAS=ON WITH_BLAS=blis && python setup.py install

# Build Torchvision
RUN git clone https://github.com/pytorch/vision.git && cd vision && \
    python setup.py install
