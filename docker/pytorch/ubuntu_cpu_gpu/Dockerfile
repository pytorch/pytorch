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

ENV TZ=America/Chicago
RUN ln -snf /usr/share/zoneinfo/$TZ /etc/localtime && echo $TZ > /etc/timezone

RUN apt-get update && apt-get install -y --no-install-recommends \
        build-essential \
        ca-certificates \
        ccache \
        cmake \
        vim \
        ssh \
        curl \
        git \
        git-lfs \
        libjpeg-dev \
        libpng-dev \
        openmpi-bin \
        wget \
        clang-11 \
        clang-11-doc \
        clang-11-examples \
        clang-format-11 \
        clang-tidy-11 \
        clang-tools-11 \
        clangd-11 \
        libc++-11-dev \
        libc++1-11 \
        libc++abi-11-dev \
        libc++abi1-11 \
        libclang-11-dev \
        libclang-common-11-dev \
        libclang-cpp11 \
        libclang-cpp11-dev \
        libclang1-11 \
        libfuzzer-11-dev \
        liblld-11 \
        liblld-11-dev \
        liblldb-11 \
        liblldb-11-dev \
        libllvm-11-ocaml-dev \
        libllvm11 \
        libomp-11-dev \
        libomp-11-doc \
        libomp5-11 \
        lld-11 \
        lldb-11 \
        llvm-11 \
        llvm-11-dev \
        llvm-11-doc \
        llvm-11-examples \
        llvm-11-runtime \
        llvm-11-tools \
        python3-clang-11 \
        openjdk-14-jdk \
        rsync \
        gdb \
        ninja-build \
        python3-lldb-11 && rm -rf /var/lib/apt/lists/*

RUN ( \
    echo 'LogLevel DEBUG2'; \
    echo 'PermitRootLogin yes'; \
    echo 'PasswordAuthentication yes'; \
    echo 'Subsystem sftp /usr/lib/openssh/sftp-server'; \
  ) > /etc/ssh/sshd_config \
  && mkdir /run/sshd

RUN apt-get update && apt-get -y install openssl libssl-dev

RUN wget https://github.com/Kitware/CMake/releases/download/v3.20.2/cmake-3.20.2.tar.gz && \
    tar -zxvf cmake-3.20.2.tar.gz && \
    cd cmake-3.20.2 && \
    ./bootstrap && \
    make && \
    make install -j 64

RUN  yes password | passwd

RUN /usr/sbin/update-ccache-symlinks
RUN mkdir /opt/ccache && ccache --set-config=cache_dir=/opt/ccache
ENV PATH /opt/conda/bin:/usr/lib/llvm-11/bin/:$PATH



FROM dev-base as conda


ARG username=user
ARG uid=1000
ARG gid=100
ENV USER $username
ENV UID $uid
ENV GID $gid
ENV HOME /home/$USER
RUN adduser \
    --gecos "Non-root user" \
    --uid $UID \
    --gid $GID \
    --home $HOME \
    $USER

RUN usermod -s /bin/bash user

RUN yes password | passwd user

# Necessary step for Azure Pipelines Docker Build
# Docker image is build by root, but the build process
# is running from a non-priveledged user


ARG PYTHON_VERSION
ENV PYTHON_VER=$PYTHON_VERSION
RUN wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh && \
    chmod +x Miniconda3-latest-Linux-x86_64.sh && \
    ./Miniconda3-latest-Linux-x86_64.sh -b -p /opt/conda && \
    rm Miniconda3-latest-Linux-x86_64.sh && \
    /opt/conda/bin/conda install -y python=${PYTHON_VER} conda-build pyyaml numpy ipython cython typing typing_extensions mkl mkl-include ninja && \
    /opt/conda/bin/conda clean -ya

RUN chmod -R ugo+rw /opt/conda/

USER $USER

# make non-activate conda commands available
ENV PATH=/opt/conda/bin:$PATH
# make conda activate command available from /bin/bash --login shells
RUN echo ". /opt/conda/etc/profile.d/conda.sh" >> ~/.profile
# make conda activate command available from /bin/bash --interative shells
RUN conda init bash

USER root

WORKDIR /opt/pytorch
RUN chmod -R ugo+rw /opt/pytorch/
# Environment variables for PyTorch
ARG TORCH_CUDA_ARCH_LIST_VAR
RUN if [ -z "$TORCH_CUDA_ARCH_LIST_VAR" ] ; then \
    echo "Continuing CPU build ..."; \
  else \
    echo "Setting CUDA env vars and installing openmpi ..."; \
    # Set MPI links to avoid libmpi_cxx.so.1 not found error
    ln -s /usr/lib/x86_64-linux-gnu/libmpi_cxx.so.20 /usr/lib/x86_64-linux-gnu/libmpi_cxx.so.1; \
    ln -s /usr/lib/x86_64-linux-gnu/libmpi.so.20.10.1 /usr/lib/x86_64-linux-gnu/libmpi.so.12; \
  fi
# If the build argument TORCH_CUDA_ARCH_LIST_VAR is given, container will be
# set for GPU/CUDA build, else for CPU build.


USER $USER

ENV TORCH_CUDA_ARCH_LIST=${TORCH_CUDA_ARCH_LIST_VAR:+${TORCH_CUDA_ARCH_LIST_VAR}}
ENV TORCH_NVCC_FLAGS=${TORCH_CUDA_ARCH_LIST_VAR:+"-Xfatbin -compress-all"}
ENV CMAKE_PREFIX_PATH="/opt/conda/bin/../"

## Install MKL
#RUN wget https://raw.githubusercontent.com/pytorch/builder/f121b0919d799b5ea2030c92ca266cf4cddf6656/common/install_mkl.sh
#RUN bash ./install_mkl.sh && rm install_mkl.sh

USER root

COPY entrypoint.sh /usr/local/bin/
RUN chown $UID:$GID /usr/local/bin/entrypoint.sh && \
    chmod u+x /usr/local/bin/entrypoint.sh

RUN chmod -R 777 /opt
RUN chown -R user /opt

ENTRYPOINT service ssh restart && bash
