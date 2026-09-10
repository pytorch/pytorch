ARG CUDA_VERSION=12.6
ARG BASE_TARGET=cuda${CUDA_VERSION}
ARG BASE_IMAGE=amd64/almalinux:8.10-20250519
FROM ${BASE_IMAGE} as base

ENV LC_ALL en_US.UTF-8
ENV LANG en_US.UTF-8
ENV LANGUAGE en_US.UTF-8

ARG DEVTOOLSET_VERSION=13

RUN yum -y update
RUN yum -y install epel-release
# install glibc-langpack-en make sure en_US.UTF-8 locale is available
RUN yum -y install glibc-langpack-en
RUN yum install -y sudo wget curl perl util-linux xz bzip2 git patch which perl zlib-devel openssl-devel yum-utils autoconf automake make gcc-toolset-${DEVTOOLSET_VERSION}-gcc gcc-toolset-${DEVTOOLSET_VERSION}-gcc-c++ gcc-toolset-${DEVTOOLSET_VERSION}-gcc-gfortran gcc-toolset-${DEVTOOLSET_VERSION}-gdb
# Just add everything as a safe.directory for git since these will be used in multiple places with git
RUN git config --global --add safe.directory '*'
ENV PATH=/opt/rh/gcc-toolset-${DEVTOOLSET_VERSION}/root/usr/bin:$PATH

# cmake-3.18.4 from pip
# NS: Apr 1 2026 3.18.4 is gone, reported here https://github.com/scikit-build/cmake-python-distributions/issues/693
# pip 9 mishandles truthy PIP_NO_CACHE_DIR values; fixed in pip 19:
# https://github.com/pypa/pip/pull/5884
RUN yum install -y python3-pip && \
    python3 -mpip install --no-cache-dir cmake==3.18.4.post1 && \
    ln -s /usr/local/bin/cmake /usr/bin/cmake3
ENV PIP_NO_CACHE_DIR=1
RUN rm -rf /usr/local/cuda-*

FROM base as patchelf
# Install patchelf
ADD ./common/install_patchelf.sh install_patchelf.sh
RUN bash ./install_patchelf.sh && rm install_patchelf.sh && cp $(which patchelf) /patchelf

FROM base as conda
# Install Anaconda
ADD ./common/install_conda_docker.sh install_conda.sh
RUN bash ./install_conda.sh && rm install_conda.sh

# Install CUDA
FROM base as cuda
ARG CUDA_VERSION=12.6
ARG DEVTOOLSET_VERSION=13
RUN rm -rf /usr/local/cuda-*
ADD ./common/install_cuda.sh install_cuda.sh
COPY ./common/install_nccl.sh install_nccl.sh
COPY ./ci_commit_pins/nccl* /ci_commit_pins/
COPY ./common/install_cusparselt.sh install_cusparselt.sh
ENV CUDA_HOME=/usr/local/cuda-${CUDA_VERSION}
# Preserve CUDA_VERSION for the builds
ENV CUDA_VERSION=${CUDA_VERSION}
# Make things in our path by default
ENV PATH=/usr/local/cuda-${CUDA_VERSION}/bin:/opt/rh/gcc-toolset-${DEVTOOLSET_VERSION}/root/usr/bin:$PATH

FROM cuda as cuda12.6
RUN bash ./install_cuda.sh 12.6
ENV DESIRED_CUDA=12.6

FROM cuda as cuda12.8
RUN bash ./install_cuda.sh 12.8
ENV DESIRED_CUDA=12.8

FROM cuda as cuda12.9
RUN bash ./install_cuda.sh 12.9
ENV DESIRED_CUDA=12.9

FROM cuda as cuda13.0
RUN bash ./install_cuda.sh 13.0
ENV DESIRED_CUDA=13.0

FROM cuda as cuda13.2
RUN bash ./install_cuda.sh 13.2
ENV DESIRED_CUDA=13.2

FROM cuda as cuda13.4
RUN bash ./install_cuda.sh 13.4
ENV DESIRED_CUDA=13.4

FROM base as rocm
ARG PYTORCH_ROCM_ARCH
ARG ROCM_VERSION
ARG THEROCK_INDEX_URL
ENV PYTORCH_ROCM_ARCH ${PYTORCH_ROCM_ARCH}
# TheRock wheels require Python 3.10+; AlmaLinux 8's system Python is too old.
COPY --from=conda /opt/conda /opt/conda
ENV PATH=/opt/conda/bin:$PATH
ADD ./common/install_mkl.sh install_mkl.sh
RUN bash ./install_mkl.sh && rm install_mkl.sh
ENV MKLROOT /opt/intel
# torch extensions link libatomic.so.1, which the plain AlmaLinux base lacks.
RUN yum install -y libatomic
ADD ./common/install_rocm.sh install_rocm.sh
RUN bash ./install_rocm.sh && rm install_rocm.sh
# Unlike manywheel, this image has no build_env_setup.py to read /etc/rocm_env.sh,
# and install_rocm.sh only hooks /etc/bashrc, which non-interactive bash skips.
ENV BASH_ENV=/etc/rocm_env.sh

# Install MNIST test data
FROM base as mnist
ADD ./common/install_mnist.sh install_mnist.sh
RUN bash ./install_mnist.sh

FROM base as all_cuda
COPY --from=cuda12.6  /usr/local/cuda-12.6 /usr/local/cuda-12.6
COPY --from=cuda12.8  /usr/local/cuda-12.8 /usr/local/cuda-12.8
COPY --from=cuda12.9  /usr/local/cuda-12.9 /usr/local/cuda-12.9
COPY --from=cuda13.0  /usr/local/cuda-13.0 /usr/local/cuda-13.0
COPY --from=cuda13.2  /usr/local/cuda-13.2 /usr/local/cuda-13.2
COPY --from=cuda13.4  /usr/local/cuda-13.4 /usr/local/cuda-13.4

# Final step
FROM ${BASE_TARGET} as final
ARG DEVTOOLSET_VERSION=13
COPY --from=patchelf           /patchelf              /usr/local/bin/patchelf
COPY --from=conda              /opt/conda             /opt/conda

# Add jni.h for java host build.
COPY ./common/install_jni.sh install_jni.sh
COPY ./java/jni.h jni.h
RUN bash ./install_jni.sh && rm install_jni.sh

ENV PATH /opt/conda/bin:$PATH
ENV LD_LIBRARY_PATH=/opt/rh/gcc-toolset-${DEVTOOLSET_VERSION}/root/usr/lib64:/opt/rh/gcc-toolset-${DEVTOOLSET_VERSION}/root/usr/lib:$LD_LIBRARY_PATH
COPY --from=mnist  /usr/local/mnist /usr/local/mnist
RUN rm -rf /usr/local/cuda
RUN chmod o+rw /usr/local
RUN touch /.condarc && \
    chmod o+rw /.condarc && \
    chmod -R o+rw /opt/conda
