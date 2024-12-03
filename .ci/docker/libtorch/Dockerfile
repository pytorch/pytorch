ARG BASE_TARGET=base
ARG GPU_IMAGE=ubuntu:20.04
FROM ${GPU_IMAGE} as base

ENV DEBIAN_FRONTEND=noninteractive

RUN apt-get clean && apt-get update
RUN apt-get install -y curl locales g++ git-all autoconf automake make cmake wget unzip sudo
# Just add everything as a safe.directory for git since these will be used in multiple places with git
RUN git config --global --add safe.directory '*'

RUN locale-gen en_US.UTF-8

ENV LC_ALL en_US.UTF-8
ENV LANG en_US.UTF-8
ENV LANGUAGE en_US.UTF-8

# Install openssl
FROM base as openssl
ADD ./common/install_openssl.sh install_openssl.sh
RUN bash ./install_openssl.sh && rm install_openssl.sh

# Install python
FROM base as python
ADD common/install_cpython.sh install_cpython.sh
RUN apt-get update -y && \
    apt-get install build-essential gdb lcov libbz2-dev libffi-dev \
        libgdbm-dev liblzma-dev libncurses5-dev libreadline6-dev \
        libsqlite3-dev libssl-dev lzma lzma-dev tk-dev uuid-dev zlib1g-dev -y && \
    bash ./install_cpython.sh && \
    rm install_cpython.sh && \
    apt-get clean

FROM base as conda
ADD ./common/install_conda_docker.sh install_conda.sh
RUN bash ./install_conda.sh && rm install_conda.sh

FROM base as cpu
# Install Anaconda
COPY --from=conda /opt/conda /opt/conda
# Install python
COPY --from=python /opt/python    /opt/python
COPY --from=python /opt/_internal /opt/_internal
ENV PATH=/opt/conda/bin:/usr/local/cuda/bin:$PATH
# Install MKL
ADD ./common/install_mkl.sh install_mkl.sh
RUN bash ./install_mkl.sh && rm install_mkl.sh

FROM cpu as cuda
ADD ./common/install_cuda.sh install_cuda.sh
ADD ./common/install_magma.sh install_magma.sh
ENV CUDA_HOME /usr/local/cuda

FROM cuda as cuda11.8
RUN bash ./install_cuda.sh 11.8
RUN bash ./install_magma.sh 11.8
RUN ln -sf /usr/local/cuda-11.8 /usr/local/cuda

FROM cuda as cuda12.1
RUN bash ./install_cuda.sh 12.1
RUN bash ./install_magma.sh 12.1
RUN ln -sf /usr/local/cuda-12.1 /usr/local/cuda

FROM cuda as cuda12.4
RUN bash ./install_cuda.sh 12.4
RUN bash ./install_magma.sh 12.4
RUN ln -sf /usr/local/cuda-12.4 /usr/local/cuda

FROM cuda as cuda12.6
RUN bash ./install_cuda.sh 12.6
RUN bash ./install_magma.sh 12.6
RUN ln -sf /usr/local/cuda-12.6 /usr/local/cuda

FROM cpu as rocm
ARG PYTORCH_ROCM_ARCH
ENV PYTORCH_ROCM_ARCH ${PYTORCH_ROCM_ARCH}
ENV MKLROOT /opt/intel
# Adding ROCM_PATH env var so that LoadHip.cmake (even with logic updated for ROCm6.0)
# find HIP works for ROCm5.7. Not needed for ROCm6.0 and above.
# Remove below when ROCm5.7 is not in support matrix anymore.
ENV ROCM_PATH /opt/rocm
# No need to install ROCm as base docker image should have full ROCm install
#ADD ./common/install_rocm.sh install_rocm.sh
ADD ./common/install_rocm_drm.sh install_rocm_drm.sh
ADD ./common/install_rocm_magma.sh install_rocm_magma.sh
# gfortran and python needed for building magma from source for ROCm
RUN apt-get update -y && \
    apt-get install gfortran -y && \
    apt-get install python -y && \
    apt-get clean

RUN bash ./install_rocm_drm.sh && rm install_rocm_drm.sh
RUN bash ./install_rocm_magma.sh && rm install_rocm_magma.sh

# Install AOTriton
COPY ./common/common_utils.sh common_utils.sh
COPY ./aotriton_version.txt aotriton_version.txt
COPY ./common/install_aotriton.sh install_aotriton.sh
RUN bash ./install_aotriton.sh /opt/rocm && rm install_aotriton.sh aotriton_version.txt
ENV AOTRITON_INSTALLED_PREFIX /opt/rocm/aotriton

FROM ${BASE_TARGET} as final
COPY --from=openssl            /opt/openssl           /opt/openssl
# Install patchelf
ADD ./common/install_patchelf.sh install_patchelf.sh
RUN bash ./install_patchelf.sh && rm install_patchelf.sh
# Install Anaconda
COPY --from=conda /opt/conda /opt/conda
# Install python
COPY --from=python /opt/python    /opt/python
COPY --from=python /opt/_internal /opt/_internal
ENV PATH=/opt/conda/bin:/usr/local/cuda/bin:$PATH
