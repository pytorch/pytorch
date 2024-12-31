# syntax = docker/dockerfile:experimental
ARG ROCM_VERSION=3.7
ARG BASE_CUDA_VERSION=11.8

ARG GPU_IMAGE=centos:7
FROM centos:7 as base

ENV LC_ALL en_US.UTF-8
ENV LANG en_US.UTF-8
ENV LANGUAGE en_US.UTF-8

ARG DEVTOOLSET_VERSION=9

# Note: This is required patch since CentOS have reached EOL
# otherwise any yum install setp will fail
RUN sed -i s/mirror.centos.org/vault.centos.org/g /etc/yum.repos.d/*.repo
RUN sed -i s/^#.*baseurl=http/baseurl=http/g /etc/yum.repos.d/*.repo
RUN sed -i s/^mirrorlist=http/#mirrorlist=http/g /etc/yum.repos.d/*.repo
RUN yum install -y wget curl perl util-linux xz bzip2 git patch which perl zlib-devel
# Just add everything as a safe.directory for git since these will be used in multiple places with git
RUN git config --global --add safe.directory '*'
RUN yum install -y yum-utils centos-release-scl
RUN yum-config-manager --enable rhel-server-rhscl-7-rpms
# Note: After running yum-config-manager --enable rhel-server-rhscl-7-rpms
# patch is required once again. Somehow this steps adds mirror.centos.org
RUN sed -i s/mirror.centos.org/vault.centos.org/g /etc/yum.repos.d/*.repo
RUN sed -i s/^#.*baseurl=http/baseurl=http/g /etc/yum.repos.d/*.repo
RUN sed -i s/^mirrorlist=http/#mirrorlist=http/g /etc/yum.repos.d/*.repo
RUN yum install -y devtoolset-${DEVTOOLSET_VERSION}-gcc devtoolset-${DEVTOOLSET_VERSION}-gcc-c++ devtoolset-${DEVTOOLSET_VERSION}-gcc-gfortran devtoolset-${DEVTOOLSET_VERSION}-binutils
ENV PATH=/opt/rh/devtoolset-${DEVTOOLSET_VERSION}/root/usr/bin:$PATH
ENV LD_LIBRARY_PATH=/opt/rh/devtoolset-${DEVTOOLSET_VERSION}/root/usr/lib64:/opt/rh/devtoolset-${DEVTOOLSET_VERSION}/root/usr/lib:$LD_LIBRARY_PATH

RUN yum --enablerepo=extras install -y epel-release

# cmake-3.18.4 from pip
RUN yum install -y python3-pip && \
    python3 -mpip install cmake==3.18.4 && \
    ln -s /usr/local/bin/cmake /usr/bin/cmake

RUN yum install -y autoconf aclocal automake make sudo

FROM base as openssl
# Install openssl (this must precede `build python` step)
# (In order to have a proper SSL module, Python is compiled
# against a recent openssl [see env vars above], which is linked
# statically. We delete openssl afterwards.)
ADD ./common/install_openssl.sh install_openssl.sh
RUN bash ./install_openssl.sh && rm install_openssl.sh

# EPEL for cmake
FROM base as patchelf
# Install patchelf
ADD ./common/install_patchelf.sh install_patchelf.sh
RUN bash ./install_patchelf.sh && rm install_patchelf.sh
RUN cp $(which patchelf) /patchelf

FROM patchelf as python
# build python
COPY manywheel/build_scripts /build_scripts
ADD ./common/install_cpython.sh /build_scripts/install_cpython.sh
RUN bash build_scripts/build.sh && rm -r build_scripts

FROM base as cuda
ARG BASE_CUDA_VERSION=10.2
# Install CUDA
ADD ./common/install_cuda.sh install_cuda.sh
RUN bash ./install_cuda.sh ${BASE_CUDA_VERSION} && rm install_cuda.sh

FROM base as intel
# MKL
ADD ./common/install_mkl.sh install_mkl.sh
RUN bash ./install_mkl.sh && rm install_mkl.sh

FROM base as magma
ARG BASE_CUDA_VERSION=10.2
# Install magma
ADD ./common/install_magma.sh install_magma.sh
RUN bash ./install_magma.sh ${BASE_CUDA_VERSION} && rm install_magma.sh

FROM base as jni
# Install java jni header
ADD ./common/install_jni.sh install_jni.sh
ADD ./java/jni.h jni.h
RUN bash ./install_jni.sh && rm install_jni.sh

FROM base as libpng
# Install libpng
ADD ./common/install_libpng.sh install_libpng.sh
RUN bash ./install_libpng.sh && rm install_libpng.sh

FROM ${GPU_IMAGE} as common
RUN sed -i s/mirror.centos.org/vault.centos.org/g /etc/yum.repos.d/*.repo
RUN sed -i s/^#.*baseurl=http/baseurl=http/g /etc/yum.repos.d/*.repo
RUN sed -i s/^mirrorlist=http/#mirrorlist=http/g /etc/yum.repos.d/*.repo
ENV LC_ALL en_US.UTF-8
ENV LANG en_US.UTF-8
ENV LANGUAGE en_US.UTF-8
RUN yum install -y \
        aclocal \
        autoconf \
        automake \
        bison \
        bzip2 \
        curl \
        diffutils \
        file \
        git \
        make \
        patch \
        perl \
        unzip \
        util-linux \
        wget \
        which \
        xz \
        yasm
RUN yum install -y \
    https://repo.ius.io/ius-release-el7.rpm \
    https://ossci-linux.s3.amazonaws.com/epel-release-7-14.noarch.rpm

RUN yum swap -y git git236-core
# git236+ would refuse to run git commands in repos owned by other users
# Which causes version check to fail, as pytorch repo is bind-mounted into the image
# Override this behaviour by treating every folder as safe
# For more details see https://github.com/pytorch/pytorch/issues/78659#issuecomment-1144107327
RUN git config --global --add safe.directory "*"

ENV SSL_CERT_FILE=/opt/_internal/certs.pem
# Install LLVM version
COPY --from=openssl            /opt/openssl                          /opt/openssl
COPY --from=python             /opt/python                           /opt/python
COPY --from=python             /opt/_internal                        /opt/_internal
COPY --from=python             /opt/python/cp39-cp39/bin/auditwheel /usr/local/bin/auditwheel
COPY --from=intel              /opt/intel                            /opt/intel
COPY --from=patchelf           /usr/local/bin/patchelf               /usr/local/bin/patchelf
COPY --from=jni                /usr/local/include/jni.h              /usr/local/include/jni.h
COPY --from=libpng             /usr/local/bin/png*                   /usr/local/bin/
COPY --from=libpng             /usr/local/bin/libpng*                /usr/local/bin/
COPY --from=libpng             /usr/local/include/png*               /usr/local/include/
COPY --from=libpng             /usr/local/include/libpng*            /usr/local/include/
COPY --from=libpng             /usr/local/lib/libpng*                /usr/local/lib/
COPY --from=libpng             /usr/local/lib/pkgconfig              /usr/local/lib/pkgconfig

FROM common as cpu_final
ARG BASE_CUDA_VERSION=10.1
ARG DEVTOOLSET_VERSION=9
# Install Anaconda
ADD ./common/install_conda_docker.sh install_conda.sh
RUN bash ./install_conda.sh && rm install_conda.sh
ENV PATH /opt/conda/bin:$PATH
RUN sed -i s/mirror.centos.org/vault.centos.org/g /etc/yum.repos.d/*.repo
RUN sed -i s/^#.*baseurl=http/baseurl=http/g /etc/yum.repos.d/*.repo
RUN sed -i s/^mirrorlist=http/#mirrorlist=http/g /etc/yum.repos.d/*.repo

RUN yum install -y yum-utils centos-release-scl
RUN yum-config-manager --enable rhel-server-rhscl-7-rpms
RUN sed -i s/mirror.centos.org/vault.centos.org/g /etc/yum.repos.d/*.repo
RUN sed -i s/^#.*baseurl=http/baseurl=http/g /etc/yum.repos.d/*.repo
RUN sed -i s/^mirrorlist=http/#mirrorlist=http/g /etc/yum.repos.d/*.repo
RUN yum install -y devtoolset-${DEVTOOLSET_VERSION}-gcc devtoolset-${DEVTOOLSET_VERSION}-gcc-c++ devtoolset-${DEVTOOLSET_VERSION}-gcc-gfortran devtoolset-${DEVTOOLSET_VERSION}-binutils
ENV PATH=/opt/rh/devtoolset-${DEVTOOLSET_VERSION}/root/usr/bin:$PATH
ENV LD_LIBRARY_PATH=/opt/rh/devtoolset-${DEVTOOLSET_VERSION}/root/usr/lib64:/opt/rh/devtoolset-${DEVTOOLSET_VERSION}/root/usr/lib:$LD_LIBRARY_PATH

# cmake is already installed inside the rocm base image, so remove if present
RUN rpm -e cmake || true
# cmake-3.18.4 from pip
RUN yum install -y python3-pip && \
    python3 -mpip install cmake==3.18.4 && \
    ln -s /usr/local/bin/cmake /usr/bin/cmake

# ninja
RUN yum install -y ninja-build

FROM cpu_final as cuda_final
RUN rm -rf /usr/local/cuda-${BASE_CUDA_VERSION}
COPY --from=cuda     /usr/local/cuda-${BASE_CUDA_VERSION}  /usr/local/cuda-${BASE_CUDA_VERSION}
COPY --from=magma    /usr/local/cuda-${BASE_CUDA_VERSION}  /usr/local/cuda-${BASE_CUDA_VERSION}
RUN ln -sf /usr/local/cuda-${BASE_CUDA_VERSION} /usr/local/cuda
ENV PATH=/usr/local/cuda/bin:$PATH

FROM cpu_final as rocm_final
ARG ROCM_VERSION=3.7
ARG PYTORCH_ROCM_ARCH
ENV PYTORCH_ROCM_ARCH ${PYTORCH_ROCM_ARCH}
# Adding ROCM_PATH env var so that LoadHip.cmake (even with logic updated for ROCm6.0)
# find HIP works for ROCm5.7. Not needed for ROCm6.0 and above.
# Remove below when ROCm5.7 is not in support matrix anymore.
ENV ROCM_PATH /opt/rocm
ENV MKLROOT /opt/intel
# No need to install ROCm as base docker image should have full ROCm install
#ADD ./common/install_rocm.sh install_rocm.sh
#RUN ROCM_VERSION=${ROCM_VERSION} bash ./install_rocm.sh && rm install_rocm.sh
ADD ./common/install_rocm_drm.sh install_rocm_drm.sh
RUN bash ./install_rocm_drm.sh && rm install_rocm_drm.sh
# cmake3 is needed for the MIOpen build
RUN ln -sf /usr/local/bin/cmake /usr/bin/cmake3
ADD ./common/install_rocm_magma.sh install_rocm_magma.sh
RUN bash ./install_rocm_magma.sh && rm install_rocm_magma.sh
ADD ./common/install_miopen.sh install_miopen.sh
RUN bash ./install_miopen.sh ${ROCM_VERSION} && rm install_miopen.sh

# Install AOTriton
COPY ./common/common_utils.sh common_utils.sh
COPY ./aotriton_version.txt aotriton_version.txt
COPY ./common/install_aotriton.sh install_aotriton.sh
RUN bash ./install_aotriton.sh /opt/rocm && rm install_aotriton.sh aotriton_version.txt
ENV AOTRITON_INSTALLED_PREFIX /opt/rocm/aotriton
