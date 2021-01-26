ARG MANYLINUX_VERSION=2014
FROM quay.io/pypa/manylinux${MANYLINUX_VERSION}_x86_64:latest as base
RUN yum install -y wget curl perl util-linux xz bzip2 git patch which perl
RUN yum install -y yum-utils centos-release-scl
RUN yum-config-manager --enable rhel-server-rhscl-7-rpms
RUN yum install -y devtoolset-7-gcc devtoolset-7-gcc-c++ devtoolset-7-gcc-gfortran devtoolset-7-binutils
ENV PATH=/opt/rh/devtoolset-7/root/usr/bin:$PATH
ENV LD_LIBRARY_PATH=/opt/rh/devtoolset-7/root/usr/lib64:/opt/rh/devtoolset-7/root/usr/lib:$LD_LIBRARY_PATH

# RUN yum install -y http://dl.fedoraproject.org/pub/epel/epel-release-latest-7.noarch.rpm

# cmake
RUN yum install -y cmake3 && \
    ln -s /usr/bin/cmake3 /usr/bin/cmake

RUN yum install -y autoconf automake make

FROM base as intel
# MKL
ADD ./docker-common/install_mkl.sh install_mkl.sh
RUN bash ./install_mkl.sh && rm install_mkl.sh

# EPEL for cmake
FROM base as patchelf
# Install patchelf
ADD ./docker-common/install_patchelf.sh install_patchelf.sh
RUN bash ./install_patchelf.sh && cp $(which patchelf) /patchelf

FROM base as magma
ARG TARGET_VERSION=10.1
# Install magma
ADD ./docker-common/install_magma.sh install_magma.sh
RUN bash ./install_magma.sh ${TARGET_VERSION} && rm install_magma.sh

FROM base as cuda-dev
ARG TARGET_VERSION=10.1
ADD ../docker-common/install_cuda.sh install_cuda.sh
RUN bash ./install_cuda.sh ${TARGET_VERSION}

FROM base as cpu
# For compat with current builder scripts, remove when scripts no longer need this
ARG DESIRED_PYTHON=3.6mu
ENV DESIRED_PYTHON=${DESIRED_PYTHON}
ENV DESIRED_CUDA=cpu
ADD                     ./docker-common/jni.h             /usr/local/include/jni.h
COPY --from=intel       /opt/intel                        /opt/intel
COPY --from=patchelf    /usr/local/bin/patchelf           /usr/local/bin/patchelf

FROM cpu as cuda
ARG TARGET_VERSION=10.1
# For compat with current builder scripts, remove when scripts no longer need this
ARG DESIRED_CUDA=cu101
ENV DESIRED_CUDA=${DESIRED_CUDA}
COPY --from=cuda-dev /usr/local/cuda-${TARGET_VERSION} /usr/local/cuda-${TARGET_VERSION}
COPY --from=magma    /usr/local/cuda-${TARGET_VERSION} /usr/local/cuda-${TARGET_VERSION}

FROM cpu as rocm
ARG TARGET_VERSION=3.7
ENV DESIRED_CUDA=rocm${TARGET_VERSION}
RUN yum install -y \
        epel-release \
        kmod \
        openblas-devel \
        dkms
RUN echo -e "[ROCm]\nname=ROCm\nbaseurl=http://repo.radeon.com/rocm/yum/${ROCM_VERSION}\nenabled=1\ngpgcheck=0" >> /etc/yum.repos.d/rocm.repo
RUN yum install -y \
        rocm-dev \
        rocm-utils \
        rocfft \
        miopen-hip \
        rocblas \
        hipsparse \
        rocrand \
        rccl \
        hipcub \
        rocthrust \
        rocprofiler-dev \
        roctracer-dev
