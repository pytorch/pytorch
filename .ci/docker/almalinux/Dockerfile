ARG CUDA_VERSION=12.4
ARG BASE_TARGET=cuda${CUDA_VERSION}
FROM amd64/almalinux:8 as base

ENV LC_ALL en_US.UTF-8
ENV LANG en_US.UTF-8
ENV LANGUAGE en_US.UTF-8

ARG DEVTOOLSET_VERSION=11

ENV LC_ALL en_US.UTF-8
ENV LANG en_US.UTF-8
ENV LANGUAGE en_US.UTF-8

RUN yum -y update
RUN yum -y install epel-release
RUN yum install -y sudo wget curl perl util-linux xz bzip2 git patch which perl zlib-devel openssl-devel yum-utils autoconf automake make gcc-toolset-${DEVTOOLSET_VERSION}-toolchain
# Just add everything as a safe.directory for git since these will be used in multiple places with git
RUN git config --global --add safe.directory '*'
ENV PATH=/opt/rh/gcc-toolset-${DEVTOOLSET_VERSION}/root/usr/bin:$PATH

# cmake-3.18.4 from pip
RUN yum install -y python3-pip && \
    python3 -mpip install cmake==3.18.4 && \
    ln -s /usr/local/bin/cmake /usr/bin/cmake3
RUN rm -rf /usr/local/cuda-*

FROM base as openssl
ADD ./common/install_openssl.sh install_openssl.sh
RUN bash ./install_openssl.sh && rm install_openssl.sh

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
ARG CUDA_VERSION=12.4
RUN rm -rf /usr/local/cuda-*
ADD ./common/install_cuda.sh install_cuda.sh
ENV CUDA_HOME=/usr/local/cuda-${CUDA_VERSION}
# Preserve CUDA_VERSION for the builds
ENV CUDA_VERSION=${CUDA_VERSION}
# Make things in our path by default
ENV PATH=/usr/local/cuda-${CUDA_VERSION}/bin:$PATH

FROM cuda as cuda11.8
RUN bash ./install_cuda.sh 11.8
ENV DESIRED_CUDA=11.8

FROM cuda as cuda12.1
RUN bash ./install_cuda.sh 12.1
ENV DESIRED_CUDA=12.1

FROM cuda as cuda12.4
RUN bash ./install_cuda.sh 12.4
ENV DESIRED_CUDA=12.4

FROM cuda as cuda12.6
RUN bash ./install_cuda.sh 12.6
ENV DESIRED_CUDA=12.6

# Install MNIST test data
FROM base as mnist
ADD ./common/install_mnist.sh install_mnist.sh
RUN bash ./install_mnist.sh

FROM base as all_cuda
COPY --from=cuda11.8  /usr/local/cuda-11.8 /usr/local/cuda-11.8
COPY --from=cuda12.1  /usr/local/cuda-12.1 /usr/local/cuda-12.1
COPY --from=cuda12.4  /usr/local/cuda-12.4 /usr/local/cuda-12.4
COPY --from=cuda12.6  /usr/local/cuda-12.6 /usr/local/cuda-12.6

# Final step
FROM ${BASE_TARGET} as final
COPY --from=openssl            /opt/openssl           /opt/openssl
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
