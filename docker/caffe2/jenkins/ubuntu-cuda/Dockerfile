ARG UBUNTU_VERSION
ARG CUDA_VERSION
ARG CUDNN_VERSION
FROM nvidia/cuda:${CUDA_VERSION}-cudnn${CUDNN_VERSION}-devel-ubuntu${UBUNTU_VERSION}

# Install required packages to build Caffe2
ARG EC2
ARG UBUNTU_VERSION
ADD ./install_base.sh install_base.sh
RUN bash ./install_base.sh && rm install_base.sh

# Compile/install ccache for faster builds
ADD ./install_ccache.sh install_ccache.sh
RUN bash ./install_ccache.sh && rm install_ccache.sh

# (optional) Install non-default GCC version
ARG GCC_VERSION
ADD ./install_gcc.sh install_gcc.sh
RUN if [ -n "${GCC_VERSION}" ]; then bash ./install_gcc.sh; fi
RUN rm install_gcc.sh

# Install NCCL for all CUDA builds
ARG UBUNTU_VERSION
ARG CUDA_VERSION
ADD ./install_nccl.sh install_nccl.sh
RUN bash ./install_nccl.sh && rm install_nccl.sh

# Install Python
ARG PYTHON_VERSION
ADD ./install_python.sh install_python.sh
RUN if [ -n "${PYTHON_VERSION}" ]; then bash ./install_python.sh; fi
RUN rm install_python.sh

# Install Anaconda
ARG ANACONDA_VERSION
ADD ./install_anaconda.sh install_anaconda.sh
RUN if [ -n "${ANACONDA_VERSION}" ]; then bash ./install_anaconda.sh; fi
RUN rm install_anaconda.sh

# (optional) Add Jenkins user
ARG JENKINS
ARG JENKINS_UID
ARG JENKINS_GID
ADD ./add_jenkins_user.sh add_jenkins_user.sh
RUN if [ -n "${JENKINS}" ]; then bash ./add_jenkins_user.sh ${JENKINS_UID} ${JENKINS_GID}; fi
RUN rm add_jenkins_user.sh

# Include BUILD_ENVIRONMENT environment variable in image
ARG BUILD_ENVIRONMENT
ENV BUILD_ENVIRONMENT ${BUILD_ENVIRONMENT}
