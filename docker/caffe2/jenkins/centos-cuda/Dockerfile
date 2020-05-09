ARG CENTOS_VERSION
ARG CUDA_VERSION
ARG CUDNN_VERSION
FROM nvidia/cuda:${CUDA_VERSION}-cudnn${CUDNN_VERSION}-devel-centos${CENTOS_VERSION}

# Install required packages to build Caffe2
ARG EC2
ADD ./install_base.sh install_base.sh
RUN bash ./install_base.sh && rm install_base.sh

# Install devtoolset
ARG DEVTOOLSET_VERSION
ADD ./install_devtoolset.sh install_devtoolset.sh
RUN bash ./install_devtoolset.sh
RUN rm install_devtoolset.sh
ENV BASH_ENV "/etc/profile"

# Compile/install ccache for faster builds
ADD ./install_ccache.sh install_ccache.sh
RUN bash ./install_ccache.sh && rm install_ccache.sh

# Install Python
ARG PYTHON_VERSION
ADD ./install_python.sh install_python.sh
RUN if [ -n "${PYTHON_VERSION}" ]; then bash ./install_python.sh; fi
RUN rm install_python.sh

# (optional) Add Jenkins user
ARG JENKINS
ARG JENKINS_UID
ARG JENKINS_GID
ADD ./add_jenkins_user.sh add_jenkins_user.sh
RUN if [ -n "${JENKINS}" ]; then bash ./add_jenkins_user.sh; fi
RUN rm add_jenkins_user.sh

# Include BUILD_ENVIRONMENT environment variable in image
ARG BUILD_ENVIRONMENT
ENV BUILD_ENVIRONMENT ${BUILD_ENVIRONMENT}
