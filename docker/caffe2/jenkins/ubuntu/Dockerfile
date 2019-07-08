ARG UBUNTU_VERSION
FROM ubuntu:${UBUNTU_VERSION}

# Install required packages to build Caffe2
ARG EC2
ARG UBUNTU_VERSION
ADD ./install_base.sh install_base.sh
RUN bash ./install_base.sh && rm install_base.sh

# Compile/install ccache for faster builds
ADD ./install_ccache.sh install_ccache.sh
RUN bash ./install_ccache.sh && rm install_ccache.sh

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

# (optional) Install Intel MKL
ARG MKL
ADD ./install_mkl.sh install_mkl.sh
RUN if [ -n "${MKL}" ]; then bash ./install_mkl.sh; fi
RUN rm install_mkl.sh

# (optional) Install Android NDK
ARG ANDROID
ARG ANDROID_NDK
ADD ./install_android.sh install_android.sh
RUN if [ -n "${ANDROID}" ]; then bash ./install_android.sh; fi
RUN rm install_android.sh

# (optional) Install non-default GCC version
ARG GCC_VERSION
ADD ./install_gcc.sh install_gcc.sh
RUN if [ -n "${GCC_VERSION}" ]; then bash ./install_gcc.sh; fi
RUN rm install_gcc.sh

# (optional) Install non-default clang version
ARG CLANG_VERSION
ADD ./install_clang.sh install_clang.sh
RUN if [ -n "${CLANG_VERSION}" ]; then bash ./install_clang.sh; fi
RUN rm install_clang.sh

# (optional) Install non-default CMake version
ARG CMAKE_VERSION
ADD ./install_cmake.sh install_cmake.sh
RUN if [ -n "${CMAKE_VERSION}" ]; then bash ./install_cmake.sh; fi
RUN rm install_cmake.sh

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
