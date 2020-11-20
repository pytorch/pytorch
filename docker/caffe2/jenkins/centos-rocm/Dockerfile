ARG CENTOS_VERSION
FROM centos:${CENTOS_VERSION}

# Include BUILD_ENVIRONMENT environment variable in image
ARG BUILD_ENVIRONMENT
ENV BUILD_ENVIRONMENT ${BUILD_ENVIRONMENT}

# Install required packages to build Caffe2
ARG EC2
ARG CENTOS_VERSION
ADD ./install_base.sh install_base.sh
RUN bash ./install_base.sh && rm install_base.sh

# Install devtoolset
ARG DEVTOOLSET_VERSION
ADD ./install_devtoolset.sh install_devtoolset.sh
RUN bash ./install_devtoolset.sh
RUN rm install_devtoolset.sh
ENV BASH_ENV "/etc/profile"

# (optional) Install non-default glibc version
ARG GLIBC_VERSION
ADD ./install_glibc.sh install_glibc.sh
RUN if [ -n "${GLIBC_VERSION}" ]; then bash ./install_glibc.sh; fi
RUN rm install_glibc.sh

# Install rocm
ARG ROCM_VERSION
ADD ./install_rocm.sh install_rocm.sh
RUN bash ./install_rocm.sh
RUN rm install_rocm.sh
ENV PATH /opt/rocm/bin:$PATH
ENV PATH /opt/rocm/hcc/bin:$PATH
ENV PATH /opt/rocm/hip/bin:$PATH
ENV PATH /opt/rocm/opencl/bin:$PATH
ENV LC_ALL en_US.utf8
ENV LANG en_US.utf8

# Install non-default CMake version
ARG CMAKE_VERSION
ADD ./install_cmake.sh install_cmake.sh
RUN if [ -n "${CMAKE_VERSION}" ]; then bash ./install_cmake.sh; fi
RUN rm install_cmake.sh

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
