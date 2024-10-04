ARG CENTOS_VERSION

FROM centos:${CENTOS_VERSION}

ARG CENTOS_VERSION

# Set AMD gpu targets to build for
ARG PYTORCH_ROCM_ARCH
ENV PYTORCH_ROCM_ARCH ${PYTORCH_ROCM_ARCH}

# Install required packages to build Caffe2

# Install common dependencies (so that this step can be cached separately)
COPY ./common/install_base.sh install_base.sh
RUN bash ./install_base.sh && rm install_base.sh

# Update CentOS git version
RUN yum -y remove git
RUN yum -y remove git-*
RUN yum -y install https://packages.endpoint.com/rhel/7/os/x86_64/endpoint-repo-1.9-1.x86_64.rpm || \
    (yum -y install https://packages.endpointdev.com/rhel/7/os/x86_64/endpoint-repo-1.9-1.x86_64.rpm && \
    sed -i "s/packages.endpoint/packages.endpointdev/" /etc/yum.repos.d/endpoint.repo)
RUN yum install -y git

# Install devtoolset
ARG DEVTOOLSET_VERSION
COPY ./common/install_devtoolset.sh install_devtoolset.sh
RUN bash ./install_devtoolset.sh && rm install_devtoolset.sh
ENV BASH_ENV "/etc/profile"

# (optional) Install non-default glibc version
ARG GLIBC_VERSION
COPY ./common/install_glibc.sh install_glibc.sh
RUN if [ -n "${GLIBC_VERSION}" ]; then bash ./install_glibc.sh; fi
RUN rm install_glibc.sh

# Install user
COPY ./common/install_user.sh install_user.sh
RUN bash ./install_user.sh && rm install_user.sh

# Install conda and other packages (e.g., numpy, pytest)
ARG ANACONDA_PYTHON_VERSION
ARG CONDA_CMAKE
ENV ANACONDA_PYTHON_VERSION=$ANACONDA_PYTHON_VERSION
ENV PATH /opt/conda/envs/py_$ANACONDA_PYTHON_VERSION/bin:/opt/conda/bin:$PATH
COPY requirements-ci.txt /opt/conda/requirements-ci.txt
COPY ./common/install_conda.sh install_conda.sh
COPY ./common/common_utils.sh common_utils.sh
RUN bash ./install_conda.sh && rm install_conda.sh common_utils.sh /opt/conda/requirements-ci.txt

# (optional) Install protobuf for ONNX
ARG PROTOBUF
COPY ./common/install_protobuf.sh install_protobuf.sh
RUN if [ -n "${PROTOBUF}" ]; then bash ./install_protobuf.sh; fi
RUN rm install_protobuf.sh
ENV INSTALLED_PROTOBUF ${PROTOBUF}

# (optional) Install database packages like LMDB and LevelDB
ARG DB
COPY ./common/install_db.sh install_db.sh
RUN if [ -n "${DB}" ]; then bash ./install_db.sh; fi
RUN rm install_db.sh
ENV INSTALLED_DB ${DB}

# (optional) Install vision packages like OpenCV
ARG VISION
COPY ./common/install_vision.sh ./common/cache_vision_models.sh ./common/common_utils.sh ./
RUN if [ -n "${VISION}" ]; then bash ./install_vision.sh; fi
RUN rm install_vision.sh cache_vision_models.sh common_utils.sh
ENV INSTALLED_VISION ${VISION}

# Install rocm
ARG ROCM_VERSION
COPY ./common/install_rocm.sh install_rocm.sh
RUN bash ./install_rocm.sh
RUN rm install_rocm.sh
COPY ./common/install_rocm_magma.sh install_rocm_magma.sh
RUN bash ./install_rocm_magma.sh
RUN rm install_rocm_magma.sh
COPY ./common/install_amdsmi.sh install_amdsmi.sh
RUN bash ./install_amdsmi.sh
RUN rm install_amdsmi.sh
ENV PATH /opt/rocm/bin:$PATH
ENV PATH /opt/rocm/hcc/bin:$PATH
ENV PATH /opt/rocm/hip/bin:$PATH
ENV PATH /opt/rocm/opencl/bin:$PATH
ENV PATH /opt/rocm/llvm/bin:$PATH
ENV MAGMA_HOME /opt/rocm/magma
ENV LANG en_US.utf8
ENV LC_ALL en_US.utf8

# (optional) Install non-default CMake version
ARG CMAKE_VERSION
COPY ./common/install_cmake.sh install_cmake.sh
RUN if [ -n "${CMAKE_VERSION}" ]; then bash ./install_cmake.sh; fi
RUN rm install_cmake.sh

# (optional) Install non-default Ninja version
ARG NINJA_VERSION
COPY ./common/install_ninja.sh install_ninja.sh
RUN if [ -n "${NINJA_VERSION}" ]; then bash ./install_ninja.sh; fi
RUN rm install_ninja.sh

ARG TRITON
# Install triton, this needs to be done before sccache because the latter will
# try to reach out to S3, which docker build runners don't have access
ENV CMAKE_C_COMPILER cc
ENV CMAKE_CXX_COMPILER c++
COPY ./common/install_triton.sh install_triton.sh
COPY ./common/common_utils.sh common_utils.sh
COPY ci_commit_pins/triton.txt triton.txt
COPY triton_version.txt triton_version.txt
RUN if [ -n "${TRITON}" ]; then bash ./install_triton.sh; fi
RUN rm install_triton.sh common_utils.sh triton.txt triton_version.txt

# Install AOTriton (Early fail)
COPY ./aotriton_version.txt aotriton_version.txt
COPY ./common/common_utils.sh common_utils.sh
COPY ./common/install_aotriton.sh install_aotriton.sh
RUN ["/bin/bash", "-c", "./install_aotriton.sh /opt/rocm && rm -rf install_aotriton.sh aotriton_version.txt common_utils.sh"]
ENV AOTRITON_INSTALLED_PREFIX /opt/rocm/aotriton

# Install ccache/sccache (do this last, so we get priority in PATH)
COPY ./common/install_cache.sh install_cache.sh
ENV PATH /opt/cache/bin:$PATH
RUN bash ./install_cache.sh && rm install_cache.sh

# Include BUILD_ENVIRONMENT environment variable in image
ARG BUILD_ENVIRONMENT
ENV BUILD_ENVIRONMENT ${BUILD_ENVIRONMENT}

USER jenkins
CMD ["bash"]
