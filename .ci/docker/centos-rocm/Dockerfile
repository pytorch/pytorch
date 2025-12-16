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
RUN yum -y install https://packages.endpointdev.com/rhel/7/os/x86_64/endpoint-repo-1.9-1.x86_64.rpm && \
    sed -i 's/packages.endpoint/packages.endpointdev/' /etc/yum.repos.d/endpoint.repo
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
ARG BUILD_ENVIRONMENT
ENV ANACONDA_PYTHON_VERSION=$ANACONDA_PYTHON_VERSION
ENV PATH /opt/conda/envs/py_$ANACONDA_PYTHON_VERSION/bin:/opt/conda/bin:$PATH
COPY requirements-ci.txt /opt/conda/requirements-ci.txt
COPY ./common/install_conda.sh install_conda.sh
COPY ./common/common_utils.sh common_utils.sh
RUN bash ./install_conda.sh && rm install_conda.sh common_utils.sh /opt/conda/requirements-ci.txt

# (optional) Install vision packages like OpenCV
ARG VISION
COPY ./common/install_vision.sh ./common/cache_vision_models.sh ./common/common_utils.sh ./
RUN if [ -n "${VISION}" ]; then bash ./install_vision.sh; fi
RUN rm install_vision.sh cache_vision_models.sh common_utils.sh
ENV INSTALLED_VISION ${VISION}

# Install rocm
ARG ROCM_VERSION
RUN mkdir ci_commit_pins
COPY ./common/common_utils.sh common_utils.sh
COPY ./ci_commit_pins/rocm-composable-kernel.txt ci_commit_pins/rocm-composable-kernel.txt
COPY ./common/install_rocm.sh install_rocm.sh
RUN bash ./install_rocm.sh
RUN rm install_rocm.sh common_utils.sh
RUN rm -r ci_commit_pins
COPY ./common/install_rocm_magma.sh install_rocm_magma.sh
RUN bash ./install_rocm_magma.sh ${ROCM_VERSION}
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

# Install ccache/sccache (do this last, so we get priority in PATH)
COPY ./common/install_cache.sh install_cache.sh
ENV PATH /opt/cache/bin:$PATH
RUN bash ./install_cache.sh && rm install_cache.sh

WORKDIR /app
RUN apt-get update && apt-get install -y git python3 python3-pip python3-setuptools
RUN git clone https://github.com/ROCm/rocm-libraries.git
RUN git checkout users/asunderwood/origami-torch-ext
RUN pip install nanobind
WORKDIR /app/rocm-libraries/shared/origami
RUN pip install python/

# Include BUILD_ENVIRONMENT environment variable in image
ARG BUILD_ENVIRONMENT
ENV BUILD_ENVIRONMENT ${BUILD_ENVIRONMENT}

USER jenkins
CMD ["bash"]
