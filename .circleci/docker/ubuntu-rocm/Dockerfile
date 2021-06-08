ARG UBUNTU_VERSION

FROM ubuntu:${UBUNTU_VERSION}

ARG UBUNTU_VERSION

ENV DEBIAN_FRONTEND noninteractive

# Install common dependencies (so that this step can be cached separately)
ARG EC2
ADD ./common/install_base.sh install_base.sh
RUN bash ./install_base.sh && rm install_base.sh

# Install clang
ARG LLVMDEV
ARG CLANG_VERSION
ADD ./common/install_clang.sh install_clang.sh
RUN bash ./install_clang.sh && rm install_clang.sh

# Install user
ADD ./common/install_user.sh install_user.sh
RUN bash ./install_user.sh && rm install_user.sh

# Install conda and other packages (e.g., numpy, coverage, pytest)
ENV PATH /opt/conda/bin:$PATH
ARG ANACONDA_PYTHON_VERSION
ADD ./common/install_conda.sh install_conda.sh
RUN bash ./install_conda.sh && rm install_conda.sh

# Install gcc
ARG GCC_VERSION
ADD ./common/install_gcc.sh install_gcc.sh
RUN bash ./install_gcc.sh && rm install_gcc.sh

# (optional) Install protobuf for ONNX
ARG PROTOBUF
ADD ./common/install_protobuf.sh install_protobuf.sh
RUN if [ -n "${PROTOBUF}" ]; then bash ./install_protobuf.sh; fi
RUN rm install_protobuf.sh
ENV INSTALLED_PROTOBUF ${PROTOBUF}

# (optional) Install database packages like LMDB and LevelDB
ARG DB
ADD ./common/install_db.sh install_db.sh
RUN if [ -n "${DB}" ]; then bash ./install_db.sh; fi
RUN rm install_db.sh
ENV INSTALLED_DB ${DB}

# (optional) Install vision packages like OpenCV and ffmpeg
ARG VISION
ADD ./common/install_vision.sh install_vision.sh
RUN if [ -n "${VISION}" ]; then bash ./install_vision.sh; fi
RUN rm install_vision.sh
ENV INSTALLED_VISION ${VISION}

# Install rocm
ARG ROCM_VERSION
ADD ./common/install_rocm.sh install_rocm.sh
RUN bash ./install_rocm.sh
RUN rm install_rocm.sh
ENV PATH /opt/rocm/bin:$PATH
ENV PATH /opt/rocm/hcc/bin:$PATH
ENV PATH /opt/rocm/hip/bin:$PATH
ENV PATH /opt/rocm/opencl/bin:$PATH
ENV PATH /opt/rocm/llvm/bin:$PATH
ENV MAGMA_HOME /opt/rocm/magma
ENV LANG C.UTF-8
ENV LC_ALL C.UTF-8

# (optional) Install non-default CMake version
ARG CMAKE_VERSION
ADD ./common/install_cmake.sh install_cmake.sh
RUN if [ -n "${CMAKE_VERSION}" ]; then bash ./install_cmake.sh; fi
RUN rm install_cmake.sh

# (optional) Install non-default Ninja version
ARG NINJA_VERSION
ADD ./common/install_ninja.sh install_ninja.sh
RUN if [ -n "${NINJA_VERSION}" ]; then bash ./install_ninja.sh; fi
RUN rm install_ninja.sh

# Install ccache/sccache (do this last, so we get priority in PATH)
ADD ./common/install_cache.sh install_cache.sh
ENV PATH /opt/cache/bin:$PATH
RUN bash ./install_cache.sh && rm install_cache.sh

# Include BUILD_ENVIRONMENT environment variable in image
ARG BUILD_ENVIRONMENT
ENV BUILD_ENVIRONMENT ${BUILD_ENVIRONMENT}

USER jenkins
CMD ["bash"]
