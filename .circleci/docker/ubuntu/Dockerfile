ARG UBUNTU_VERSION

FROM ubuntu:${UBUNTU_VERSION}

ARG UBUNTU_VERSION

ENV DEBIAN_FRONTEND noninteractive

ARG CLANG_VERSION

# Install common dependencies (so that this step can be cached separately)
ARG EC2
COPY ./common/install_base.sh install_base.sh
RUN bash ./install_base.sh && rm install_base.sh

# Install clang
ARG LLVMDEV
COPY ./common/install_clang.sh install_clang.sh
RUN bash ./install_clang.sh && rm install_clang.sh

# (optional) Install thrift.
ARG THRIFT
COPY ./common/install_thrift.sh install_thrift.sh
RUN if [ -n "${THRIFT}" ]; then bash ./install_thrift.sh; fi
RUN rm install_thrift.sh
ENV INSTALLED_THRIFT ${THRIFT}

# Install user
COPY ./common/install_user.sh install_user.sh
RUN bash ./install_user.sh && rm install_user.sh

# Install katex
ARG KATEX
COPY ./common/install_docs_reqs.sh install_docs_reqs.sh
RUN bash ./install_docs_reqs.sh && rm install_docs_reqs.sh

# Install conda and other packages (e.g., numpy, pytest)
ENV PATH /opt/conda/bin:$PATH
ARG ANACONDA_PYTHON_VERSION
ARG CONDA_CMAKE
COPY requirements-ci.txt /opt/conda/requirements-ci.txt
COPY ./common/install_conda.sh install_conda.sh
RUN bash ./install_conda.sh && rm install_conda.sh
RUN rm /opt/conda/requirements-ci.txt

# Install gcc
ARG GCC_VERSION
COPY ./common/install_gcc.sh install_gcc.sh
RUN bash ./install_gcc.sh && rm install_gcc.sh

# Install lcov for C++ code coverage
COPY ./common/install_lcov.sh install_lcov.sh
RUN  bash ./install_lcov.sh && rm install_lcov.sh

# Install cuda and cudnn
ARG CUDA_VERSION
RUN wget -q https://raw.githubusercontent.com/pytorch/builder/main/common/install_cuda.sh -O install_cuda.sh
RUN bash ./install_cuda.sh ${CUDA_VERSION} && rm install_cuda.sh
ENV DESIRED_CUDA ${CUDA_VERSION}
ENV PATH /usr/local/nvidia/bin:/usr/local/cuda/bin:$PATH

# (optional) Install UCC
ARG UCX_COMMIT
ARG UCC_COMMIT
ENV UCX_COMMIT $UCX_COMMIT
ENV UCC_COMMIT $UCC_COMMIT
ENV UCX_HOME /usr
ENV UCC_HOME /usr
ADD ./common/install_ucc.sh install_ucc.sh
RUN if [ -n "${UCX_COMMIT}" ] && [ -n "${UCC_COMMIT}" ]; then bash ./install_ucc.sh; fi
RUN rm install_ucc.sh

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

# (optional) Install vision packages like OpenCV and ffmpeg
ARG VISION
COPY ./common/install_vision.sh install_vision.sh
RUN if [ -n "${VISION}" ]; then bash ./install_vision.sh; fi
RUN rm install_vision.sh
ENV INSTALLED_VISION ${VISION}

# (optional) Install Android NDK
ARG ANDROID
ARG ANDROID_NDK
ARG GRADLE_VERSION
COPY ./common/install_android.sh install_android.sh
COPY ./android/AndroidManifest.xml AndroidManifest.xml
COPY ./android/build.gradle build.gradle
RUN if [ -n "${ANDROID}" ]; then bash ./install_android.sh; fi
RUN rm install_android.sh
RUN rm AndroidManifest.xml
RUN rm build.gradle
ENV INSTALLED_ANDROID ${ANDROID}

# (optional) Install Vulkan SDK
ARG VULKAN_SDK_VERSION
COPY ./common/install_vulkan_sdk.sh install_vulkan_sdk.sh
RUN if [ -n "${VULKAN_SDK_VERSION}" ]; then bash ./install_vulkan_sdk.sh; fi
RUN rm install_vulkan_sdk.sh

# (optional) Install swiftshader
ARG SWIFTSHADER
COPY ./common/install_swiftshader.sh install_swiftshader.sh
RUN if [ -n "${SWIFTSHADER}" ]; then bash ./install_swiftshader.sh; fi
RUN rm install_swiftshader.sh

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

COPY ./common/install_openssl.sh install_openssl.sh
RUN bash ./install_openssl.sh
ENV OPENSSL_ROOT_DIR /opt/openssl
ENV OPENSSL_DIR /opt/openssl
RUN rm install_openssl.sh

# Install ccache/sccache (do this last, so we get priority in PATH)
COPY ./common/install_cache.sh install_cache.sh
ENV PATH /opt/cache/bin:$PATH
# See https://github.com/pytorch/pytorch/issues/82174
# TODO(sdym@fb.com):
# check if this is needed after full off Xenial migration
ENV CARGO_NET_GIT_FETCH_WITH_CLI true
RUN bash ./install_cache.sh && rm install_cache.sh

# Add jni.h for java host build
COPY ./common/install_jni.sh install_jni.sh
COPY ./java/jni.h jni.h
RUN bash ./install_jni.sh && rm install_jni.sh

# Install Open MPI for CUDA
COPY ./common/install_openmpi.sh install_openmpi.sh
RUN if [ -n "${CUDA_VERSION}" ]; then bash install_openmpi.sh; fi
RUN rm install_openmpi.sh

# Include BUILD_ENVIRONMENT environment variable in image
ARG BUILD_ENVIRONMENT
ENV BUILD_ENVIRONMENT ${BUILD_ENVIRONMENT}

# Install LLVM dev version (Defined in the pytorch/builder github repository)
COPY --from=pytorch/llvm:9.0.1 /opt/llvm /opt/llvm

# AWS specific CUDA build guidance
ENV TORCH_CUDA_ARCH_LIST Maxwell
ENV TORCH_NVCC_FLAGS "-Xfatbin -compress-all"
ENV CUDA_PATH /usr/local/cuda

USER jenkins
CMD ["bash"]
