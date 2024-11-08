ARG UBUNTU_VERSION

FROM ubuntu:${UBUNTU_VERSION}

ARG UBUNTU_VERSION

ENV DEBIAN_FRONTEND noninteractive

ARG CLANG_VERSION

# Install common dependencies (so that this step can be cached separately)
COPY ./common/install_base.sh install_base.sh
RUN bash ./install_base.sh && rm install_base.sh

# Install clang
ARG LLVMDEV
COPY ./common/install_clang.sh install_clang.sh
RUN bash ./install_clang.sh && rm install_clang.sh

# Install user
COPY ./common/install_user.sh install_user.sh
RUN bash ./install_user.sh && rm install_user.sh

# Install katex
ARG KATEX
COPY ./common/install_docs_reqs.sh install_docs_reqs.sh
RUN bash ./install_docs_reqs.sh && rm install_docs_reqs.sh

# Install conda and other packages (e.g., numpy, pytest)
ARG ANACONDA_PYTHON_VERSION
ARG CONDA_CMAKE
ARG DOCS
ENV ANACONDA_PYTHON_VERSION=$ANACONDA_PYTHON_VERSION
ENV PATH /opt/conda/envs/py_$ANACONDA_PYTHON_VERSION/bin:/opt/conda/bin:$PATH
ENV DOCS=$DOCS
COPY requirements-ci.txt requirements-docs.txt /opt/conda/
COPY ./common/install_conda.sh install_conda.sh
COPY ./common/common_utils.sh common_utils.sh
RUN bash ./install_conda.sh && rm install_conda.sh common_utils.sh /opt/conda/requirements-ci.txt /opt/conda/requirements-docs.txt
RUN if [ -n "${UNINSTALL_DILL}" ]; then pip uninstall -y dill; fi

# Install gcc
ARG GCC_VERSION
COPY ./common/install_gcc.sh install_gcc.sh
RUN bash ./install_gcc.sh && rm install_gcc.sh

# Install lcov for C++ code coverage
COPY ./common/install_lcov.sh install_lcov.sh
RUN  bash ./install_lcov.sh && rm install_lcov.sh

# Install cuda and cudnn
ARG CUDA_VERSION
COPY ./common/install_cuda.sh install_cuda.sh
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

# (optional) Install vision packages like OpenCV
ARG VISION
COPY ./common/install_vision.sh ./common/cache_vision_models.sh ./common/common_utils.sh ./
RUN if [ -n "${VISION}" ]; then bash ./install_vision.sh; fi
RUN rm install_vision.sh cache_vision_models.sh common_utils.sh
ENV INSTALLED_VISION ${VISION}

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

ARG INDUCTOR_BENCHMARKS
COPY ./common/install_inductor_benchmark_deps.sh install_inductor_benchmark_deps.sh
COPY ./common/common_utils.sh common_utils.sh
COPY ci_commit_pins/huggingface.txt huggingface.txt
COPY ci_commit_pins/timm.txt timm.txt
RUN if [ -n "${INDUCTOR_BENCHMARKS}" ]; then bash ./install_inductor_benchmark_deps.sh; fi
RUN rm install_inductor_benchmark_deps.sh common_utils.sh timm.txt huggingface.txt

ARG TRITON
# Install triton, this needs to be done before sccache because the latter will
# try to reach out to S3, which docker build runners don't have access
COPY ./common/install_triton.sh install_triton.sh
COPY ./common/common_utils.sh common_utils.sh
COPY ci_commit_pins/triton.txt triton.txt
RUN if [ -n "${TRITON}" ]; then bash ./install_triton.sh; fi
RUN rm install_triton.sh common_utils.sh triton.txt

ARG TRITON_CPU
COPY ./common/install_triton.sh install_triton.sh
COPY ./common/common_utils.sh common_utils.sh
COPY ci_commit_pins/triton-cpu.txt triton-cpu.txt
RUN if [ -n "${TRITON_CPU}" ]; then bash ./install_triton.sh; fi
RUN rm install_triton.sh common_utils.sh triton-cpu.txt

ARG EXECUTORCH
# Build and install executorch
COPY ./common/install_executorch.sh install_executorch.sh
COPY ./common/common_utils.sh common_utils.sh
COPY ci_commit_pins/executorch.txt executorch.txt
RUN if [ -n "${EXECUTORCH}" ]; then bash ./install_executorch.sh; fi
RUN rm install_executorch.sh common_utils.sh executorch.txt

ARG HALIDE
# Build and install halide
COPY ./common/install_halide.sh install_halide.sh
COPY ./common/common_utils.sh common_utils.sh
COPY ci_commit_pins/halide.txt halide.txt
RUN if [ -n "${HALIDE}" ]; then bash ./install_halide.sh; fi
RUN rm install_halide.sh common_utils.sh halide.txt

ARG ONNX
# Install ONNX dependencies
COPY ./common/install_onnx.sh ./common/common_utils.sh ./
RUN if [ -n "${ONNX}" ]; then bash ./install_onnx.sh; fi
RUN rm install_onnx.sh common_utils.sh

# (optional) Build ACL
ARG ACL
COPY ./common/install_acl.sh install_acl.sh
RUN if [ -n "${ACL}" ]; then bash ./install_acl.sh; fi
RUN rm install_acl.sh
ENV INSTALLED_ACL ${ACL}

# Install ccache/sccache (do this last, so we get priority in PATH)
ARG SKIP_SCCACHE_INSTALL
COPY ./common/install_cache.sh install_cache.sh
ENV PATH /opt/cache/bin:$PATH
RUN if [ -z "${SKIP_SCCACHE_INSTALL}" ]; then bash ./install_cache.sh; fi
RUN rm install_cache.sh

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
ARG SKIP_LLVM_SRC_BUILD_INSTALL
COPY --from=pytorch/llvm:9.0.1 /opt/llvm /opt/llvm
RUN if [ -n "${SKIP_LLVM_SRC_BUILD_INSTALL}" ]; then set -eu; rm -rf /opt/llvm; fi

# AWS specific CUDA build guidance
ENV TORCH_CUDA_ARCH_LIST Maxwell
ENV TORCH_NVCC_FLAGS "-Xfatbin -compress-all"
ENV CUDA_PATH /usr/local/cuda

USER jenkins
CMD ["bash"]
