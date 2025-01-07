ARG UBUNTU_VERSION
ARG CUDA_VERSION
ARG IMAGE_NAME

FROM ${IMAGE_NAME}

ARG UBUNTU_VERSION
ARG CUDA_VERSION

ENV DEBIAN_FRONTEND noninteractive

# Install common dependencies (so that this step can be cached separately)
COPY ./common/install_base.sh install_base.sh
RUN bash ./install_base.sh && rm install_base.sh

# Install user
COPY ./common/install_user.sh install_user.sh
RUN bash ./install_user.sh && rm install_user.sh

# Install katex
ARG KATEX
COPY ./common/install_docs_reqs.sh install_docs_reqs.sh
RUN bash ./install_docs_reqs.sh && rm install_docs_reqs.sh

# Install conda and other packages (e.g., numpy, pytest)
ARG ANACONDA_PYTHON_VERSION
ENV ANACONDA_PYTHON_VERSION=$ANACONDA_PYTHON_VERSION
ENV PATH /opt/conda/envs/py_$ANACONDA_PYTHON_VERSION/bin:/opt/conda/bin:$PATH
ARG CONDA_CMAKE
COPY requirements-ci.txt /opt/conda/requirements-ci.txt
COPY ./common/install_conda.sh install_conda.sh
COPY ./common/common_utils.sh common_utils.sh
COPY ./common/install_magma_conda.sh install_magma_conda.sh
RUN bash ./install_conda.sh && rm install_conda.sh install_magma_conda.sh common_utils.sh /opt/conda/requirements-ci.txt

# Install gcc
ARG GCC_VERSION
COPY ./common/install_gcc.sh install_gcc.sh
RUN bash ./install_gcc.sh && rm install_gcc.sh

# Install clang
ARG CLANG_VERSION
COPY ./common/install_clang.sh install_clang.sh
RUN bash ./install_clang.sh && rm install_clang.sh

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

COPY ./common/install_openssl.sh install_openssl.sh
ENV OPENSSL_ROOT_DIR /opt/openssl
RUN bash ./install_openssl.sh
ENV OPENSSL_DIR /opt/openssl

ARG INDUCTOR_BENCHMARKS
ARG ANACONDA_PYTHON_VERSION
ENV ANACONDA_PYTHON_VERSION=$ANACONDA_PYTHON_VERSION
COPY ./common/install_inductor_benchmark_deps.sh install_inductor_benchmark_deps.sh
COPY ./common/common_utils.sh common_utils.sh
COPY ci_commit_pins/huggingface.txt huggingface.txt
COPY ci_commit_pins/timm.txt timm.txt
RUN if [ -n "${INDUCTOR_BENCHMARKS}" ]; then bash ./install_inductor_benchmark_deps.sh; fi
RUN rm install_inductor_benchmark_deps.sh common_utils.sh timm.txt huggingface.txt

# (optional) Install non-default CMake version
ARG CMAKE_VERSION
COPY ./common/install_cmake.sh install_cmake.sh
RUN if [ -n "${CMAKE_VERSION}" ]; then bash ./install_cmake.sh; fi
RUN rm install_cmake.sh

ARG TRITON
# Install triton, this needs to be done before sccache because the latter will
# try to reach out to S3, which docker build runners don't have access
COPY ./common/install_triton.sh install_triton.sh
COPY ./common/common_utils.sh common_utils.sh
COPY ci_commit_pins/triton.txt triton.txt
COPY triton_version.txt triton_version.txt
RUN if [ -n "${TRITON}" ]; then bash ./install_triton.sh; fi
RUN rm install_triton.sh common_utils.sh triton.txt triton_version.txt

ARG HALIDE
# Build and install halide
COPY ./common/install_halide.sh install_halide.sh
COPY ./common/common_utils.sh common_utils.sh
COPY ci_commit_pins/halide.txt halide.txt
RUN if [ -n "${HALIDE}" ]; then bash ./install_halide.sh; fi
RUN rm install_halide.sh common_utils.sh halide.txt

# Install ccache/sccache (do this last, so we get priority in PATH)
COPY ./common/install_cache.sh install_cache.sh
ENV PATH /opt/cache/bin:$PATH
# See https://github.com/pytorch/pytorch/issues/82174
# TODO(sdym@fb.com):
# check if this is needed after full off Xenial migration
ENV CARGO_NET_GIT_FETCH_WITH_CLI true
RUN bash ./install_cache.sh && rm install_cache.sh
ENV CMAKE_CUDA_COMPILER_LAUNCHER=/opt/cache/bin/sccache

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

# AWS specific CUDA build guidance
ENV TORCH_CUDA_ARCH_LIST Maxwell
ENV TORCH_NVCC_FLAGS "-Xfatbin -compress-all"
ENV CUDA_PATH /usr/local/cuda

# Install LLVM dev version (Defined in the pytorch/builder github repository)
COPY --from=pytorch/llvm:9.0.1 /opt/llvm /opt/llvm

# Install CUDNN
ARG CUDNN_VERSION
ARG CUDA_VERSION
COPY ./common/install_cudnn.sh install_cudnn.sh
RUN if [ -n "${CUDNN_VERSION}" ]; then bash install_cudnn.sh; fi
RUN rm install_cudnn.sh

# Install CUSPARSELT
ARG CUDA_VERSION
COPY ./common/install_cusparselt.sh install_cusparselt.sh
RUN bash install_cusparselt.sh
RUN rm install_cusparselt.sh

# Install CUDSS
ARG CUDA_VERSION
COPY ./common/install_cudss.sh install_cudss.sh
RUN bash install_cudss.sh
RUN rm install_cudss.sh

# Delete /usr/local/cuda-11.X/cuda-11.X symlinks
RUN if [ -h /usr/local/cuda-11.6/cuda-11.6 ]; then rm /usr/local/cuda-11.6/cuda-11.6; fi
RUN if [ -h /usr/local/cuda-11.7/cuda-11.7 ]; then rm /usr/local/cuda-11.7/cuda-11.7; fi
RUN if [ -h /usr/local/cuda-12.1/cuda-12.1 ]; then rm /usr/local/cuda-12.1/cuda-12.1; fi
RUN if [ -h /usr/local/cuda-12.4/cuda-12.4 ]; then rm /usr/local/cuda-12.4/cuda-12.4; fi

USER jenkins
CMD ["bash"]
