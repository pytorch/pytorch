# syntax=docker/dockerfile:1
ARG UBUNTU_VERSION

# =============================================================================
# Stage: openssl-builder — parallel with base
# =============================================================================
FROM ubuntu:${UBUNTU_VERSION} AS openssl-builder
ENV DEBIAN_FRONTEND=noninteractive
RUN apt-get update && apt-get install -y --no-install-recommends \
    wget make gcc libc6-dev perl sudo ca-certificates
RUN --mount=type=bind,source=./common/install_openssl.sh,target=/install_openssl.sh \
    bash /install_openssl.sh

# =============================================================================
# Stage: base
# =============================================================================
FROM ubuntu:${UBUNTU_VERSION} AS base

ARG UBUNTU_VERSION
ARG CLANG_VERSION

ENV DEBIAN_FRONTEND=noninteractive

# Allow APT to run as root
RUN echo 'APT::Sandbox::User "root";' | tee -a /etc/apt/apt.conf.d/10sandbox

# Install common dependencies
RUN --mount=type=bind,source=./common/install_base.sh,target=/install_base.sh \
    bash /install_base.sh

# Install sccache early so cmake-based compilation steps (Halide, Triton,
# executorch) benefit from cached object files. sccache auto-detects its
# backend from env: SCCACHE_BUCKET for S3 (CI), SCCACHE_DIR for local disk.
ARG SCCACHE_BUCKET
ARG SCCACHE_REGION
ARG SCCACHE_S3_USE_SSL
RUN curl -sL "https://github.com/mozilla/sccache/releases/download/v0.13.0/sccache-v0.13.0-$(uname -m)-unknown-linux-musl.tar.gz" \
    | tar xz --strip-components=1 -C /usr/local/bin/ --wildcards '*/sccache' \
    || echo "WARNING: sccache download failed, builds will proceed without compilation cache"
ENV CMAKE_C_COMPILER_LAUNCHER=sccache
ENV CMAKE_CXX_COMPILER_LAUNCHER=sccache

# Install clang
ARG LLVMDEV
RUN --mount=type=bind,source=./common/install_clang.sh,target=/install_clang.sh \
    bash /install_clang.sh

# Install user
RUN --mount=type=bind,source=./common/install_user.sh,target=/install_user.sh \
    bash /install_user.sh

# Install katex
ARG KATEX
RUN --mount=type=bind,source=./common/install_docs_reqs.sh,target=/install_docs_reqs.sh \
    bash /install_docs_reqs.sh

# Install conda and other packages (e.g., numpy, pytest)
ARG ANACONDA_PYTHON_VERSION
ARG PYTHON_FREETHREADED
ARG DOCS
ENV ANACONDA_PYTHON_VERSION=$ANACONDA_PYTHON_VERSION
ENV PYTHON_FREETHREADED=$PYTHON_FREETHREADED
ENV PATH=/opt/conda/envs/py_$ANACONDA_PYTHON_VERSION/bin:/opt/conda/bin:$PATH
ENV DOCS=$DOCS
RUN --mount=type=bind,source=./common/install_conda.sh,target=/install_conda.sh \
    --mount=type=bind,source=./common/common_utils.sh,target=/common_utils.sh \
    --mount=type=bind,source=./common/install_magma_conda.sh,target=/install_magma_conda.sh \
    --mount=type=bind,source=requirements-ci.txt,target=/opt/conda/requirements-ci.txt \
    --mount=type=bind,source=requirements-docs.txt,target=/opt/conda/requirements-docs.txt \
    bash /install_conda.sh

ARG UNINSTALL_DILL
RUN if [ -n "${UNINSTALL_DILL}" ]; then pip uninstall -y dill; fi

# Install gcc
ARG GCC_VERSION
RUN --mount=type=bind,source=./common/install_gcc.sh,target=/install_gcc.sh \
    bash /install_gcc.sh

# Install lcov for C++ code coverage
RUN --mount=type=bind,source=./common/install_lcov.sh,target=/install_lcov.sh \
    bash /install_lcov.sh

# Install cuda and cudnn
ARG CUDA_VERSION
RUN --mount=type=bind,source=./common/install_cuda.sh,target=/install_cuda.sh \
    --mount=type=bind,source=./common/install_nccl.sh,target=/install_nccl.sh \
    --mount=type=bind,source=./common/install_cusparselt.sh,target=/install_cusparselt.sh \
    --mount=type=bind,source=ci_commit_pins/,target=/ci_commit_pins/ \
    bash /install_cuda.sh ${CUDA_VERSION}
ENV DESIRED_CUDA=${CUDA_VERSION}
ENV PATH=/usr/local/nvidia/bin:/usr/local/cuda/bin:$PATH
ENV USE_SYSTEM_NCCL=1
ENV NCCL_INCLUDE_DIR="/usr/local/cuda/include/"
ENV NCCL_LIB_DIR="/usr/local/cuda/lib64/"

# (optional) Install UCC
ARG UCX_COMMIT
ARG UCC_COMMIT
ENV UCX_COMMIT=$UCX_COMMIT
ENV UCC_COMMIT=$UCC_COMMIT
ENV UCX_HOME=/usr
ENV UCC_HOME=/usr
RUN --mount=type=bind,source=./common/install_ucc.sh,target=/install_ucc.sh \
    if [ -n "${UCX_COMMIT}" ] && [ -n "${UCC_COMMIT}" ]; then bash /install_ucc.sh; fi

# (optional) Install vision packages like OpenCV
ARG VISION
RUN --mount=type=bind,source=./common/install_vision.sh,target=/install_vision.sh \
    --mount=type=bind,source=./common/cache_vision_models.sh,target=/cache_vision_models.sh \
    --mount=type=bind,source=./common/common_utils.sh,target=/common_utils.sh \
    if [ -n "${VISION}" ]; then bash /install_vision.sh; fi
ENV INSTALLED_VISION=${VISION}

# (optional) Install non-default Ninja version
ARG NINJA_VERSION
RUN --mount=type=bind,source=./common/install_ninja.sh,target=/install_ninja.sh \
    if [ -n "${NINJA_VERSION}" ]; then bash /install_ninja.sh; fi

# Install inductor benchmark deps
ARG INDUCTOR_BENCHMARKS
ENV BUILD_AOT_INDUCTOR_TEST=${INDUCTOR_BENCHMARKS}
RUN --mount=type=bind,source=./common/install_inductor_benchmark_deps.sh,target=/install_inductor_benchmark_deps.sh \
    --mount=type=bind,source=./common/common_utils.sh,target=/common_utils.sh \
    --mount=type=bind,source=ci_commit_pins/huggingface-requirements.txt,target=/huggingface-requirements.txt \
    --mount=type=bind,source=ci_commit_pins/timm.txt,target=/timm.txt \
    --mount=type=bind,source=ci_commit_pins/torchbench.txt,target=/torchbench.txt \
    if [ -n "${INDUCTOR_BENCHMARKS}" ]; then bash /install_inductor_benchmark_deps.sh; fi

ARG INSTALL_MINGW
RUN --mount=type=bind,source=./common/install_mingw.sh,target=/install_mingw.sh \
    if [ -n "${INSTALL_MINGW}" ]; then bash /install_mingw.sh; fi

# =============================================================================
# Stage: triton-builder — parallel cmake build benefits from sccache
# =============================================================================
ARG TRITON
ARG TRITON_CPU

FROM base AS triton-builder
ARG TRITON
ARG TRITON_CPU
ARG UBUNTU_VERSION
ARG GCC_VERSION
ARG CLANG_VERSION
RUN --mount=type=bind,source=./common/install_triton.sh,target=/install_triton.sh \
    --mount=type=bind,source=./common/common_utils.sh,target=/common_utils.sh \
    --mount=type=bind,source=ci_commit_pins/triton.txt,target=/triton.txt \
    --mount=type=bind,source=ci_commit_pins/triton-cpu.txt,target=/triton-cpu.txt \
    bash /install_triton.sh

# =============================================================================
# Stage: final
# =============================================================================
FROM base AS final

ARG TRITON
ARG TRITON_CPU
ARG CUDA_VERSION

# Copy OpenSSL from parallel builder
COPY --from=openssl-builder /opt/openssl /opt/openssl
RUN ln -sf /opt/openssl/lib/lib* /usr/lib
ENV OPENSSL_ROOT_DIR=/opt/openssl
ENV OPENSSL_DIR=/opt/openssl

COPY --from=triton-builder /opt/triton /opt/triton
RUN if [ -n "${TRITON}" ] || [ -n "${TRITON_CPU}" ]; then pip install /opt/triton/*.whl; chown -R jenkins:jenkins /opt/conda; fi
RUN rm -rf /opt/triton

ARG EXECUTORCH
RUN --mount=type=bind,source=./common/install_executorch.sh,target=/install_executorch.sh \
    --mount=type=bind,source=./common/common_utils.sh,target=/common_utils.sh \
    --mount=type=bind,source=ci_commit_pins/executorch.txt,target=/executorch.txt \
    if [ -n "${EXECUTORCH}" ]; then bash /install_executorch.sh; fi

ARG HALIDE
RUN --mount=type=bind,source=./common/install_halide.sh,target=/install_halide.sh \
    --mount=type=bind,source=./common/common_utils.sh,target=/common_utils.sh \
    --mount=type=bind,source=ci_commit_pins/halide.txt,target=/halide.txt \
    if [ -n "${HALIDE}" ]; then bash /install_halide.sh; fi

ARG PALLAS
ARG TPU
RUN --mount=type=bind,source=./common/requirements_tpu.txt,target=/requirements_tpu.txt \
    --mount=type=bind,source=./common/install_jax.sh,target=/install_jax.sh \
    --mount=type=bind,source=./common/common_utils.sh,target=/common_utils.sh \
    --mount=type=bind,source=ci_commit_pins/jax.txt,target=/ci_commit_pins/jax.txt \
    if [ -n "${TPU}" ]; then \
      bash -c "source /common_utils.sh && pip_install -r /requirements_tpu.txt" && \
      bash /install_jax.sh tpu; \
    elif [ -n "${PALLAS}" ]; then \
      bash /install_jax.sh ${CUDA_VERSION:-cpu}; \
    fi

ARG ONNX
RUN --mount=type=bind,source=./common/install_onnx.sh,target=/install_onnx.sh \
    --mount=type=bind,source=./common/common_utils.sh,target=/common_utils.sh \
    if [ -n "${ONNX}" ]; then bash /install_onnx.sh; fi

ARG ACL
RUN --mount=type=bind,source=./common/install_acl.sh,target=/install_acl.sh \
    if [ -n "${ACL}" ]; then bash /install_acl.sh; fi
ENV INSTALLED_ACL=${ACL}

ARG OPENBLAS
RUN --mount=type=bind,source=./common/install_openblas.sh,target=/install_openblas.sh \
    if [ -n "${OPENBLAS}" ]; then bash /install_openblas.sh; fi
ENV INSTALLED_OPENBLAS=${OPENBLAS}

# Install ccache/sccache
ARG SKIP_SCCACHE_INSTALL
COPY ./common/install_cache.sh install_cache.sh
ENV PATH=/opt/cache/bin:$PATH
RUN if [ -z "${SKIP_SCCACHE_INSTALL}" ]; then bash ./install_cache.sh; fi
RUN rm install_cache.sh

# Add jni.h for java host build
RUN --mount=type=bind,source=./common/install_jni.sh,target=/install_jni.sh \
    --mount=type=bind,source=./java/jni.h,target=/jni.h \
    bash /install_jni.sh

# Install Open MPI for CUDA
RUN --mount=type=bind,source=./common/install_openmpi.sh,target=/install_openmpi.sh \
    if [ -n "${CUDA_VERSION}" ]; then bash /install_openmpi.sh; fi

ARG BUILD_ENVIRONMENT
ENV BUILD_ENVIRONMENT=${BUILD_ENVIRONMENT}

ARG SKIP_LLVM_SRC_BUILD_INSTALL
COPY --from=pytorch/llvm:9.0.1 /opt/llvm /opt/llvm
RUN if [ -n "${SKIP_LLVM_SRC_BUILD_INSTALL}" ]; then rm -rf /opt/llvm; fi

ENV TORCH_NVCC_FLAGS="-Xfatbin -compress-all"
ENV CUDA_PATH=/usr/local/cuda

USER jenkins
CMD ["bash"]
