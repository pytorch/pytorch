ARG UBUNTU_VERSION

FROM ubuntu:${UBUNTU_VERSION}

ARG UBUNTU_VERSION

ENV DEBIAN_FRONTEND noninteractive
ENV PIP_NO_CACHE_DIR=1

# Set AMD gpu targets to build for
ARG PYTORCH_ROCM_ARCH
ENV PYTORCH_ROCM_ARCH ${PYTORCH_ROCM_ARCH}

# Install common dependencies (so that this step can be cached separately)
COPY ./common/install_base.sh install_base.sh
RUN bash ./install_base.sh && rm install_base.sh
RUN apt-get update && apt-get install -y --no-install-recommends libtbb-dev && rm -rf /var/lib/apt/lists/*

# Install user
COPY ./common/install_user.sh install_user.sh
RUN bash ./install_user.sh && rm install_user.sh

# Install katex
ARG KATEX
COPY ./common/install_docs_reqs.sh install_docs_reqs.sh
RUN bash ./install_docs_reqs.sh && rm install_docs_reqs.sh

# Install conda and other packages (e.g., numpy, pytest)
ARG ANACONDA_PYTHON_VERSION
ARG BUILD_ENVIRONMENT
ENV ANACONDA_PYTHON_VERSION=$ANACONDA_PYTHON_VERSION
ENV PATH /opt/conda/envs/py_$ANACONDA_PYTHON_VERSION/bin:/opt/conda/bin:$PATH
COPY requirements-ci.txt /opt/conda/requirements-ci.txt
COPY ./common/install_conda.sh install_conda.sh
COPY ./common/common_utils.sh common_utils.sh
RUN bash ./install_conda.sh && rm install_conda.sh common_utils.sh /opt/conda/requirements-ci.txt

# Install gcc
ARG GCC_VERSION
COPY ./common/install_gcc.sh install_gcc.sh
RUN bash ./install_gcc.sh && rm install_gcc.sh

# Install clang
ARG CLANG_VERSION
COPY ./common/install_clang.sh install_clang.sh
RUN bash ./install_clang.sh && rm install_clang.sh

# Install rocm
ARG ROCM_VERSION
ARG THEROCK_INDEX_URL
ARG USE_MSLK
ENV ROCM_VERSION=${ROCM_VERSION}
COPY ./common/install_rocm.sh install_rocm.sh
RUN bash ./install_rocm.sh && rm install_rocm.sh

# ROCm environment is written by install_rocm.sh to /etc/rocm_env.sh and sourced
# by CI scripts. TheRock installs ROCm under the Python environment, not /opt/rocm.

ENV LANG C.UTF-8
ENV LC_ALL C.UTF-8

# Install amdsmi
COPY ./common/install_amdsmi.sh install_amdsmi.sh
RUN bash ./install_amdsmi.sh
RUN rm install_amdsmi.sh

ARG INDUCTOR_BENCHMARKS
ARG ANACONDA_PYTHON_VERSION
ENV ANACONDA_PYTHON_VERSION=$ANACONDA_PYTHON_VERSION
COPY ./common/install_inductor_benchmark_deps.sh install_inductor_benchmark_deps.sh
COPY ./common/common_utils.sh common_utils.sh
COPY ci_commit_pins/huggingface-requirements.txt huggingface-requirements.txt
COPY ci_commit_pins/timm.txt timm.txt
COPY ci_commit_pins/torchbench.txt torchbench.txt
RUN if [ -n "${INDUCTOR_BENCHMARKS}" ]; then bash ./install_inductor_benchmark_deps.sh; fi
RUN rm install_inductor_benchmark_deps.sh common_utils.sh timm.txt huggingface-requirements.txt torchbench.txt

ARG TRITON
# Install triton, this needs to be done before sccache because the latter will
# try to reach out to S3, which docker build runners don't have access
COPY ./common/install_triton.sh install_triton.sh
COPY ./common/common_utils.sh common_utils.sh
COPY ci_commit_pins/triton.txt triton.txt
COPY triton_version.txt triton_version.txt
RUN if [ -n "${TRITON}" ]; then bash ./install_triton.sh; fi
RUN rm install_triton.sh common_utils.sh triton.txt triton_version.txt


# Install Rust toolchain (needed to build the torch._rust extension).
COPY ./common/install_rust.sh install_rust.sh
COPY ./ci_commit_pins/rust.txt rust.txt
ENV RUSTUP_HOME /opt/rust
ENV CARGO_HOME /opt/rust
ENV PATH /opt/rust/bin:$PATH
RUN bash ./install_rust.sh && rm install_rust.sh rust.txt

# Install ccache/sccache (do this last, so we get priority in PATH)
COPY ./common/install_cache.sh install_cache.sh
COPY ./common/patches /opt/cache/patches
ENV PATH /opt/cache/bin:$PATH
RUN bash ./install_cache.sh && rm install_cache.sh

# Install Open MPI for ROCm
COPY ./common/install_openmpi.sh install_openmpi.sh
RUN if [ -n "${CUDA_VERSION}" ]; then bash install_openmpi.sh; fi
RUN rm install_openmpi.sh

# Install rocm-origami
COPY ./common/common_utils.sh common_utils.sh
COPY ci_commit_pins/rocm-origami.txt rocm-origami.txt
RUN bash -c 'source /etc/rocm_env.sh && source common_utils.sh && env_run pip install "rocm-origami==$(cat rocm-origami.txt)"' && rm -f common_utils.sh rocm-origami.txt

ARG BUILD_ENVIRONMENT
ENV BUILD_ENVIRONMENT ${BUILD_ENVIRONMENT}

USER jenkins
CMD ["bash"]
