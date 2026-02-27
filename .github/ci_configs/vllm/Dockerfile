ARG CUDA_VERSION=12.9.1
ARG PYTHON_VERSION=3.12

# BUILD_BASE_IMAGE: used to build vllm wheels, It can be replaced with a different base image from local machine,
# by default, it uses the torch-nightly-base stage from this docker image
ARG BUILD_BASE_IMAGE=torch-nightly-base
ARG FINAL_BASE_IMAGE=nvidia/cuda:${CUDA_VERSION}-devel-ubuntu22.04

# The logic is copied from https://github.com/vllm-project/vllm/blob/main/docker/Dockerfile
ARG GET_PIP_URL="https://bootstrap.pypa.io/get-pip.py"

#################### TORCH NIGHTLY BASE IMAGE ####################
FROM nvidia/cuda:${CUDA_VERSION}-devel-ubuntu22.04 as torch-nightly-base

ARG CUDA_VERSION
ARG PYTHON_VERSION
ARG GET_PIP_URL

# Install system dependencies and uv, then create Python virtual environment
RUN apt-get update -y \
    && apt-get install -y ccache software-properties-common git curl sudo vim python3-pip \
    && curl -LsSf https://astral.sh/uv/install.sh | sh \
    && $HOME/.local/bin/uv venv /opt/venv --python ${PYTHON_VERSION} \
    && rm -f /usr/bin/python3 /usr/bin/python3-config /usr/bin/pip \
    && ln -s /opt/venv/bin/python3 /usr/bin/python3 \
    && ln -s /opt/venv/bin/python3-config /usr/bin/python3-config \
    && ln -s /opt/venv/bin/pip /usr/bin/pip \
    && python3 --version && python3 -m pip --version

# Upgrade to GCC 10 to avoid https://gcc.gnu.org/bugzilla/show_bug.cgi?id=92519
# as it was causing spam when compiling the CUTLASS kernels
RUN apt-get install -y gcc-10 g++-10
RUN update-alternatives --install /usr/bin/gcc gcc /usr/bin/gcc-10 110 --slave /usr/bin/g++ g++ /usr/bin/g++-10
RUN <<EOF
gcc --version
EOF

# Install uv for faster pip installs
RUN --mount=type=cache,target=/root/.cache/uv \
    python3 -m pip install uv==0.8.4

ENV UV_HTTP_TIMEOUT=500
ENV UV_INDEX_STRATEGY="unsafe-best-match"
# Use copy mode to avoid hardlink failures with Docker cache mounts
ENV UV_LINK_MODE=copy
#################### TORCH NIGHTLY  BASE IMAGE ####################


#################### BASE BUILD IMAGE ####################
FROM ${BUILD_BASE_IMAGE} AS base
USER root

ARG CUDA_VERSION
ARG PYTHON_VERSION

# Only work with PyTorch manylinux builder
ENV PATH="/opt/python/cp312-cp312/bin:${PATH}"

# Install some system dependencies and double check python version
RUN if command -v apt-get >/dev/null; then \
        apt-get update -y \
        && apt-get install -y ccache software-properties-common git wget sudo vim; \
    else \
        dnf install -y git wget sudo; \
    fi \
    && python3 --version && python3 -m pip --version

# Install uv for faster pip installs if not existed
RUN --mount=type=cache,target=/root/.cache/uv \
    python3 -m pip install uv==0.8.4

ENV UV_HTTP_TIMEOUT=500
ENV UV_INDEX_STRATEGY="unsafe-best-match"
# Use copy mode to avoid hardlink failures with Docker cache mounts
ENV UV_LINK_MODE=copy

WORKDIR /workspace

# Install build and runtime dependencies
COPY requirements/common.txt requirements/common.txt
COPY use_existing_torch.py use_existing_torch.py
COPY pyproject.toml pyproject.toml

# Install build and runtime dependencies without stable torch version
RUN python3 use_existing_torch.py

# Default mount file as placeholder, this just avoid the mount error
# change to a different vllm folder if this does not exist anymore
ARG TORCH_WHEELS_PATH="./requirements"
ARG PINNED_TORCH_VERSION

# Install torch, torchaudio and torchvision based on the input
# if TORCH_WHEELS_PATH is default "./requirements", it will pull thethe nightly versions using pip
# otherwise, it will use the whls from TORCH_WHEELS_PATH from the host machine
RUN --mount=type=bind,source=${TORCH_WHEELS_PATH},target=/dist \
    --mount=type=cache,target=/root/.cache/uv \
    if [ -n "$TORCH_WHEELS_PATH" ] && [ "$TORCH_WHEELS_PATH" != "./requirements" ] && [ -d "/dist" ] && ls /dist/torch*.whl >/dev/null 2>&1; then \
        echo "[INFO] Installing torch wheels to build vllm"; \
        torch_whl=$(find /dist -maxdepth 1 -name 'torch-*.whl' -print -quit); \
        vision_whl=$(find /dist -name 'torchvision*.whl' | head -n1 | xargs); \
        audio_whl=$(find /dist -name 'torchaudio*.whl' | head -n1 | xargs); \
        uv pip install --system "${torch_whl}[opt-einsum]" "${vision_whl}" "${audio_whl}" /dist/*.whl; \
    elif [ -n "$PINNED_TORCH_VERSION" ]; then \
        echo "[INFO] Installing pinned torch nightly version to build vllm: $PINNED_TORCH_VERSION"; \
        uv pip install --system "$PINNED_TORCH_VERSION" --index-url https://download.pytorch.org/whl/nightly/cu$(echo $CUDA_VERSION | cut -d. -f1,2 | tr -d '.'); \
    else \
        echo "[INFO] Installing torch nightly with latest one to build vllm"; \
        uv pip install --system torch torchvision torchaudio --index-url https://download.pytorch.org/whl/nightly/cu$(echo $CUDA_VERSION | cut -d. -f1,2 | tr -d '.'); \
    fi

# Install numba 0.61.2 for cuda environment
RUN --mount=type=cache,target=/root/.cache/uv \
    uv pip install --system numba==0.61.2

# Install common dependencies from vllm common.txt
RUN --mount=type=cache,target=/root/.cache/uv \
    uv pip install --system -r requirements/common.txt

RUN uv pip freeze | grep -i '^torch\|^torchvision\|^torchaudio' > torch_build_versions.txt
RUN cat torch_build_versions.txt
RUN pip freeze | grep -E 'torch|torchvision|torchaudio'
#################### BASE BUILD IMAGE ####################


#################### WHEEL BUILD IMAGE ####################
FROM base AS build
ARG TARGETPLATFORM

COPY . .
RUN python3 use_existing_torch.py

RUN --mount=type=cache,target=/root/.cache/uv \
    uv pip install --system -r requirements/build.txt

ARG GIT_REPO_CHECK=0
RUN --mount=type=bind,source=.git,target=.git \
    if [ "$GIT_REPO_CHECK" != "0" ]; then bash tools/check_repo.sh ; fi

ARG max_jobs=16
ENV MAX_JOBS=${max_jobs}
ARG nvcc_threads=8
ENV NVCC_THREADS=$nvcc_threads

ARG USE_SCCACHE
ARG SCCACHE_BUCKET_NAME=vllm-build-sccache
ARG SCCACHE_REGION_NAME=us-west-2
ARG SCCACHE_S3_NO_CREDENTIALS=0

ARG torch_cuda_arch_list='8.0;8.9;9.0;10.0;12.0'
ENV TORCH_CUDA_ARCH_LIST=${torch_cuda_arch_list}

# Use sccache to speed up compilation
RUN --mount=type=cache,target=/root/.cache/uv \
    --mount=type=bind,source=.git,target=.git \
    if [ "$USE_SCCACHE" = "1" ]; then \
        echo "Installing sccache..."; \
        if [ "$TARGETPLATFORM" = "linux/arm64" ]; then \
            SCCACHE_ARCHIVE="sccache-v0.8.1-aarch64-unknown-linux-musl"; \
        else \
            SCCACHE_ARCHIVE="sccache-v0.8.1-x86_64-unknown-linux-musl"; \
        fi; \
        curl -L -o sccache.tar.gz "https://github.com/mozilla/sccache/releases/download/v0.8.1/${SCCACHE_ARCHIVE}.tar.gz" \
        && tar -xzf sccache.tar.gz \
        && sudo mv "${SCCACHE_ARCHIVE}"/sccache /usr/bin/sccache \
        && rm -rf sccache.tar.gz "${SCCACHE_ARCHIVE}" \
        && export SCCACHE_BUCKET=${SCCACHE_BUCKET_NAME} \
        && export SCCACHE_REGION=${SCCACHE_REGION_NAME} \
        && export SCCACHE_S3_NO_CREDENTIALS=${SCCACHE_S3_NO_CREDENTIALS} \
        && export SCCACHE_IDLE_TIMEOUT=0 \
        && export CMAKE_BUILD_TYPE=Release \
        && export VLLM_DOCKER_BUILD_CONTEXT=1 \
        && sccache --show-stats \
        && python3 setup.py bdist_wheel --dist-dir=vllm-dist --py-limited-api=cp38 \
        && sccache --show-stats; \
    fi

ARG vllm_target_device="cuda"
ENV VLLM_TARGET_DEVICE=${vllm_target_device}
ENV CCACHE_DIR=/root/.cache/ccache
RUN --mount=type=cache,target=/root/.cache/ccache \
    --mount=type=cache,target=/root/.cache/uv \
    --mount=type=bind,source=.git,target=.git  \
    if [ "$USE_SCCACHE" != "1" ]; then \
        # Clean any existing CMake artifacts
        rm -rf .deps && \
        mkdir -p .deps && \
        export VLLM_DOCKER_BUILD_CONTEXT=1 && \
        python3 setup.py bdist_wheel --dist-dir=vllm-dist --py-limited-api=cp38; \
    fi
#################### WHEEL BUILD IMAGE ####################


################### VLLM INSTALLED IMAGE ####################
FROM ${FINAL_BASE_IMAGE} AS vllm-base
USER root

ARG CUDA_VERSION
ARG PYTHON_VERSION
ARG GET_PIP_URL

# Only work with PyTorch manylinux builder
ENV PATH="/opt/python/cp312-cp312/bin:${PATH}"

# prepare for environment starts
WORKDIR /workspace

# Install Python and other dependencies
RUN if command -v apt-get >/dev/null; then \
        apt-get update -y \
        && apt-get install -y ccache software-properties-common git sudo vim python3-pip; \
    else \
        dnf install -y git wget sudo; \
    fi \
    && curl -LsSf https://astral.sh/uv/install.sh | sh \
    && $HOME/.local/bin/uv venv /opt/venv --python ${PYTHON_VERSION} \
    && rm -f /usr/bin/python3 /usr/bin/python3-config /usr/bin/pip \
    && ln -s /opt/venv/bin/python3 /usr/bin/python3 \
    && ln -s /opt/venv/bin/python3-config /usr/bin/python3-config \
    && ln -s /opt/venv/bin/pip /usr/bin/pip \
    && python3 --version && python3 -m pip --version

# Get the torch versions, and whls used in previous stage
COPY --from=base /workspace/torch_build_versions.txt ./torch_build_versions.txt
COPY --from=build /workspace/vllm-dist /wheels/vllm
RUN echo "[INFO] Listing current directory before torch install step:" && \
    ls -al && \
    echo "[INFO] Showing torch_build_versions.txt content:" && \
    cat torch_build_versions.txt

# Install uv for faster pip installs if not existed
RUN --mount=type=cache,target=/root/.cache/uv \
    python3 -m pip install uv==0.8.4

ENV UV_HTTP_TIMEOUT=500
ENV UV_INDEX_STRATEGY="unsafe-best-match"
# Use copy mode to avoid hardlink failures with Docker cache mounts
ENV UV_LINK_MODE=copy

# Install build and runtime dependencies
COPY requirements/build.txt requirements/build.txt
COPY use_existing_torch.py use_existing_torch.py
RUN python3 use_existing_torch.py
RUN cat requirements/build.txt

RUN --mount=type=cache,target=/root/.cache/uv \
    uv pip install --system -r requirements/build.txt

# Default mount file as placeholder, this just avoid the mount error
ARG TORCH_WHEELS_PATH="./requirements"
# Install torch, torchaudio and torchvision. If TORCH_WHEELS_PATH is default
# to ./requirements, it will pull the nightly versions using pip. Otherwise,
# it will use the local wheels from TORCH_WHEELS_PATH
RUN --mount=type=bind,source=${TORCH_WHEELS_PATH},target=/dist \
    --mount=type=cache,target=/root/.cache/uv \
    if [ -n "$TORCH_WHEELS_PATH" ] && [ "$TORCH_WHEELS_PATH" != "./requirements" ] && [ -d "/dist" ] && ls /dist/torch*.whl >/dev/null 2>&1; then \
        torch_whl=$(find /dist -maxdepth 1 -name 'torch-*.whl' -print -quit); \
        vision_whl=$(find /dist -name 'torchvision*.whl' | head -n1 | xargs); \
        audio_whl=$(find /dist -name 'torchaudio*.whl' | head -n1 | xargs); \
        echo "[INFO] Use wheels to build : '${torch_whl}' '${audio_whl}' '${vision_whl}'"; \
        uv pip install --system "${torch_whl}[opt-einsum]" "${vision_whl}" "${audio_whl}" /dist/*.whl; \
    else \
        echo "[INFO] Installing torch versions from torch_build_versions.txt"; \
        uv pip install --system $(cat torch_build_versions.txt | xargs) --index-url https://download.pytorch.org/whl/nightly/cu$(echo $CUDA_VERSION | cut -d. -f1,2 | tr -d '.'); \
    fi

RUN --mount=type=cache,target=/root/.cache/uv \
    uv pip install --system --pre apache-tvm-ffi==0.1.0b15

# Install the vllm wheel from previous stage
RUN --mount=type=cache,target=/root/.cache/uv \
    uv pip install --system /wheels/vllm/*.whl --verbose

# Logging to confirm the torch versions
RUN pip freeze | grep -E 'torch|vllm'
RUN uv pip freeze | grep -i '^torch\|^torchvision\|^torchaudio\|^vllm' > build_summary.txt
################### VLLM INSTALLED IMAGE ####################


#################### EXPORT STAGE ####################
FROM scratch as export-wheels

# Just copy the wheels we prepared in previous stages
COPY --from=build /workspace/vllm-dist /wheels/vllm
COPY --from=vllm-base /workspace/build_summary.txt /wheels/build_summary.txt
