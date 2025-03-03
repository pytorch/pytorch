# syntax=docker/dockerfile:1

# NOTE: Building this image require's docker version >= 23.0.
#
# For reference:
# - https://docs.docker.com/build/dockerfile/frontend/#stable-channel

ARG BASE_IMAGE=ubuntu:22.04
ARG PYTHON_VERSION=3.11

# Stage 1: Base Development Environment
FROM ${BASE_IMAGE} as dev-base
RUN apt-get update && DEBIAN_FRONTEND=noninteractive apt-get install -y --no-install-recommends \
        build-essential \
        ca-certificates \
        ccache \
        cmake \
        curl \
        git \
        libjpeg-dev \
        libpng-dev && \
    rm -rf /var/lib/apt/lists/*
RUN /usr/sbin/update-ccache-symlinks
RUN mkdir /opt/ccache && ccache --set-config=cache_dir=/opt/ccache
ENV PATH /opt/conda/bin:$PATH

# Stage 2: Conda Environment Setup (Dependencies Only)
FROM dev-base as conda
ARG PYTHON_VERSION=3.11
# Automatically set by buildx
ARG TARGETPLATFORM
# translating Docker's TARGETPLATFORM into miniconda arches
RUN case ${TARGETPLATFORM} in \
        "linux/arm64")  MINICONDA_ARCH=aarch64  ;; \
        *)              MINICONDA_ARCH=x86_64   ;; \
    esac && \
    curl -fsSL -v -o ~/miniconda.sh -O  "https://github.com/conda-forge/miniforge/releases/latest/download/Miniforge3-Linux-${MINICONDA_ARCH}.sh"
COPY requirements.txt .
# Manually invoke bash on miniconda script per https://github.com/conda/conda/issues/10431
RUN chmod +x ~/miniconda.sh && \
    bash ~/miniconda.sh -b -p /opt/conda && \
    rm ~/miniconda.sh && \
    /opt/conda/bin/conda install -y python=${PYTHON_VERSION} cmake conda-build pyyaml numpy ipython && \
    /opt/conda/bin/python -mpip install -r requirements.txt && \
    /opt/conda/bin/conda clean -ya

# Stage 3: Git Submodule Update
FROM dev-base as submodule-update
WORKDIR /opt/pytorch
COPY . .
RUN git submodule update --init --recursive

# Stage 4: Build Triton and Install Python Package
FROM conda as build
ARG CMAKE_VARS
WORKDIR /opt/pytorch
COPY --from=conda /opt/conda /opt/conda
COPY --from=submodule-update /opt/pytorch /opt/pytorch
RUN make triton
RUN --mount=type=cache,target=/opt/ccache \
    export eval ${CMAKE_VARS} && \
    TORCH_CUDA_ARCH_LIST="7.0 7.2 7.5 8.0 8.6 8.7 8.9 9.0 9.0a" TORCH_NVCC_FLAGS="-Xfatbin -compress-all" \
    CMAKE_PREFIX_PATH="$(dirname $(which conda))/../" \
    python setup.py install

# Stage 5: Torch dependencies Installation
FROM conda as torch-packages
ARG INSTALL_CHANNEL=whl/nightly
ARG CUDA_PATH=cu121
ARG PYTORCH_VERSION=2.7.0.dev20250217 #we need to set this 
ARG TARGETPLATFORM

# Install necessary tools for dependency extraction
RUN apt-get update && DEBIAN_FRONTEND=noninteractive apt-get install -y --no-install-recommends \
    jq \
    && rm -rf /var/lib/apt/lists/*

# Define packages to install
ENV PACKAGES="torch torchvision torchaudio"

# Step 1: Preinstall Torch Dependencies (Cached Separately)
RUN set -eux; \
    # Determine PIP_PRE_FLAG based on INSTALL_CHANNEL
    if echo "${INSTALL_CHANNEL}" | grep -q "nightly"; then \
        PIP_PRE_FLAG="--pre"; \
    else \
        PIP_PRE_FLAG=""; \
    fi; \
    # Determine EXTRA_INDEX_URL based on TARGETPLATFORM
    if [ "$TARGETPLATFORM" = "linux/arm64" ]; then \
        EXTRA_INDEX_URL="https://download.pytorch.org/whl/cpu/"; \
    else \
        EXTRA_INDEX_URL="https://download.pytorch.org/${INSTALL_CHANNEL}/${CUDA_PATH%.}/"; \
    fi; \
    echo "Using EXTRA_INDEX_URL=${EXTRA_INDEX_URL} with PIP_PRE_FLAG=${PIP_PRE_FLAG}"; \
    # Create a temporary directory to store dependencies
    TEMP_DIR=$(mktemp -d); \
    trap 'rm -rf "$TEMP_DIR"' EXIT; \
    DEP_REQUIREMENTS="${TEMP_DIR}/dependencies.txt"; \
    touch "$DEP_REQUIREMENTS"; \
    # Convert e.g. "3.11" to "311" for wheel filename
    PY_DOTLESS="$(echo "${PYTHON_VERSION}" | tr -d '.')"; \
    echo "PYTHON_VERSION=${PYTHON_VERSION} -> PY_DOTLESS=${PY_DOTLESS}"; \
    # Iterate over each package to collect dependencies
    for PACKAGE in $PACKAGES; do \
        echo "Fetching dependencies for $PACKAGE"; \
        if echo "${INSTALL_CHANNEL}" | grep -E '(nightly|test)' > /dev/null; then \
            METADATA_URL="https://download.pytorch.org/${INSTALL_CHANNEL}/${CUDA_PATH}/${PACKAGE}-${PYTORCH_VERSION}+${CUDA_PATH}-cp${PY_DOTLESS}-cp${PY_DOTLESS}-manylinux_2_28_x86_64.whl.metadata"; \
            echo "Attempting to fetch .whl.metadata from: ${METADATA_URL}"; \
            if curl -fsSL "${METADATA_URL}" -o "${TEMP_DIR}/${PACKAGE}.metadata"; then \
                jq -r '.requires_dist[]?' "${TEMP_DIR}/${PACKAGE}.metadata" >> "$DEP_REQUIREMENTS"; \
                rm -f "${TEMP_DIR}/${PACKAGE}.metadata"; \
            else \
                curl -fsSL "https://pypi.org/pypi/${PACKAGE}/json" -o "${TEMP_DIR}/${PACKAGE}.json"; \
                jq -r '.info.requires_dist[]?' "${TEMP_DIR}/${PACKAGE}.json" >> "$DEP_REQUIREMENTS"; \
                rm -f "${TEMP_DIR}/${PACKAGE}.json"; \
            fi; \
        else \
            curl -fsSL "https://pypi.org/pypi/${PACKAGE}/json" -o "${TEMP_DIR}/${PACKAGE}.json"; \
            jq -r '.info.requires_dist[]?' "${TEMP_DIR}/${PACKAGE}.json" >> "$DEP_REQUIREMENTS"; \
            rm -f "${TEMP_DIR}/${PACKAGE}.json"; \
        fi; \
    done; \
    # Remove duplicates while preserving environment markers
    sort -u "$DEP_REQUIREMENTS" -o "$DEP_REQUIREMENTS"; \
    \
    # Install dependencies using pip with the requirements file
    if [ -s "$DEP_REQUIREMENTS" ]; then \
        echo "Installing dependencies from $DEP_REQUIREMENTS"; \
        pip install --no-cache-dir $PIP_PRE_FLAG --extra-index-url "$EXTRA_INDEX_URL" -r "$DEP_REQUIREMENTS"; \
    else \
        echo "No dependencies to install."; \
    fi;

# Step 2: Install Torch packages without dependencies
RUN set -eux; \
    if echo "${INSTALL_CHANNEL}" | grep -q "nightly"; then \
    PIP_PRE_FLAG="--pre"; \
    else \
    PIP_PRE_FLAG=""; \
    fi; \
    if [ "$TARGETPLATFORM" = "linux/arm64" ]; then \
    INDEX_URL="https://download.pytorch.org/whl/cpu/"; \
    else \
    INDEX_URL="https://download.pytorch.org/${INSTALL_CHANNEL}/${CUDA_PATH%.}/"; \
    fi; \
    echo "Installing main packages with PIP_PRE_FLAG=${PIP_PRE_FLAG}"; \
    pip install --no-cache-dir $PACKAGES --no-deps $PIP_PRE_FLAG --index-url $INDEX_URL && \
    pip install torchelastic && \
    /opt/conda/bin/conda clean -ya

# Stage 6: Official Environment Setup
FROM ${BASE_IMAGE} as official
ARG PYTORCH_VERSION
ARG TRITON_VERSION
ARG TARGETPLATFORM
ARG CUDA_VERSION
LABEL com.nvidia.volumes.needed="nvidia_driver"
RUN apt-get update && DEBIAN_FRONTEND=noninteractive apt-get install -y --no-install-recommends \
        ca-certificates \
        g++ \
        libjpeg-dev \
        libpng-dev \
        && rm -rf /var/lib/apt/lists/*

# Copy the Conda environment with dependencies only
COPY --from=conda /opt/conda /opt/conda

# Install main Torch packages in the official stage
COPY --from=torch-packages /opt/conda /opt/conda

RUN if test -n "${TRITON_VERSION}" -a "${TARGETPLATFORM}" != "linux/arm64"; then \
        DEBIAN_FRONTEND=noninteractive apt install -y --no-install-recommends gcc; \
    rm -rf /var/lib/apt/lists/*; \
    fi
ENV PATH /opt/conda/bin:$PATH
ENV NVIDIA_VISIBLE_DEVICES all
ENV NVIDIA_DRIVER_CAPABILITIES compute,utility
ENV LD_LIBRARY_PATH /usr/local/nvidia/lib:/usr/local/nvidia/lib64
ENV PATH /usr/local/nvidia/bin:/usr/local/cuda/bin:$PATH
ENV PYTORCH_VERSION=${PYTORCH_VERSION}
WORKDIR /workspace

# Stage 7: Development Environment
FROM official as dev
# Should override the already installed version from the official-image stage
COPY --from=build /opt/conda /opt/conda
