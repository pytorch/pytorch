FROM mcr.microsoft.com/vscode/devcontainers/base:ubuntu-22.04

# Tools needed for development
RUN apt-get -y update && \
    apt-get install -y \
    build-essential \
    cmake \
    ninja-build \
    git \
    python3 \
    python3-pip \
    python3-dev \
    python3-venv \
    libopenblas-dev

# Tools needed for llvm
RUN apt-get install --no-install-recommends -y lsb-release wget software-properties-common gnupg && \
    sudo apt-get clean -y

# Create Python virtual environment
# RUN python3 -m venv /opt/venv
# ENV PATH="/opt/venv/bin:$PATH"
RUN pip3 install --upgrade pip

# Install CLANG if version is specified
ARG CLANG_VERSION
RUN if [ -n "$CLANG_VERSION" ]; then \
    wget https://apt.llvm.org/llvm.sh; \
    chmod +x llvm.sh; \
    ./llvm.sh "${CLANG_VERSION}"; \
    echo 'export CC=clang' >> ~/.bashrc; \
    echo 'export CXX=clang++' >> ~/.bashrc; \
    apt-get install --no-install-recommends -y clang libomp-dev && \
    apt-get clean -y; \
    fi


# Install CUDA if version is specified
ARG CUDA_VERSION
RUN if [ -n "$CUDA_VERSION" ]; then \
    CUDA_REPO_VERSION=$(echo ${CUDA_VERSION} | sed 's/\./\-/g'); \
    wget https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2204/x86_64/cuda-keyring_1.0-1_all.deb && \
    dpkg -i cuda-keyring_1.0-1_all.deb && \
    apt-get install --no-install-recommends -y cuda-toolkit-${CUDA_VERSION} && \
    apt-get clean -y; \
    fi

# Set PATH for CUDA
ENV PATH="/usr/local/cuda/bin:${PATH}"
ENV LD_LIBRARY_PATH="/usr/local/cuda/lib64:${LD_LIBRARY_PATH}"
ENV PIP_BREAK_SYSTEM_PACKAGES=1
