FROM mcr.microsoft.com/vscode/devcontainers/miniconda:0-3

#  I am suprised this is needed
RUN conda init

# Copy environment.yml (if found) to a temp location so we update the environment. Also
# copy "noop.txt" so the COPY instruction does not fail if no environment.yml exists.
COPY .devcontainer/cuda/environment.yml .devcontainer/noop.txt /tmp/conda-tmp/
RUN if [ -f "/tmp/conda-tmp/environment.yml" ]; then umask 0002 && /opt/conda/bin/conda env update -n base -f /tmp/conda-tmp/environment.yml; fi \
    && sudo rm -rf /tmp/conda-tmp

# Tools needed for llvm
RUN sudo apt-get -y update
RUN sudo apt install -y lsb-release wget software-properties-common gnupg

# Install CLANG if version is specified
ARG CLANG_VERSION
RUN if [ -n "$CLANG_VERSION" ]; then \
    sudo wget https://apt.llvm.org/llvm.sh; \
    chmod +x llvm.sh; \
    sudo ./llvm.sh "${CLANG_VERSION}"; \
    echo 'export CC=clang' >> ~/.bashrc; \
    echo 'export CXX=clang++' >> ~/.bashrc; \
    sudo apt update; \
    sudo apt install -y clang; \
    sudo apt install -y libomp-dev; \
    fi


# Install cuda if version is specified
ARG CUDA_VERSION
RUN if [ -n "$CUDA_VERSION" ]; then \
       conda install cuda -c "nvidia/label/cuda-${CUDA_VERSION}"; \
    fi
