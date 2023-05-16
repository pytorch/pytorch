FROM mcr.microsoft.com/vscode/devcontainers/miniconda:0-3

#  I am suprised this is needed
RUN conda init

# Copy environment.yml (if found) to a temp location so we update the environment. Also
# copy "noop.txt" so the COPY instruction does not fail if no environment.yml exists.
COPY .devcontainer/cuda/environment.yml .devcontainer/noop.txt /tmp/conda-tmp/
RUN if [ -f "/tmp/conda-tmp/environment.yml" ]; then umask 0002 && /opt/conda/bin/conda env update -n base -f /tmp/conda-tmp/environment.yml; fi \
    && sudo rm -rf /tmp/conda-tmp

# Install the full llvm sweet
# Tools needed for llvm
RUN sudo apt-get -y update
RUN sudo apt install -y lsb-release wget software-properties-common gnupg

# # If USE_CLANG is set we will install
# Can't build succesfully, The exports need to go in the .sh script
# But also clang-15 might be two new for cuda 11.8

# ARG USE_CLANG
# RUN if [ -n "$USE_CLANG" ]; then \
#     sudo bash -c "$(wget -O - https://apt.llvm.org/llvm.sh)"; \
#     sudo .devcontainer/scripts/update_alternatives_clang.sh 15 10; \
#     export CC=clang; \
#     export CXX=clang++; \
#     fi

# This installs llvm toolchain

# Instal cuda if version is specified
ARG CUDA_VERSION
RUN if [ -n "$CUDA_VERSION" ]; then \
       conda install cuda -c "nvidia/label/cuda-${CUDA_VERSION}"; \
    fi
# [Optional] Uncomment this section to install additional OS packages.
# RUN apt-get update && export DEBIAN_FRONTEND=noninteractive \
#     && apt-get -y install --no-install-recommends <your-package-list-here>
