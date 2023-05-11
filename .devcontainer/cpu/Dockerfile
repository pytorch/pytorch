FROM mcr.microsoft.com/vscode/devcontainers/miniconda:0-3

# This was needed to make sure cmake work
# RUN python -m pip install --upgrade pip
# RUN pip install pyOpenSSL --upgrade
# RUN /opt/conda/bin/conda update -n base -c defaults conda


#  I am suprised this is needed
RUN conda init

# Copy environment.yml (if found) to a temp location so we update the environment. Also
# copy "noop.txt" so the COPY instruction does not fail if no environment.yml exists.
COPY .devcontainer/cpu/environment.yml .devcontainer/noop.txt /tmp/conda-tmp/
RUN if [ -f "/tmp/conda-tmp/environment.yml" ]; then umask 0002 && /opt/conda/bin/conda env update -n base -f /tmp/conda-tmp/environment.yml; fi \
    && sudo rm -rf /tmp/conda-tmp

# Install the full llvm sweet
# Tools needed for llvm
RUN sudo apt-get -y update
RUN sudo apt install -y lsb-release wget software-properties-common gnupg
# This installs llvm toolchain
# RUN sudo bash -c "$(wget -O - https://apt.llvm.org/llvm.sh)"

# [Optional] Uncomment this section to install additional OS packages.
# RUN apt-get update && export DEBIAN_FRONTEND=noninteractive \
#     && apt-get -y install --no-install-recommends <your-package-list-here>
