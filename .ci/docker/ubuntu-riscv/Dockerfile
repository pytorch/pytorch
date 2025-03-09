ARG UBUNTU_VERSION
FROM --platform=linux/amd64 ubuntu:${UBUNTU_VERSION} as base

ARG UBUNTU_VERSION
ENV PYTHON_VERSION=3.12.3
ARG RISCV_GCC_VERSION=14

ENV DEBIAN_FRONTEND=noninteractive
ENV CC=riscv64-linux-gnu-gcc-${RISCV_GCC_VERSION}
ENV CXX=riscv64-linux-gnu-g++-${RISCV_GCC_VERSION}
ENV QEMU_LD_PREFIX=/usr/riscv64-linux-gnu/

# Install basic dependencies
RUN apt-get update && apt-get install -y \
    ninja-build \
    autoconf \
    automake \
    libtool \
    patchelf \
    ccache \
    git \
    wget \
    python3-pip \
    python3-venv \
    python-is-python3 \
    cmake \
    sudo \
    lsb-release \
    gcc-${RISCV_GCC_VERSION}-riscv64-linux-gnu \
    g++-${RISCV_GCC_VERSION}-riscv64-linux-gnu \
    && rm -rf /var/lib/apt/lists/*

# Install user
COPY ./common/install_user.sh install_user.sh
RUN bash ./install_user.sh && rm install_user.sh

FROM base as python
# Build and install RISC-V Python
WORKDIR /opt
RUN wget https://www.python.org/ftp/python/${PYTHON_VERSION}/Python-${PYTHON_VERSION}.tgz \
    && tar -xf Python-${PYTHON_VERSION}.tgz --no-same-permissions --no-same-owner \
    && cd Python-${PYTHON_VERSION} \
    && mkdir build && cd build \
    && ../configure \
        --host=riscv64-linux-gnu \
        --build=x86_64-linux-gnu \
        --prefix=/opt/python-riscv \
        --enable-shared \
        --disable-ipv6 \
        --enable-optimizations \
        ac_cv_file__dev_ptmx=no \
        ac_cv_file__dev_ptc=no \
        --with-build-python=/usr/bin/python3 \
    && make -j$(nproc) \
    && make install

FROM base as final
COPY --from=python             /opt/python-riscv                           /opt/python-riscv

# Install crossenv
RUN pip install crossenv --break-system-packages \
    && /usr/bin/python3 -m crossenv /opt/python-riscv/bin/python3 /opt/riscv-cross-env

# Set up cross Python environment
SHELL ["/bin/bash", "-c"]
RUN source /opt/riscv-cross-env/bin/activate \
    && pip install setuptools pyyaml typing_extensions wheel \
    && python -mpip install --pre numpy==2.0.2

# Set default environment variables for PyTorch build
ENV Python_ROOT_DIR=/opt/python-riscv/

USER jenkins
CMD ["bash"]