# Cross-compilation Docker container for RISC-V architecture
ARG UBUNTU_VERSION
FROM --platform=linux/amd64 ubuntu:${UBUNTU_VERSION} as base

ARG UBUNTU_VERSION

ENV GCC_VERSION=14
ENV PYTHON_VERSION=3.12.3
ENV DEBIAN_FRONTEND=noninteractive
ENV CC=riscv64-linux-gnu-gcc-${GCC_VERSION}
ENV CXX=riscv64-linux-gnu-g++-${GCC_VERSION}
ENV QEMU_LD_PREFIX=/usr/riscv64-linux-gnu/
ENV SYSROOT=/opt/sysroot

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
    gcc-${GCC_VERSION}-riscv64-linux-gnu \
    g++-${GCC_VERSION}-riscv64-linux-gnu \
    pkg-config \
    && rm -rf /var/lib/apt/lists/*

# Install user
COPY ./common/install_user.sh install_user.sh
RUN bash ./install_user.sh && rm install_user.sh

FROM base as python
ARG ZLIB_VERSION=1.3.2
ARG FFI_VERSION=3.4.6
ARG BZ2_VERSION=1.0.8
ARG XZ_VERSION=5.4.6
ARG OPENSSL_VERSION=3.2.1

# Set up sysroot directory for dependencies
ENV PKG_CONFIG_PATH=${SYSROOT}/lib/pkgconfig
ENV PKG_CONFIG_SYSROOT_DIR=${SYSROOT}

WORKDIR /opt

# Build zlib (for compression)
RUN echo "--- Building zlib ---" \
    && wget -c https://www.zlib.net/zlib-${ZLIB_VERSION}.tar.gz \
    && tar -xf zlib-${ZLIB_VERSION}.tar.gz --no-same-permissions --no-same-owner \
    && cd zlib-${ZLIB_VERSION}/ \
    && mkdir build && cd build \
    && ../configure --prefix=${SYSROOT} \
    && make -j$(nproc) && make install \
    && cd ../..

# Build libffi (for ctypes module)
RUN echo "--- Building libffi ---" \
    && wget -c https://github.com/libffi/libffi/releases/download/v${FFI_VERSION}/libffi-${FFI_VERSION}.tar.gz \
    && tar -xf libffi-${FFI_VERSION}.tar.gz --no-same-permissions --no-same-owner \
    && cd libffi-${FFI_VERSION}/ \
    && mkdir build && cd build \
    && ../configure --prefix=${SYSROOT} --host=riscv64-linux-gnu --build=x86_64-linux-gnu \
    && make -j$(nproc) && make install \
    && cd ../..

# Build bzip2 (for bz2 module)
RUN echo "--- Building bzip2 ---" \
    && wget -c https://sourceware.org/pub/bzip2/bzip2-${BZ2_VERSION}.tar.gz \
    && tar -xf bzip2-${BZ2_VERSION}.tar.gz --no-same-permissions --no-same-owner \
    && cd bzip2-${BZ2_VERSION}/ \
    && make CC=riscv64-linux-gnu-gcc-${GCC_VERSION} bzip2 bzip2recover libbz2.a \
    && make CC=riscv64-linux-gnu-gcc-${GCC_VERSION} -f Makefile-libbz2_so \
    && make install PREFIX=${SYSROOT} \
    && cp libbz2.so.${BZ2_VERSION} ${SYSROOT}/lib/ \
    && cd ${SYSROOT}/lib/ \
    && ln -sf libbz2.so.${BZ2_VERSION} libbz2.so.1.0 \
    && ln -sf libbz2.so.1.0 libbz2.so \
    && cd /opt/

# Build xz (for lzma module)
RUN echo "--- Building xz ---" \
    && wget -c https://github.com/tukaani-project/xz/releases/download/v${XZ_VERSION}/xz-${XZ_VERSION}.tar.gz \
    && tar -xf xz-${XZ_VERSION}.tar.gz --no-same-permissions --no-same-owner \
    && cd xz-${XZ_VERSION} \
    && mkdir build && cd build \
    && ../configure --prefix=${SYSROOT} --host=riscv64-linux-gnu --build=x86_64-linux-gnu \
    && make -j$(nproc) && make install \
    && cd ../..

# Build OpenSSL (for ssl module)
RUN echo "--- Building OpenSSL ---" \
    && wget -c https://www.openssl.org/source/openssl-${OPENSSL_VERSION}.tar.gz \
    && tar -xf openssl-${OPENSSL_VERSION}.tar.gz --no-same-permissions --no-same-owner \
    && cd openssl-${OPENSSL_VERSION}/ \
    && mkdir build && cd build \
    && ../Configure linux64-riscv64 --prefix=${SYSROOT} \
    && make -j$(nproc) && make install_sw \
    && cd ../..

# Build SQLite3 (for sqlite3 module)
RUN echo "--- Building SQLite3 ---" \
    && wget -c https://www.sqlite.org/2024/sqlite-autoconf-3450200.tar.gz \
    && tar -xf sqlite-autoconf-3450200.tar.gz --no-same-permissions --no-same-owner \
    && cd sqlite-autoconf-3450200 \
    && mkdir build && cd build \
    && ../configure --prefix=${SYSROOT} --host=riscv64-linux-gnu --build=x86_64-linux-gnu \
    && make -j$(nproc) && make install \
    && cd ../..

# Build and install RISC-V Python with all modules
RUN wget -c https://www.python.org/ftp/python/${PYTHON_VERSION}/Python-${PYTHON_VERSION}.tgz \
    && tar -xf Python-${PYTHON_VERSION}.tgz --no-same-permissions --no-same-owner \
    && cd Python-${PYTHON_VERSION} \
    && mkdir build && cd build \
    && ../configure \
        --host=riscv64-linux-gnu \
        --build=x86_64-linux-gnu \
        --prefix=${SYSROOT} \
        --enable-shared \
        --disable-ipv6 \
        --with-build-python=/usr/bin/python3 \
        --with-ensurepip=no \
        ac_cv_file__dev_ptmx=yes \
        ac_cv_file__dev_ptc=no \
    && make -j$(nproc) \
    && make install

FROM base as final
COPY --from=python             /opt/sysroot                       /opt/sysroot

# Install crossenv and cmake
RUN pip install crossenv cmake==4.0.0 --break-system-packages \
    && /usr/bin/python3 -m crossenv ${SYSROOT}/bin/python3 /opt/riscv-cross-env

# Add pip-installed cmake binaries to PATH
ENV PATH="/usr/local/bin:${PATH}"

# Set up cross Python environment
SHELL ["/bin/bash", "-c"]
RUN source /opt/riscv-cross-env/bin/activate \
    && pip install setuptools pyyaml typing_extensions wheel

# Set default environment variables for PyTorch build
ENV Python_ROOT_DIR=${SYSROOT}
ENV OPENSSL_ROOT_DIR=${SYSROOT}

USER jenkins
CMD ["bash"]
