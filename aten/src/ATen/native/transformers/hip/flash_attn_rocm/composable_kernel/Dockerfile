FROM ubuntu:20.04

ARG ROCMVERSION=5.3
ARG compiler_version="release"
ARG compiler_commit=""

RUN set -xe

ARG DEB_ROCM_REPO=http://repo.radeon.com/rocm/apt/.apt_$ROCMVERSION/
# Add rocm repository
RUN apt-get update
RUN apt-get install -y wget gnupg
RUN wget -qO - http://repo.radeon.com/rocm/rocm.gpg.key | apt-key add -
RUN sh -c "echo deb [arch=amd64] $DEB_ROCM_REPO ubuntu main > /etc/apt/sources.list.d/rocm.list"
RUN wget --no-check-certificate -qO - https://apt.kitware.com/keys/kitware-archive-latest.asc 2>/dev/null | apt-key add -
RUN sh -c "echo deb http://mirrors.kernel.org/ubuntu focal main universe | tee -a /etc/apt/sources.list"

# Install dependencies
RUN apt-get update && DEBIAN_FRONTEND=noninteractive apt-get install -y --allow-unauthenticated \
    apt-utils \
    build-essential \
    ccache \
    cmake-data \
    cmake \
    curl \
    git \
    hip-rocclr \
    jq \
    libelf-dev \
    libncurses5-dev \
    libnuma-dev \
    libpthread-stubs0-dev \
    llvm-amdgpu \
    pkg-config \
    python \
    python3 \
    python-dev \
    python3-dev \
    python3-pip \
    software-properties-common \
    rocm-dev \
    rocm-device-libs \
    rocm-cmake \
    vim \
    zlib1g-dev \
    openssh-server \
    clang-format-10 \
    kmod && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/*

# Setup ubsan environment to printstacktrace
RUN ln -s /usr/bin/llvm-symbolizer-3.8 /usr/local/bin/llvm-symbolizer
ENV UBSAN_OPTIONS=print_stacktrace=1

# Install an init system
RUN wget https://github.com/Yelp/dumb-init/releases/download/v1.2.0/dumb-init_1.2.0_amd64.deb
RUN dpkg -i dumb-init_*.deb && rm dumb-init_*.deb

ARG PREFIX=/opt/rocm
# Install packages for processing the performance results
RUN pip3 install --upgrade pip
RUN pip3 install sqlalchemy
RUN pip3 install pymysql
RUN pip3 install pandas
RUN pip3 install setuptools-rust
RUN pip3 install sshtunnel
# Setup ubsan environment to printstacktrace
ENV UBSAN_OPTIONS=print_stacktrace=1

ENV LC_ALL=C.UTF-8
ENV LANG=C.UTF-8
RUN groupadd -f render

# Install the new rocm-cmake version
RUN git clone -b master https://github.com/RadeonOpenCompute/rocm-cmake.git  && \
  cd rocm-cmake && mkdir build && cd build && \
  cmake  .. && cmake --build . && cmake --build . --target install

WORKDIR /

ENV compiler_version=$compiler_version
ENV compiler_commit=$compiler_commit
RUN sh -c "echo compiler version = '$compiler_version'"
RUN sh -c "echo compiler commit = '$compiler_commit'"

RUN --mount=type=ssh if [ "$compiler_version" = "amd-stg-open" ]; then \
        sed -i '/$HIP_CLANG_TARGET = chomp($HIP_CLANG_TARGET);/c\    chomp($HIP_CLANG_TARGET);' /opt/rocm/hip/bin/hipcc.pl && \
        sed -i '/$HIP_CLANG_TARGET = chomp($HIP_CLANG_TARGET);/c\    chomp($HIP_CLANG_TARGET);' /opt/rocm/bin/hipcc.pl; \
    fi

RUN --mount=type=ssh if [ "$compiler_version" != "release" ] && [ "$compiler_commit" = "" ]; then \
        git clone -b "$compiler_version" https://github.com/RadeonOpenCompute/llvm-project.git && \
        cd llvm-project && mkdir build && cd build && \
        cmake -DCMAKE_INSTALL_PREFIX=/opt/rocm/llvm -DCMAKE_BUILD_TYPE=Release -DLLVM_ENABLE_ASSERTIONS=1 -DLLVM_TARGETS_TO_BUILD="AMDGPU;X86" -DLLVM_ENABLE_PROJECTS="clang;lld;compiler-rt" ../llvm && \
        make -j 8 ; \
    else echo "using the release compiler"; \
    fi

RUN --mount=type=ssh if [ "$compiler_version" != "release" ] && [ "$compiler_commit" != "" ]; then \
        git clone -b "$compiler_version" https://github.com/RadeonOpenCompute/llvm-project.git && \
        cd llvm-project && git checkout "$compiler_commit" && echo "checking out commit $compiler_commit" && mkdir build && cd build && \
        cmake -DCMAKE_INSTALL_PREFIX=/opt/rocm/llvm -DCMAKE_BUILD_TYPE=Release -DLLVM_ENABLE_ASSERTIONS=1 -DLLVM_TARGETS_TO_BUILD="AMDGPU;X86" -DLLVM_ENABLE_PROJECTS="clang;lld;compiler-rt" ../llvm && \
        make -j 8 ; \
    else echo "using the release compiler"; \
    fi


#ENV HIP_CLANG_PATH='/llvm-project/build/bin'
#RUN sh -c "echo HIP_CLANG_PATH = '$HIP_CLANG_PATH'"
