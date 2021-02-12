# ARG ROCM_VERSION=4.0
# ARG UBUNTU_VERSION=18.04

# FROM rocm/pytorch:rocm${ROCM_VERSION}_ubuntu${UBUNTU_VERSION}_py3.6_pytorch
FROM rocm/pytorch:rocm4.0_ubuntu18.04_py3.6_pytorch

ARG SSH_PORT=22222
ARG OFED_VERSION=5.0-2.1.8.0
ARG UBUNTU_VERSION
ARG DEBIAN_FRONTEND=noninteractive

RUN apt-get update && \
    apt-get install -y --no-install-recommends \
    build-essential \
    jq \
    vim \
    git \
    curl \
    wget \
    lshw \
    dmidecode \
    util-linux \
    automake \
    autoconf \
    libtool \
    perftest \
    net-tools \
    openssh-client \
    openssh-server \
    pciutils \
    libaio-dev \
    libcap2

# Install CMake
RUN wget https://github.com/Kitware/CMake/releases/download/v3.17.1/cmake-3.17.1-Linux-x86_64.sh \
    -q -O /tmp/cmake-install.sh && \
    chmod u+x /tmp/cmake-install.sh && \
    mkdir /usr/local/cmake && \
    /tmp/cmake-install.sh --skip-license --prefix=/usr/local/cmake && \
    rm /tmp/cmake-install.sh

# Install Open MPI
RUN mkdir -p /tmp/openmpi && \
    cd /tmp/openmpi && \
    wget https://www.open-mpi.org/software/ompi/v4.0/downloads/openmpi-4.0.0.tar.gz && \
    tar zxf openmpi-4.0.0.tar.gz && \
    cd openmpi-4.0.0 && \
    ./configure --enable-orterun-prefix-by-default && \
    make -j $(nproc) all && \
    make install && \
    ldconfig && \
    rm -rf /tmp/openmpi

# Configure SSH
RUN mkdir -p /root/.ssh && \
    touch /root/.ssh/authorized_keys && \
    chmod 644 /root/.ssh/authorized_keys && \
    cat /etc/ssh/ssh_host_ed25519_key.pub >> /root/.ssh/authorized_keys && \
    mkdir -p /var/run/sshd && \
    sed -i "s/[# ]*PermitRootLogin prohibit-password/PermitRootLogin yes/" /etc/ssh/sshd_config && \
    sed -i "s/[# ]*Port.*/Port ${SSH_PORT}/" /etc/ssh/sshd_config && \
    echo "PermitUserEnvironment yes" >> /etc/ssh/sshd_config
RUN echo -e "Host node\n\
    HostName 127.0.0.1\n\
    Port ${SSH_PORT}\n\
    IdentityFile /etc/ssh/ssh_host_ed25519_key\n\
    StrictHostKeyChecking no\n"\
    >> /root/.ssh/config

# Install OFED
RUN rm /usr/bin/python && ln -s /usr/bin/python2.7 /usr/bin/python
RUN cd /tmp && \
    wget http://content.mellanox.com/ofed/MLNX_OFED-${OFED_VERSION}/MLNX_OFED_LINUX-${OFED_VERSION}-ubuntu18.04-x86_64.tgz -q && \
    tar -xzvf MLNX_OFED_LINUX-${OFED_VERSION}-ubuntu18.04-x86_64.tgz && \
    PATH=/usr/bin:${PATH} MLNX_OFED_LINUX-${OFED_VERSION}-ubuntu18.04-x86_64/mlnxofedinstall --user-space-only --without-fw-update --force --all && \
    rm -rf MLNX_OFED_LINUX-${OFED_VERSION}*

ENV PATH="/usr/local/cmake/bin:${PATH}" \
    LD_LIBRARY_PATH="/usr/local/lib:${LD_LIBRARY_PATH}" 

RUN rm /usr/bin/python && ln -s /usr/bin/python3.6 /usr/bin/python && \
    rm /usr/bin/python3 && ln -s /usr/bin/python3.6 /usr/bin/python3

RUN pip3 install --upgrade pip && \
    pip3 install \
    psutil \
    pyyaml \
    ConfigParser \
    pandas \
    transformers \
    matplotlib \
    azure-cosmosdb-table \
    azure-storage-blob \
    msrestazure

RUN HOROVOD_GPU_ALLREDUCE=NCCL HOROVOD_GPU_ROCM=1 HOROVOD_GPU=ROCM HOROVOD_ROCM_HOME=/opt/rocm HOROVOD_WITH_PYTORCH=1 \
    pip3 install --no-cache-dir horovod[pytorch]


WORKDIR /root

