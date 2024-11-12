FROM --platform=linux/riscv64 docker.io/ubuntu:24.04 as base

ENV DEBIAN_FRONTEND noninteractive

RUN apt update -y
RUN apt upgrade -y
RUN apt install -y \
  sudo \
  gcc-14 \
  g++-14 \
  make \
  libc6-dev \
  autoconf \
  curl \
  ccache \
  cmake \
  git \
  ninja-build \
  libomp-dev \
  libffi-dev \
  libssl-dev \
  zlib1g-dev \
  wget \
  libblas-dev \
  libopenblas-dev \
  liblapack-dev \
  libatlas-base-dev \
  zstd \
  perl \
  python3 \
  python3-dev \
  python3-pip \
  python3-venv \
  python-is-python3

RUN update-alternatives --install /usr/bin/gcc gcc /usr/bin/gcc-14 20
RUN update-alternatives --install /usr/bin/g++ g++ /usr/bin/g++-14 20

ENV VENV_PATH=/opt/pyvenv
RUN python3 -m venv $VENV_PATH
ENV PATH="$VENV_PATH/bin:${PATH}"
RUN pip install --upgrade pip
RUN pip install pyyaml
RUN pip install setuptools
RUN pip install typing_extensions

WORKDIR /home/ubuntu
RUN git clone https://github.com/pytorch/pytorch.git

WORKDIR /home/ubuntu/pytorch
RUN git submodule sync
RUN git submodule update --init --recursive

RUN python setup.py bdist_wheel
RUN pip install "$(echo dist/*.whl)"

WORKDIR /home/ubuntu
COPY .github/scripts/riscv/entrypoint.sh /usr/local/bin/entrypoint.sh
RUN chmod +x /usr/local/bin/entrypoint.sh

ENTRYPOINT [ "/usr/local/bin/entrypoint.sh" ]
CMD [ "/bin/bash" ]
