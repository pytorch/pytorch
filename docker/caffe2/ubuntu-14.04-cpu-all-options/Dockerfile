FROM caffe2ai/caffe2:c2v0.8.1.cpu.min.ubuntu14.04
LABEL maintainer="aaronmarkham@fb.com"

# caffe2 install with cpu support

RUN apt-get update && apt-get install -y --no-install-recommends \
    libgflags-dev \
    libgflags2 \
    libgtest-dev \
    libiomp-dev \
    libleveldb-dev \
    liblmdb-dev \
    libopencv-dev \
    libopenmpi-dev \
    libsnappy-dev \
    openmpi-bin \
    openmpi-doc \
    python-numpy \
    python-pydot \
    python-setuptools \
    python-scipy \
    wget \
    && rm -rf /var/lib/apt/lists/*

# tornado 5.0 requires Python 2.7.9+ or 3.4+
RUN pip install --no-cache-dir \
    flask \
    graphviz \
    jupyter \
    matplotlib \
    pydot \
    python-nvd3 \
    pyyaml \
    requests \
    scikit-image \
    scipy \
    setuptools \
    'tornado<5'

########## INSTALLATION STEPS ###################
RUN cd caffe2 && mkdir build && cd build \
    && cmake .. \
    -DUSE_CUDA=OFF \
    -DUSE_NNPACK=OFF \
    -DUSE_ROCKSDB=OFF \
    && make -j"$(nproc)" install \
    && ldconfig \
    && make clean \
    && cd .. \
    && rm -rf build

ENV PYTHONPATH /usr/local
