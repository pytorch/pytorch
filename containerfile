# TO: localhost/pytorch:remoting

FROM quay.io/centos/centos:10
USER 0

# ---

RUN dnf install -y git cmake  gcc gcc-c++

# --- PyTorch. --- #

ARG PYTORCH_VERSION=remoting
ARG PYTORCH_URL=https://github.com/kpouget/pytorch/

WORKDIR /app/pytorch

RUN dnf --enablerepo=crb install -y python3-pybind11
RUN dnf install -y python3-devel

RUN git clone $PYTORCH_URL src --branch ${PYTORCH_VERSION} --single-branch \
 && cd src \
 && git submodule sync \
 && git submodule update --init --recursive

RUN pip3 install -r src/requirements.txt

RUN cd src \
    && cmake --version \
    && export USE_CUDA=OFF USE_XPU=OFF CMAKE_POLICY_VERSION_MINIMUM=3.5 \
    && bash -x ./rebuild.sh || true

USER 1001

WORKDIR /app
