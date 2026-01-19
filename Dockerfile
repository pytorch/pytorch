# syntax=docker/dockerfile:1

# NOTE: Building this image require's docker version >= 23.0.
#
# For reference:
# - https://docs.docker.com/build/dockerfile/frontend/#stable-channel

ARG BASE_IMAGE=ubuntu:24.04

FROM ${BASE_IMAGE} as dev-base
RUN apt-get update && DEBIAN_FRONTEND=noninteractive apt-get install -y --no-install-recommends \
        build-essential \
        ca-certificates \
        ccache \
        cmake \
        curl \
        git \
        libjpeg-dev \
        libpng-dev \
        python3 \
        python3-pip \
        python-is-python3 \
        python3-dev && \
    rm -rf /var/lib/apt/lists/*
# Remove PEP 668 restriction (safe in containers)
RUN rm -f /usr/lib/python*/EXTERNALLY-MANAGED
RUN /usr/sbin/update-ccache-symlinks
RUN mkdir /opt/ccache && ccache --set-config=cache_dir=/opt/ccache

FROM dev-base as python-deps
COPY requirements.txt requirements-build.txt .
# Install Python packages to system Python
RUN pip3 install --upgrade --ignore-installed pip setuptools wheel && \
    pip3 install cmake pyyaml numpy ipython -r requirements.txt

FROM dev-base as submodule-update
WORKDIR /opt/pytorch
COPY . .
RUN git submodule update --init --recursive

FROM python-deps as pytorch-installs
ARG CUDA_PATH=cu121
ARG INSTALL_CHANNEL=whl/nightly
# Automatically set by buildx
ARG TARGETPLATFORM

# INSTALL_CHANNEL whl - release, whl/nightly - nightly, whl/test - test channels
RUN case ${TARGETPLATFORM} in \
         "linux/arm64")  pip3 install --extra-index-url https://download.pytorch.org/whl/cpu/ torch torchvision torchaudio ;; \
         *)              pip3 install --index-url https://download.pytorch.org/${INSTALL_CHANNEL}/${CUDA_PATH#.}/ torch torchvision torchaudio ;; \
    esac
RUN pip3 install torchelastic
RUN IS_CUDA=$(python3 -c 'import torch ; print(torch.cuda._is_compiled())'); \
    echo "Is torch compiled with cuda: ${IS_CUDA}"; \
    if test "${IS_CUDA}" != "True" -a ! -z "${CUDA_VERSION}"; then \
        exit 1; \
    fi

FROM ${BASE_IMAGE} as official
ARG PYTORCH_VERSION
ARG TRITON_VERSION
ARG TARGETPLATFORM
ARG CUDA_VERSION
LABEL com.nvidia.volumes.needed="nvidia_driver"
RUN apt-get update && DEBIAN_FRONTEND=noninteractive apt-get install -y --no-install-recommends \
        ca-certificates \
        libjpeg-dev \
        libpng-dev \
        python-is-python3 \
        python3 \
        python3-dev \
        python3-pip \
        && rm -rf /var/lib/apt/lists/*
# Copy Python packages from pytorch-installs stage
COPY --from=pytorch-installs /usr/local/lib/python3.12 /usr/local/lib/python3.12
COPY --from=pytorch-installs /usr/local/bin /usr/local/bin
RUN if test -n "${CUDA_VERSION}" -a "${TARGETPLATFORM}" != "linux/arm64"; then \
        apt-get update -qq && \
        DEBIAN_FRONTEND=noninteractive apt-get install -y --no-install-recommends gcc && \
        rm -rf /var/lib/apt/lists/*; \
    fi
ENV NVIDIA_VISIBLE_DEVICES all
ENV NVIDIA_DRIVER_CAPABILITIES compute,utility
ENV LD_LIBRARY_PATH /usr/local/nvidia/lib:/usr/local/nvidia/lib64
ENV PATH /usr/local/nvidia/bin:/usr/local/cuda/bin:$PATH
ENV PYTORCH_VERSION ${PYTORCH_VERSION}
WORKDIR /workspace

FROM official as dev
# Should override the already installed version from the official-image stage
COPY --from=python-deps /usr/local/lib/python3.12 /usr/local/lib/python3.12
COPY --from=python-deps /usr/local/bin /usr/local/bin
COPY --from=submodule-update /opt/pytorch /opt/pytorch
