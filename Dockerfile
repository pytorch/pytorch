# syntax=docker/dockerfile:1

# NOTE: Building this image require's docker version >= 23.0.
#
# For reference:
# - https://docs.docker.com/build/dockerfile/frontend/#stable-channel

ARG BASE_IMAGE=ubuntu:22.04
ARG PYTHON_VERSION=3.11

FROM ${BASE_IMAGE} as dev-base
RUN apt-get update && DEBIAN_FRONTEND=noninteractive apt-get install -y --no-install-recommends \
        build-essential \
        ca-certificates \
        ccache \
        cmake \
        curl \
        git \
        libjpeg-dev \
        libpng-dev && \
    rm -rf /var/lib/apt/lists/*
RUN /usr/sbin/update-ccache-symlinks
RUN mkdir /opt/ccache && ccache --set-config=cache_dir=/opt/ccache
ENV PATH /opt/uv/bin:$PATH

FROM dev-base as uv
ARG PYTHON_VERSION=3.11
# Install uv
RUN curl -LsSf https://astral.sh/uv/install.sh | sh && \
    mv /root/.local/bin/uv /usr/local/bin/uv
COPY requirements.txt requirements-build.txt .
# Create virtual environment and install packages
RUN uv venv /opt/uv --python ${PYTHON_VERSION} && \
    uv pip install --python /opt/uv/bin/python cmake pyyaml numpy ipython -r requirements.txt

FROM dev-base as submodule-update
WORKDIR /opt/pytorch
COPY . .
RUN git submodule update --init --recursive

FROM uv as uv-installs
ARG PYTHON_VERSION=3.11
ARG CUDA_PATH=cu121
ARG INSTALL_CHANNEL=whl/nightly
# Automatically set by buildx
ARG TARGETPLATFORM

# INSTALL_CHANNEL whl - release, whl/nightly - nightly, whl/test - test channels
RUN case ${TARGETPLATFORM} in \
         "linux/arm64")  uv pip install --python /opt/uv/bin/python --extra-index-url https://download.pytorch.org/whl/cpu/ torch torchvision torchaudio ;; \
         *)              uv pip install --python /opt/uv/bin/python --index-url https://download.pytorch.org/${INSTALL_CHANNEL}/${CUDA_PATH#.}/ torch torchvision torchaudio ;; \
    esac
RUN /opt/uv/bin/pip install torchelastic
RUN IS_CUDA=$(python -c 'import torch ; print(torch.cuda._is_compiled())'); \
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
        && rm -rf /var/lib/apt/lists/*
COPY --from=uv-installs /opt/uv /opt/uv
RUN if test -n "${TRITON_VERSION}" -a "${TARGETPLATFORM}" != "linux/arm64"; then \
        DEBIAN_FRONTEND=noninteractive apt install -y --no-install-recommends gcc; \
        rm -rf /var/lib/apt/lists/*; \
    fi
ENV PATH /opt/uv/bin:$PATH
ENV NVIDIA_VISIBLE_DEVICES all
ENV NVIDIA_DRIVER_CAPABILITIES compute,utility
ENV LD_LIBRARY_PATH /usr/local/nvidia/lib:/usr/local/nvidia/lib64
ENV PATH /usr/local/nvidia/bin:/usr/local/cuda/bin:$PATH
ENV PYTORCH_VERSION ${PYTORCH_VERSION}
WORKDIR /workspace

FROM official as dev
# Should override the already installed version from the official-image stage
COPY --from=uv /opt/uv /opt/uv
COPY --from=submodule-update /opt/pytorch /opt/pytorch
