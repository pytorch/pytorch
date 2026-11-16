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

# The torch packages and the index they come from, in one place so that the
# stage that resolves them and the stage that installs them cannot disagree.
# Sourced by `sh`; expects TARGETPLATFORM, INSTALL_CHANNEL and CUDA_PATH.
FROM scratch as torch-pip-args
COPY <<'EOF' /torch_pip_args.sh
case "${TARGETPLATFORM}" in
    "linux/arm64") TORCH_INDEX_ARGS="--extra-index-url https://download.pytorch.org/whl/cpu/" ;;
    *)             TORCH_INDEX_ARGS="--index-url https://download.pytorch.org/${INSTALL_CHANNEL}/${CUDA_PATH}/" ;;
esac
TORCH_PACKAGES="torch torchvision torchaudio"
# torchaudio does not publish wheels against the CUDA 13.2 index yet,
# so skip it when CUDA_PATH=cu132 to keep this Dockerfile building.
if [ "${TARGETPLATFORM}" != "linux/arm64" ] && [ "${CUDA_PATH}" = "cu132" ]; then
    TORCH_PACKAGES="torch torchvision"
fi
EOF

# Work out exactly which versions `pip install torch torchvision torchaudio`
# would install, without downloading any wheels (pip reads the small PEP 658
# .metadata files). The result is split in three lock files, by how often the
# contents change:
#   /deps-pinned.lock    packages whose version the torch, torchvision or
#                        torchaudio wheel (or a pinned package, transitively)
#                        fixes with `==`: nvidia-*, cuda-toolkit, triton, ... These
#                        are the large wheels, and they change only when torch
#                        or cuda-toolkit bumps a pin.
#   /deps-floating.lock  everything else they depend on (setuptools, fsspec,
#                        sympy, numpy, pillow, ...). Their requirements are open
#                        ranges, so pip takes the newest release of each: small,
#                        but they change whenever one of them is released.
#   /torch.lock          torch, torchvision and torchaudio. Different every night.
# The `official` stage installs the three in separate layers, in that order. A
# layer is keyed on the contents of its lock file, so the large pinned layer is
# rebuilt, and clients re-download it, only when a pin changes. A new release of
# a floating package replaces only the small layer after it.
FROM dev-base as resolve-torch-deps
ARG CUDA_PATH=cu121
ARG INSTALL_CHANNEL=whl/nightly
# Automatically set by buildx
ARG TARGETPLATFORM
# Passed a new value on every build so that this cheap step always sees the
# current index and is never served from the build cache.
ARG RESOLVE_CACHE_BUST
RUN --mount=type=bind,from=torch-pip-args,source=/torch_pip_args.sh,target=/torch_pip_args.sh <<'EOF'
set -eu
. /torch_pip_args.sh
# Resolve with the same pip the `official` stage installs with. Ubuntu's own pip
# (24.0) does not resolve the current nightly wheels: on the cu130 and cu132
# indexes it backtracks to torch 2.15.0.dev20260928 instead of the newest build.
pip3 install --upgrade --ignore-installed pip
echo "Resolving ${TORCH_PACKAGES} with pip $(pip3 --version | cut -d' ' -f2) from ${TORCH_INDEX_ARGS} (${RESOLVE_CACHE_BUST:-})"
# shellcheck disable=SC2086
pip3 install --dry-run --ignore-installed --quiet --report /tmp/report.json ${TORCH_INDEX_ARGS} ${TORCH_PACKAGES}
python3 - <<'PY'
import json
import re

TORCH = {"torch", "torchvision", "torchaudio"}
# One requirement line of a wheel's METADATA: name, extras, then either a
# parenthesised or a bare specifier, then an optional environment marker.
REQUIREMENT = re.compile(
    r"\s*([A-Za-z0-9][A-Za-z0-9._-]*)\s*(?:\[[^\]]*\])?\s*"
    r"(?:\(([^)]*)\)|([<>=!~][^;]*))?\s*(?:;.*)?$"
)
# `==1.2.3`, or `==1.2.*` as cuda-toolkit pins the nvidia-* wheels: either one fixes
# the version, give or take a patch release.
EXACT = re.compile(r"==\s*[^\s,]+")


def normalize(name):
    return re.sub(r"[-_.]+", "-", name).lower()


def exact_pins(requires_dist):
    """Names that a wheel requires with a single `==` specifier."""
    for requirement in requires_dist or ():
        match = REQUIREMENT.match(requirement)
        if match and EXACT.fullmatch((match[2] or match[3] or "").strip()):
            yield normalize(match[1])


metadata = {}
for item in json.load(open("/tmp/report.json"))["install"]:
    metadata[normalize(item["metadata"]["name"])] = item["metadata"]

# A package is pinned if the torch packages, or a package that is itself pinned,
# require it with `==`. Only packages that are being installed count, which
# also drops requirements behind an extra or marker that does not apply.
pinned, todo = set(), [name for name in metadata if name in TORCH]
while todo:
    for dep in exact_pins(metadata[todo.pop()].get("requires_dist")):
        if dep in metadata and dep not in TORCH and dep not in pinned:
            pinned.add(dep)
            todo.append(dep)

locks = {"/torch.lock": [], "/deps-pinned.lock": [], "/deps-floating.lock": []}
for name, meta in metadata.items():
    if name in TORCH:
        path = "/torch.lock"
    elif name in pinned:
        path = "/deps-pinned.lock"
    else:
        path = "/deps-floating.lock"
    locks[path].append(f"{meta['name']}=={meta['version']}")
# Sorted, so that equal sets of packages always give a byte-identical file.
for path, pins in locks.items():
    with open(path, "w") as f:
        f.write("".join(f"{pin}\n" for pin in sorted(pins, key=str.lower)))
PY
cat /torch.lock /deps-pinned.lock /deps-floating.lock
EOF

FROM ${BASE_IMAGE} as official
ARG TARGETPLATFORM
ARG CUDA_VERSION
ARG CUDA_PATH=cu121
ARG INSTALL_CHANNEL=whl/nightly
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
RUN if test -n "${CUDA_VERSION}" -a "${TARGETPLATFORM}" != "linux/arm64"; then \
        apt-get update -qq && \
        DEBIAN_FRONTEND=noninteractive apt-get install -y --no-install-recommends gcc && \
        rm -rf /var/lib/apt/lists/*; \
    fi

# Install CUDA toolkit for devel images
# Before the Python layers on purpose: a layer is rebuilt whenever one before it
# changes, and the torch layers change every night.
ARG BUILD_TYPE
RUN if [ "${BUILD_TYPE}" = "dev" ] && [ -n "${CUDA_VERSION}" ]; then \
    apt-get update && apt-get install -y --no-install-recommends \
        wget gnupg2 ca-certificates && \
    # Add NVIDIA repository
    NVARCH=$(uname -m | sed 's/x86_64/x86_64/' | sed 's/aarch64/sbsa/') && \
    wget -qO - https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2204/${NVARCH}/3bf863cc.pub | apt-key add - && \
    echo "deb https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2204/${NVARCH} /" > /etc/apt/sources.list.d/cuda.list && \
    # Install CUDA toolkit
    CUDA_PKG_VERSION=$(echo ${CUDA_VERSION} | cut -d'.' -f1,2 | tr '.' '-') && \
    apt-get update && apt-get install -y --no-install-recommends \
        cuda-toolkit-${CUDA_PKG_VERSION} && \
    apt-get clean && rm -rf /var/lib/apt/lists/* && \
    # Configure LD
    echo "/usr/local/cuda/lib64" >> /etc/ld.so.conf.d/cuda.conf && \
    ldconfig; \
fi

# The Python environment is installed here, directly into the image, in layers
# ordered from the one that changes least to the one that changes every night.
# Installing in a build stage and copying the result in, as this file used to,
# fuses everything into a single multi-GB layer that is replaced nightly: with
# the cu130 nightly, 3.4 GB of a 3.5 GB image was re-downloaded every day, of
# which only ~0.7 GB is the torch wheels themselves.
#
# 1. The packages `python-deps` installs (same commands, so the image contents
#    are unchanged). Changes when requirements*.txt does.
RUN --mount=type=bind,source=requirements.txt,target=/tmp/req/requirements.txt \
    --mount=type=bind,source=requirements-build.txt,target=/tmp/req/requirements-build.txt \
    pip3 install --break-system-packages --no-cache-dir --upgrade --ignore-installed pip setuptools wheel && \
    pip3 install --break-system-packages --no-cache-dir cmake pyyaml numpy ipython -r /tmp/req/requirements.txt
# 2. The large packages torch pins exactly (nvidia-*, cuda-toolkit, triton, ...):
#    see `resolve-torch-deps`. Changes when torch bumps a pin, not every nightly.
#    The lock lists the whole closure, so nothing is resolved here.
RUN --mount=type=bind,from=torch-pip-args,source=/torch_pip_args.sh,target=/torch_pip_args.sh \
    --mount=type=bind,from=resolve-torch-deps,source=/deps-pinned.lock,target=/tmp/deps-pinned.lock \
    . /torch_pip_args.sh && \
    pip3 install --break-system-packages --no-cache-dir --no-deps ${TORCH_INDEX_ARGS} -r /tmp/deps-pinned.lock
# 3. The small packages with open version ranges (setuptools, fsspec, sympy,
#    numpy, pillow, ...). Changes whenever one of them has a new release; after
#    layer 2 so that it does not invalidate it.
RUN --mount=type=bind,from=torch-pip-args,source=/torch_pip_args.sh,target=/torch_pip_args.sh \
    --mount=type=bind,from=resolve-torch-deps,source=/deps-floating.lock,target=/tmp/deps-floating.lock \
    . /torch_pip_args.sh && \
    pip3 install --break-system-packages --no-cache-dir --no-deps ${TORCH_INDEX_ARGS} -r /tmp/deps-floating.lock
# 4. torch, torchvision and torchaudio, at the exact versions that were resolved
#    together with the dependencies above. This is the nightly layer.
RUN --mount=type=bind,from=torch-pip-args,source=/torch_pip_args.sh,target=/torch_pip_args.sh \
    --mount=type=bind,from=resolve-torch-deps,source=/torch.lock,target=/tmp/torch.lock \
    . /torch_pip_args.sh && \
    pip3 install --break-system-packages --no-cache-dir --no-deps ${TORCH_INDEX_ARGS} -r /tmp/torch.lock
RUN pip3 install --break-system-packages --no-cache-dir torchelastic
RUN IS_CUDA=$(python3 -c 'import torch ; print(torch.cuda._is_compiled())'); \
    echo "Is torch compiled with cuda: ${IS_CUDA}"; \
    if test "${IS_CUDA}" != "True" -a ! -z "${CUDA_VERSION}"; then \
        exit 1; \
    fi

# Declared only now. A build arg that is in scope for a RUN can invalidate its
# cache, and these change on every nightly build.
ARG PYTORCH_VERSION
ARG TRITON_VERSION
ENV NVIDIA_VISIBLE_DEVICES all
ENV NVIDIA_DRIVER_CAPABILITIES compute,utility
ENV LD_LIBRARY_PATH /usr/local/nvidia/lib:/usr/local/nvidia/lib64
ENV PATH /usr/local/nvidia/bin:/usr/local/cuda/bin:$PATH
ENV PYTORCH_VERSION ${PYTORCH_VERSION}
WORKDIR /workspace

FROM official as dev
# Set CUDA environment (always set, needed even if CUDA already in base)
ENV PATH=/usr/local/cuda/bin:${PATH}
ENV LD_LIBRARY_PATH=/usr/local/cuda/lib64:${LD_LIBRARY_PATH}
ENV CUDA_HOME=/usr/local/cuda

# Should override the already installed version from the official-image stage
COPY --from=python-deps /usr/local/lib/python3.12 /usr/local/lib/python3.12
COPY --from=python-deps /usr/local/bin /usr/local/bin
COPY --from=submodule-update /opt/pytorch /opt/pytorch
