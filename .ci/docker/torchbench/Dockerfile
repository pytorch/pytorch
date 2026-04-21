# Standalone multiarch image that pre-installs TorchBench, HuggingFace, and timm
# dependencies into /opt/conda/envs/py_3.10 so they can be COPY'd into CI images.
#
# Builds for both x86_64 and aarch64.

FROM ubuntu:22.04

ENV DEBIAN_FRONTEND=noninteractive

RUN apt-get update && apt-get install -y --no-install-recommends \
    ca-certificates \
    curl \
    git \
    sudo \
    && rm -rf /var/lib/apt/lists/*

# Create jenkins user (matches CI images)
RUN groupadd -g 1000 jenkins && \
    useradd -m -u 1000 -g jenkins jenkins && \
    echo "jenkins ALL=(ALL) NOPASSWD:ALL" >> /etc/sudoers

# Install miniforge as jenkins (conda-forge by default, matches CI base images)
ARG PYTHON_VERSION=3.10
ENV ANACONDA_PYTHON_VERSION=${PYTHON_VERSION}
RUN mkdir -p /opt/conda && chown jenkins:jenkins /opt/conda
USER jenkins
RUN ARCH=$(uname -m) && \
    curl -fsSL -o /tmp/miniforge.sh \
      "https://github.com/conda-forge/miniforge/releases/latest/download/Miniforge3-Linux-${ARCH}.sh" && \
    bash /tmp/miniforge.sh -b -f -p /opt/conda && \
    rm /tmp/miniforge.sh && \
    /opt/conda/bin/conda create -y -n py_${PYTHON_VERSION} python=${PYTHON_VERSION} && \
    /opt/conda/bin/conda clean -ya
ENV PATH=/opt/conda/envs/py_${ANACONDA_PYTHON_VERSION}/bin:/opt/conda/bin:$PATH

# Install pango via conda (matches monolithic build)
RUN conda install -c conda-forge -n py_${ANACONDA_PYTHON_VERSION} -y pango && \
    conda clean -ya

# Pin setuptools<82 (visdom's setup.py needs pkg_resources)
RUN pip install --progress-bar off 'setuptools<82'

# Install a temporary CPU-only PyTorch to satisfy TorchBench install checks
RUN pip install --progress-bar off \
    torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu

# Copy pinned commits/requirements
COPY --chown=jenkins:jenkins ci_commit_pins/torchbench.txt /pins/torchbench.txt
COPY --chown=jenkins:jenkins ci_commit_pins/huggingface-requirements.txt /pins/huggingface-requirements.txt
COPY --chown=jenkins:jenkins ci_commit_pins/timm.txt /pins/timm.txt

# Use the CI base image's pins as pip constraints so any shared package
# (scipy, numpy, pandas, ...) lands on the exact same version as the base
# image. This keeps the later `COPY --from=torchbench /opt/conda/envs/...`
# from producing a mixed install where .py files and compiled .so files
# come from different versions of the same package.
COPY --chown=jenkins:jenkins requirements-ci.txt /pins/requirements-ci.txt
ENV PIP_CONSTRAINT=/pins/requirements-ci.txt

# Install TorchBench (into / to match monolithic build's CWD)
WORKDIR /torchbench
RUN TORCHBENCH_COMMIT=$(cat /pins/torchbench.txt) && \
    git clone https://github.com/pytorch/benchmark . && \
    git checkout "${TORCHBENCH_COMMIT}" && \
    python install.py --continue_on_fail

# Install HuggingFace deps
RUN pip install --progress-bar off -r /pins/huggingface-requirements.txt

# Install timm
RUN TIMM_COMMIT=$(cat /pins/timm.txt) && \
    pip install --progress-bar off \
    "git+https://github.com/huggingface/pytorch-image-models@${TIMM_COMMIT}"

# Uninstall temporary PyTorch (the real one gets installed in CI)
RUN pip uninstall -y torch torchvision torchaudio triton torchao || true
CMD ["bash"]
