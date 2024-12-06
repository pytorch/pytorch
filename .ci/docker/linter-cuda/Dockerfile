ARG UBUNTU_VERSION

FROM ubuntu:${UBUNTU_VERSION}

ARG UBUNTU_VERSION

ENV DEBIAN_FRONTEND noninteractive

# Install common dependencies (so that this step can be cached separately)
COPY ./common/install_base.sh install_base.sh
RUN bash ./install_base.sh && rm install_base.sh

# Install missing libomp-dev
RUN apt-get update && apt-get install -y --no-install-recommends libomp-dev && apt-get autoclean && apt-get clean && rm -rf /var/lib/apt/lists/* /tmp/* /var/tmp/*

# Install user
COPY ./common/install_user.sh install_user.sh
RUN bash ./install_user.sh && rm install_user.sh

# Install conda and other packages (e.g., numpy, pytest)
ARG ANACONDA_PYTHON_VERSION
ARG CONDA_CMAKE
ENV ANACONDA_PYTHON_VERSION=$ANACONDA_PYTHON_VERSION
ENV PATH /opt/conda/envs/py_$ANACONDA_PYTHON_VERSION/bin:/opt/conda/bin:$PATH
COPY requirements-ci.txt /opt/conda/requirements-ci.txt
COPY ./common/install_conda.sh install_conda.sh
COPY ./common/common_utils.sh common_utils.sh
COPY ./common/install_magma_conda.sh install_magma_conda.sh
RUN bash ./install_conda.sh && rm install_conda.sh install_magma_conda.sh common_utils.sh /opt/conda/requirements-ci.txt

# Install cuda and cudnn
ARG CUDA_VERSION
COPY ./common/install_cuda.sh install_cuda.sh
RUN bash ./install_cuda.sh ${CUDA_VERSION} && rm install_cuda.sh
ENV DESIRED_CUDA ${CUDA_VERSION}
ENV PATH /usr/local/nvidia/bin:/usr/local/cuda/bin:$PATH

# Note that Docker build forbids copying file outside the build context
COPY ./common/install_linter.sh install_linter.sh
COPY ./common/common_utils.sh common_utils.sh
RUN bash ./install_linter.sh
RUN rm install_linter.sh common_utils.sh

USER jenkins
CMD ["bash"]
