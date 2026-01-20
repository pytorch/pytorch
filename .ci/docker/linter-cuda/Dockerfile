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
ARG PYTHON_VERSION
ARG PIP_CMAKE
# Put venv into the env vars so users don't need to activate it
ENV PATH /var/lib/jenkins/ci_env/bin:$PATH
ENV VIRTUAL_ENV /var/lib/jenkins/ci_env
COPY requirements-ci.txt /opt/requirements-ci.txt
COPY ./common/install_python.sh install_python.sh
RUN bash ./install_python.sh && rm install_python.sh /opt/requirements-ci.txt

# Install cuda and cudnn
ARG CUDA_VERSION
COPY ./common/install_cuda.sh install_cuda.sh
COPY ./common/install_nccl.sh install_nccl.sh
COPY ./ci_commit_pins/nccl* /ci_commit_pins/
COPY ./common/install_cusparselt.sh install_cusparselt.sh
RUN bash ./install_cuda.sh ${CUDA_VERSION} && rm install_cuda.sh install_nccl.sh /ci_commit_pins/nccl* install_cusparselt.sh
ENV DESIRED_CUDA ${CUDA_VERSION}
ENV PATH /usr/local/nvidia/bin:/usr/local/cuda/bin:$PATH

# Note that Docker build forbids copying file outside the build context
COPY ./common/install_linter.sh install_linter.sh
RUN bash ./install_linter.sh
RUN rm install_linter.sh

RUN chown -R jenkins:jenkins /var/lib/jenkins/ci_env

USER jenkins
CMD ["bash"]
