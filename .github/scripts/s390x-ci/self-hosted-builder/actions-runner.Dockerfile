# Self-Hosted IBM Z Github Actions Runner.

# Temporary image: amd64 dependencies.
FROM docker.io/amd64/ubuntu:23.10 as ld-prefix
ENV DEBIAN_FRONTEND=noninteractive
RUN apt-get update && apt-get -y install ca-certificates libicu72 libssl3

# Main image.
FROM docker.io/s390x/ubuntu:23.10

# Packages for pytorch building and testing.
ENV DEBIAN_FRONTEND=noninteractive
RUN apt-get update && apt-get -y install \
        cmake \
        curl \
        gcc \
        git \
        jq \
        zip \
        libxml2-dev \
        libxslt-dev \
        ninja-build \
        python-is-python3 \
        python3 \
        python3-dev \
        python3-pip \
        pybind11-dev \
        python3-numpy \
        libopenblas-dev \
        liblapack-dev \
        libgloo-dev \
        python3-yaml \
        python3-scipy \
        virtualenv

# amd64 dependencies.
COPY --from=ld-prefix / /usr/x86_64-linux-gnu/
RUN ln -fs ../lib/x86_64-linux-gnu/ld-linux-x86-64.so.2 /usr/x86_64-linux-gnu/lib64/
RUN ln -fs /etc/resolv.conf /usr/x86_64-linux-gnu/etc/
ENV QEMU_LD_PREFIX=/usr/x86_64-linux-gnu

# Scripts.
COPY fs/ /

RUN chmod +x /usr/bin/actions-runner /usr/bin/entrypoint

# install podman
RUN apt -y install podman podman-docker

# amd64 Github Actions Runner.
RUN useradd -m actions-runner
USER actions-runner
WORKDIR /home/actions-runner

# set up python virtual environment which is later used by runner.
# build workflows use "python -m pip install ...",
# and it doesn't work for non-root user
RUN virtualenv --system-site-packages venv

# copy prebuilt manywheel docker image for builds and tests
# build command is:
# GPU_ARCH_TYPE=cpu-s390x "$(pwd)/manywheel/build_docker.sh"
# and save command is:
# docker image save -o manywheel-s390x.tar pytorch/manylinuxs390x-builder:cpu-s390x
#
COPY --chown=actions-runner:actions-runner manywheel-s390x.tar /home/actions-runner/manywheel-s390x.tar

RUN curl -L https://github.com/actions/runner/releases/download/v2.317.0/actions-runner-linux-x64-2.317.0.tar.gz | tar -xz

ENTRYPOINT ["/usr/bin/entrypoint"]
CMD ["/usr/bin/actions-runner"]
