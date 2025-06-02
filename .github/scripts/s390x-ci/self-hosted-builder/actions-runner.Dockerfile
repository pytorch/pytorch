# Self-Hosted IBM Z Github Actions Runner.

# Temporary image: amd64 dependencies.
FROM --platform=linux/amd64 docker.io/ubuntu:24.04 as ld-prefix
ENV DEBIAN_FRONTEND=noninteractive
RUN apt-get update && apt-get -y install ca-certificates libicu74 libssl3

# Patched podman
FROM --platform=linux/s390x docker.io/ubuntu:24.04 as podman
ENV DEBIAN_FRONTEND=noninteractive
RUN sed -i 's/^Types: deb$/Types: deb deb-src/' /etc/apt/sources.list.d/ubuntu.sources
RUN apt-get update && \
    apt-get install -y \
        cmake \
        curl \
        devscripts \
        dpkg-dev \
        gdb \
        less \
        make \
        python3 \
        python3-pip \
        quilt \
        rsync \
        software-properties-common \
        stress-ng \
        vim \
        nano \
        wget && \
    apt-get build-dep -y podman && \
    apt-get source podman

COPY podman-patches/podman-25245.patch /tmp/podman-25245.patch
COPY podman-patches/podman-25102-backport.patch /tmp/podman-25102-backport.patch

# import and apply patches
# patches:
# https://github.com/containers/podman/pull/25102
# https://github.com/containers/podman/pull/25245
RUN cd /libpod-* && \
    quilt import /tmp/podman-25245.patch && quilt push && \
    quilt import /tmp/podman-25102-backport.patch && quilt push && \
    dch -i "Fix podman deadlock and add option to clean up build leftovers" && \
    /bin/rm /tmp/podman-25245.patch /tmp/podman-25102-backport.patch

# build patched podman
RUN cd /libpod-* && \
    debuild -i -us -uc -b && \
    /bin/rm /podman-remote_*.deb && \
    mkdir /tmp/podman && cp -v /podman*.deb /tmp/podman

# Main image.
FROM --platform=linux/s390x docker.io/ubuntu:24.04

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
# RUN apt-get update && apt -y install podman podman-docker

# install patched podman
COPY --from=podman /tmp/podman /tmp/podman
RUN apt-get update && apt -y install /tmp/podman/*.deb && /bin/rm -rfv /tmp/podman

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

RUN curl -L https://github.com/actions/runner/releases/download/v2.322.0/actions-runner-linux-x64-2.322.0.tar.gz | tar -xz

ENTRYPOINT ["/usr/bin/entrypoint"]
CMD ["/usr/bin/actions-runner"]

# podman requires additional settings to use docker.io by default
RUN mkdir -pv .config/containers ; echo 'unqualified-search-registries = ["docker.io"]' > .config/containers/registries.conf
