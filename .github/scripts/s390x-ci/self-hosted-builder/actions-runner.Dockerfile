# Self-Hosted IBM Z Github Actions Runner.

# Temporary image: amd64 dependencies.
FROM docker.io/amd64/ubuntu:22.04 as ld-prefix
ENV DEBIAN_FRONTEND=noninteractive
RUN apt-get update && apt-get -y install ca-certificates libicu70 libssl3

# Main image.
FROM docker.io/s390x/ubuntu:22.04

# Packages for pytorch building and testing.
ENV DEBIAN_FRONTEND=noninteractive
RUN apt-get update && apt-get -y install \
        cmake \
        curl \
        gcc \
        git \
        jq \
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

# amd64 Github Actions Runner.
RUN useradd -m actions-runner
USER actions-runner
WORKDIR /home/actions-runner
RUN curl -L https://github.com/actions/runner/releases/download/v2.309.0/actions-runner-linux-x64-2.309.0.tar.gz | tar -xz

# repository
ARG repo

# repository token
ARG token

RUN ./config.sh \
        --unattended \
        --url "https://github.com/${repo}" \
        --token "${token}" \
        --no-default-labels \
        --labels self-hosted,linux.s390x

ENTRYPOINT ["/usr/bin/entrypoint"]
CMD ["/usr/bin/actions-runner"]
