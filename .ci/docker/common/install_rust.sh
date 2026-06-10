#!/bin/bash
# Install a persistent, system-wide Rust toolchain.
#
# PyTorch's build compiles the torch._rust extension (see setup.py
# _build_rust_extensions), so cargo must be on PATH for the build user at build
# time. This script is shared by the Ubuntu CI build images and the
# almalinux/manylinux CD wheel-builder images, so it supports both apt and yum.

set -ex

# Pinned version; see ci_commit_pins/rust.txt. Bump deliberately.
RUST_VERSION="$(cat rust.txt)"

# System-wide locations; must match RUSTUP_HOME/CARGO_HOME/PATH in the Dockerfile.
export RUSTUP_HOME="${RUSTUP_HOME:-/opt/rust}"
export CARGO_HOME="${CARGO_HOME:-/opt/rust}"

# Ensure curl + CA certs are present (distro-agnostic).
if command -v apt-get >/dev/null; then
  apt-get update
  apt-get install -y --no-install-recommends ca-certificates curl
  rm -rf /var/lib/apt/lists/*
elif command -v dnf >/dev/null; then
  dnf install -y ca-certificates curl
elif command -v yum >/dev/null; then
  yum install -y ca-certificates curl
fi

curl --retry 3 --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | \
  sh -s -- -y --no-modify-path --profile minimal --default-toolchain "${RUST_VERSION}"

# Let the build user read the toolchain and write the cargo registry/git caches
# at job time. Mirrors the official rust docker image.
chmod -R a+w "${RUSTUP_HOME}"

"${CARGO_HOME}/bin/cargo" --version
"${CARGO_HOME}/bin/rustc" --version
