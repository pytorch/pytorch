#!/bin/bash
#
# Install dependencies for TorchInductor on TPU.

# Install dependencies from requirements.txt first
pip install -r requirements.txt

# Install JAX nightly builds and other TPU dependencies
pip install --pre -i https://us-python.pkg.dev/ml-oss-artifacts-published/jax/simple/ -f https://storage.googleapis.com/jax-releases/libtpu_releases.html jax==0.8.0.dev20251013 jaxlib==0.8.0.dev20251013 libtpu==0.0.25.dev20251012+nightly tpu-info==0.6.0 setuptools==78.1.0  # @lint-ignore
