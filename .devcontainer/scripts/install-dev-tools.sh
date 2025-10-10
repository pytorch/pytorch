#!/usr/bin/env bash
# Run this command from the PyTorch directory after cloning the source code using the “Get the PyTorch Source“ section below
pip install -r requirements.txt
git submodule sync
git submodule update --init --recursive

# This takes some time
make setup-lint

# Add CMAKE_PREFIX_PATH to bashrc
echo 'export CMAKE_PREFIX_PATH=/usr/local' >> ~/.bashrc
# Add linker path so that cuda-related libraries can be found
echo 'export LDFLAGS="-L/usr/local/cuda/lib64/ $LDFLAGS"' >> ~/.bashrc
