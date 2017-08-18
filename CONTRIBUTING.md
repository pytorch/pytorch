## Contributing to PyTorch

If you are interested in contributing to PyTorch, your contributions will fall
into two categories:
1. You want to propose a new Feature and implement it
    - post about your intended feature, and we shall discuss the design and
    implementation. Once we agree that the plan looks good, go ahead and implement it.
2. You want to implement a feature or bug-fix for an outstanding issue
    - Look at the outstanding issues here: https://github.com/pytorch/pytorch/issues
    - Especially look at the Low Priority and Medium Priority issues
    - Pick an issue and comment on the task that you want to work on this feature
    - If you need more context on a particular issue, please ask and we shall provide.

Once you finish implementing a feature or bugfix, please send a Pull Request to
https://github.com/pytorch/pytorch

If you are not familiar with creating a Pull Request, here are some guides:
- http://stackoverflow.com/questions/14680711/how-to-do-a-github-pull-request
- https://help.github.com/articles/creating-a-pull-request/


## Developing locally with PyTorch

To locally develop with PyTorch, here are some tips:

1. Uninstall all existing pytorch installs
```
conda uninstall pytorch
pip uninstall torch
pip uninstall torch # run this command twice
```

2. Locally clone a copy of PyTorch from source:

```
git clone https://github.com/pytorch/pytorch
cd pytorch
```

3. Install PyTorch in `build develop` mode:

A full set of instructions on installing PyTorch from Source are here:
https://github.com/pytorch/pytorch#from-source

The change you have to make is to replace

```
python setup.py install
```

with

```
python setup.py build develop
```

This is especially useful if you are only changing Python files.

This mode will symlink the python files from the current local source tree into the
python install.

Hence, if you modify a python file, you do not need to reinstall pytorch again and again.

For example:
- Install local pytorch in `build develop` mode
- modify your python file `torch/__init__.py` (for example)
- test functionality
- modify your python file `torch/__init__.py`
- test functionality
- modify your python file `torch/__init__.py`
- test functionality

You do not need to repeatedly install after modifying python files.


## Writing documentation

PyTorch uses [Google style](http://sphinxcontrib-napoleon.readthedocs.io/en/latest/example_google.html)
for formatting docstrings. Length of line inside docstrings block must be limited to 80 characters to
fit into Jupyter documentation popups.


## Managing multiple build trees

One downside to using `python setup.py develop` is that your development
version of pytorch will be installed globally on your account (e.g., if
you run `import torch` anywhere else, the development version will be
used.

If you want to manage multiple builds of PyTorch, you can make use of
[conda environments](https://conda.io/docs/using/envs.html) to maintain
separate Python package environments, each of which can be tied to a
specific build of PyTorch.  To set one up:

```
conda create -n pytorch-myfeature
source activate pytorch-myfeature
# if you run python now, torch will NOT be installed
python setup.py build develop
```

## C++ Development tips

If you are working on the C++ code, there are a few important things that you
will want to keep in mind:

1. How to rebuild only the code you are working on, and
2. How to make rebuilds in the absence of changes go faster.

### Build only what you need.

`python setup.py build` will build everything, but since our build system is
not very optimized for incremental rebuilds, this will actually be very slow.
Far better is to only request rebuilds of the parts of the project you are
working on:

- Working on `torch/csrc`?  Run `python setup.py develop` to rebuild
  (NB: no `build` here!)

- Working on `torch/lib/TH`, did not make any cmake changes, and just want to
  see if it compiles?  Run `(cd torch/lib/build/TH && make install -j$(getconf _NPROCESSORS_ONLN))`.  This
  applies for any other subdirectory of `torch/lib`.  **Warning: Changes you
  make here will not be visible from Python.**  See below.

- Working on `torch/lib` and want to run your changes / rerun cmake?  Run
  `python setup.py build_deps`.  Note that this will rerun cmake for
  every subdirectory in TH; if you are only working on one project,
  consider editing `torch/lib/build_all.sh` and commenting out the
  `build` lines of libraries you are not working on.

On the initial build, you can also speed things up with the environment
variables `DEBUG` and `NO_CUDA`.

- `DEBUG=1` will enable debug builds (-g -O0)
- `NO_CUDA=1` will disable compiling CUDA (in case you are developing on something not CUDA related), to save compile time.

For example:
```
NO_CUDA=1 DEBUG=1 python setup.py build develop
```

Make sure you continue to pass these flags on subsequent builds.

### Make no-op build fast.

Python `setuptools` is pretty dumb, and always rebuilds every C file in a
project. Using ccache in a situation like this is a real time-saver. However, by
default, ccache does not properly support CUDA stuff, so here are the
instructions for installing a custom `ccache` fork that has CUDA support:

```
# install and export ccache
if ! ls ~/ccache/bin/ccache
then
    sudo apt-get update
    sudo apt-get install -y automake autoconf
    sudo apt-get install -y asciidoc
    mkdir -p ~/ccache
    pushd /tmp
    rm -rf ccache
    git clone https://github.com/colesbury/ccache -b ccbin
    pushd ccache
    ./autogen.sh
    ./configure
    make install prefix=~/ccache
    popd
    popd

    mkdir -p ~/ccache/lib
    mkdir -p ~/ccache/cuda
    ln -s ~/ccache/bin/ccache ~/ccache/lib/cc
    ln -s ~/ccache/bin/ccache ~/ccache/lib/c++
    ln -s ~/ccache/bin/ccache ~/ccache/lib/gcc
    ln -s ~/ccache/bin/ccache ~/ccache/lib/g++
    ln -s ~/ccache/bin/ccache ~/ccache/cuda/nvcc

    ~/ccache/bin/ccache -M 25Gi
fi

export PATH=~/ccache/lib:$PATH
export CUDA_NVCC_EXECUTABLE=~/ccache/cuda/nvcc
```


Hope this helps, and thanks for considering to contribute.
