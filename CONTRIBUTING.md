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

`python setup.py install`

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
- modify your python file torch/__init__.py (for example)
- test functionality
- modify your python file torch/__init__.py
- test functionality
- modify your python file torch/__init__.py
- test functionality

You do not need to repeatedly install after modifying python files.

When you are developing on the C++ side of things, the environment variables `DEBUG` and `NOCUDA` are helpful.

- `DEBUG=1` will enable debug builds (-g -O0)
- `NOCUDA=1` will disable compiling CUDA (in case you are developing on something not CUDA related), to save compile time.

For example:
```
NO_CUDA=1 DEBUG=1 python setup.py build develop
```

Also, if you are developing a lot, using ccache is a real time-saver. By default, ccache does not properly support CUDA stuff, so here are the instructions for installing a custom `ccache` fork that has CUDA support:
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
