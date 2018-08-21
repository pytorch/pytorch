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


## Developing PyTorch

To develop PyTorch on your machine, here are some tips:

1. Uninstall all existing pytorch installs
```
conda uninstall pytorch
pip uninstall torch
pip uninstall torch # run this command twice
```

2. Clone a copy of PyTorch from source:

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

In case you want to reinstall, make sure that you uninstall pytorch first by running `pip uninstall torch`
and `python setup.py clean`. Then you can install in `build develop` mode again.

## Unit testing

PyTorch's testing is located under `test/`. Run the entire test suite with

```
python test/run_test.py
```

or run individual test files, like `python test/test_nn.py`, for individual test suites.

### Better local unit tests with pytest
We don't officially support `pytest`, but it works well with our `unittest` tests and offers
a number of useful features for local developing. Install it via `pip install pytest`.

If you want to just run tests that contain a specific substring, you can use the `-k` flag:

```
pytest test/test_nn.py -k Loss -v
```

The above is an example of testing a change to Loss functions: this command runs tests such as
`TestNN.test_BCELoss` and `TestNN.test_MSELoss` and can be useful to save keystrokes.

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

- Working on the Python bindings?  Run `python setup.py develop` to rebuild
  (NB: no `build` here!)

- Working on `torch/csrc` or `aten`?  Run `python setup.py rebuild_libtorch` to
  rebuild and avoid having to rebuild other dependent libraries we
  depend on.

- Working on one of the other dependent libraries? The other valid
  targets are listed in `dep_libs` in `setup.py`. prepend `build_` to
  get a target, and run as e.g. `python setup.py build_gloo`.

- Working on a test binary?  Run `(cd build && ninja bin/test_binary_name)` to
  rebuild only that test binary (without rerunning cmake).  (Replace `ninja` with
  `make` if you don't have ninja installed).

On the initial build, you can also speed things up with the environment
variables `DEBUG` and `NO_CUDA`.

- `DEBUG=1` will enable debug builds (-g -O0)
- `NO_CUDA=1` will disable compiling CUDA (in case you are developing on something not CUDA related), to save compile time.

For example:
```
NO_CUDA=1 DEBUG=1 python setup.py build develop
```

Make sure you continue to pass these flags on subsequent builds.

### Code completion and IDE support

When using `python setup.py develop`, PyTorch will generate
a `compile_commands.json` file that can be used by many editors
to provide command completion and error highlighting for PyTorch's
C++ code. You need to `pip install ninja` to generate accurate
information for the code in `torch/csrc`. More information at:
- https://sarcasm.github.io/notes/dev/compilation-database.html

### Make no-op build fast.

#### Use Ninja
Python `setuptools` is pretty dumb, and always rebuilds every C file in a
project.  If you install the ninja build system with `pip install ninja`,
then PyTorch will use it to track dependencies correctly.

#### Use CCache

Even when dependencies are tracked with file modification,
there are many situations where files get rebuilt when a previous
compilation was exactly the same.

Using ccache in a situation like this is a real time-saver. However, by
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

## CUDA Development tips

If you are working on the CUDA code, here are some useful CUDA debugging tips:

1. `CUDA_DEVICE_DEBUG=1` will enable CUDA device function debug symbols (`-g -G`).
    This will be particularly helpful in debugging device code. However, it will
    slow down the build process for about 50% (compared to only `DEBUG=1`), so use wisely.
2. `cuda-gdb` and `cuda-memcheck` are your best CUDA debugging friends. Unlike`gdb`,
   `cuda-gdb` can display actual values in a CUDA tensor (rather than all zeros).


Hope this helps, and thanks for considering to contribute.

## Windows development tips

Occasionally, you will write a patch which works on Linux, but fails CI on Windows.
There are a few aspects in which MSVC (the Windows compiler toolchain we use) is stricter
than Linux, which are worth keeping in mind when fixing these problems.

1. Symbols are NOT exported by default on Windows; instead, you have to explicitly
   mark a symbol as exported/imported in a header file with `__declspec(dllexport)` /
   `__declspec(dllimport)`.  We have codified this pattern into a set of macros
   which follow the convention `*_API`, e.g., `AT_API` inside ATen. (Every separate
   shared library needs a unique macro name, because symbol visibility is on a per
   shared library basis.)

   The upshot is if you see an "unresolved external" error in your Windows build, this
   is probably because you forgot to mark a function with `*_API`.  However, there is
   one important counterexample to this principle: if you want a *templated* function
   to be instantiated at the call site, do NOT mark it with `*_API` (if you do mark it,
   you'll have to explicitly instantiate all of the specializations used by the call
   sites.)

2. If you link against a library, this does not make its dependencies transitively
   visible. You must explicitly specify a link dependency against every library whose
   symbols you use.  (This is different from Linux where in most environments,
   transitive dependencies can be used to fulfill unresolved symbols.)

3. If you have a Windows box (we have a few on EC2 which you can request access to) and
   you want to run the build, the easiest way is to just run `.jenkins/pytorch/win-build.sh`.
   If you need to rebuild, run `REBUILD=1 .jenkins/pytorch/win-build.sh` (this will avoid
   blowing away your Conda environment.)  I recommend opening `cmd.exe`, and then running
   `bash` to work in a bash shell (which will make various Linux commands available.)

Even if you don't know anything about MSVC, you can use cmake to build simple programs on
Windows; this can be helpful if you want to learn more about some peculiar linking behavior
by reproducing it on a small example.  Here's a simple example cmake file that defines
two dynamic libraries, one linking with the other:

```
project(myproject CXX)
set(CMAKE_CXX_STANDARD 11)
add_library(foo SHARED foo.cpp)
add_library(bar SHARED bar.cpp)
# NB: don't forget to __declspec(dllexport) at least one symbol from foo,
# otherwise foo.lib will not be created.
target_link_libraries(bar PUBLIC foo)
```

You can build it with:

```
mkdir build
cd build
cmake ..
cmake --build .
```

## Caffe2 notes

In 2018, we merged Caffe2 into the PyTorch source repository.  While the
steady state aspiration is that Caffe2 and PyTorch share code freely,
in the meantime there will be some separation.

If you submit a PR to only PyTorch or only Caffe2 code, CI will only
run for the project you edited.  The logic for this is implemented
in `.jenkins/pytorch/dirty.sh` and `.jenkins/caffe2/dirty.sh`; you
can look at this to see what path prefixes constitute changes.
This also means if you ADD a new top-level path, or you start
sharing code between projects, you need to modify these files.

There are a few "unusual" directories which, for historical reasons,
are Caffe2/PyTorch specific.  Here they are:

- `CMakeLists.txt`, `Makefile`, `binaries`, `cmake`, `conda`, `modules`,
  `scripts` are Caffe2-specific.  Don't put PyTorch code in them without
  extra coordination.

- `mypy*`, `requirements.txt`, `setup.py`, `test`, `tools` are
  PyTorch-specific.  Don't put Caffe2 code in them without extra
  coordination.
