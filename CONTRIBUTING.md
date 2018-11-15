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

1. Uninstall all existing PyTorch installs:
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

## Codebase structure

* [c10](c10) - Core library files that work everywhere, both server
  and mobile.  We are slowly moving pieces from ATen/core here.
  This library is intended only to contain essential functionality,
  and appropriate to use in settings where binary size matters.  (But
  you'll have a lot of missing functionality if you try to use it
  directly.)
* [aten](aten) - C++ tensor library for PyTorch (no autograd support)
  * src
    * [TH](aten/src/TH)
      [THC](aten/src/THC)
      [THNN](aten/src/THNN)
      [THCUNN](aten/src/THCUNN) - Legacy library code from the original
      Torch.  Try not to add things here; we're slowly porting these to
      native.
      * generic - Contains actual implementations of operators,
        parametrized over `scalar_t`.  Files here get compiled N times
        per supported scalar type in PyTorch.
    * ATen
      * [core](aten/src/ATen/core) - Core functionality of ATen.  This
        is migrating to top-level c10 folder.
      * [native](aten/src/ATen/native) - Modern implementations of
        operators.  If you want to write a new operator, here is where
        it should go.  Most CPU operators go in the top level directory,
        except for operators which need to be compiled specially; see
        cpu below.
        * [cpu](aten/src/ATen/native/cpu) - Not actually CPU
          implementations of operators, but specifically implementations
          which are compiled with processor-specific instructions, like
          AVX.  See the README for more details.
        * [cuda](aten/src/ATen/native/cuda) - CUDA implementations of
          operators.
        * [sparse](aten/src/ATen/native/sparse) - CPU and CUDA
          implementations of COO sparse tensor operations
        * [mkl](aten/src/ATen/native/mkl) [mkldnn](aten/src/ATen/native/mkldnn)
          [miopen](aten/src/ATen/native/miopen) [cudnn](aten/src/ATen/native/cudnn)
          - implementations of operators which simply bind to some
            backend library.
* [torch](torch) - The actual PyTorch library.  Everything that is not
  in csrc is Python modules, following the PyTorch Python frontend
  module structure.
  * [csrc](torch/csrc) - C++ files composing the PyTorch library.  Files
    in this directory tree are a mix of Python binding code, and C++
    heavy lifting.  Consult `setup.py` for the canonical list of Python
    binding files; conventionally, they are often prefixed with
    `python_`.
    * [jit](torch/csrc/jit) - Compiler and frontend for TorchScript JIT
      frontend.
    * [autograd](torch/csrc/autograd) - Implementation of reverse-mode automatic
      differentation
    * [api](torch/csrc/api) - The PyTorch C++ frontend.
    * [distributed](torch/csrc/distributed) - Distributed training
      support for PyTorch.
* [tools](tools) - Code generation scripts for the PyTorch library.
  See README of this directory for more details.
* [test](tests) - Python unit tests for PyTorch Python frontend
  * [test_torch.py](test/test_torch.py) - Basic tests for PyTorch
    functionality
  * [test_autograd.py](test/test_autograd.py) - Tests for non-NN
    automatic differentiation support
  * [test_nn.py](test/test_nn.py) - Tests for NN operators and
    their automatic differentiation
  * [test_jit.py](test/test_jit.py) - Tests for the JIT compiler
    and TorchScript
  * ...
  * [cpp](test/cpp) - C++ unit tests for PyTorch C++ frontend
  * [expect](test/expect) - Automatically generated "expect" files
    which are used to compare against expected output.
  * [onnx](test/onnx) - Tests for ONNX export functionality,
    using both PyTorch and Caffe2.
* [caffe2](caffe2) - The Caffe2 library.
  * [core](caffe2/core) - Core files of Caffe2, e.g., tensor, workspace,
    blobs, etc.
  * [operators](caffe2/operators) - Operators of Caffe2
  * [python](caffe2/python) - Python bindings to Caffe2
  * ...

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

For C++ documentation (https://pytorch.org/cppdocs), we use
[Doxygen](http://www.doxygen.nl/) and then convert it to
[Sphinx](http://www.sphinx-doc.org/) via
[Breathe](https://github.com/michaeljones/breathe) and
[Exhale](https://github.com/svenevs/exhale). Check the [Doxygen
reference](http://www.stack.nl/~dimitri/doxygen/manual/index.html) for more
information on the documentation syntax. To build the documentation locally,
`cd` into `docs/cpp` and then `make html`.

We run Doxygen in CI (Travis) to verify that you do not use invalid Doxygen
commands. To run this check locally, run `./check-doxygen.sh` from inside
`docs/cpp`.

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
If pytorch was already built, you will need to run `python setup.py clean` once
after installing ninja for builds to succeed.

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
   which follow the convention `*_API`, e.g., `CAFFE2_API` inside Caffe2 and ATen.
   (Every separate shared library needs a unique macro name, because symbol visibility
   is on a per shared library basis. See c10/macros/Macros.h for more details.)

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
   blowing away your Conda environment.)

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

### Known MSVC (and MSVC with NVCC) bugs

The PyTorch codebase sometimes likes to use exciting C++ features, and
these exciting features lead to exciting bugs in Windows compilers.
To add insult to injury, the error messages will often not tell you
which line of code actually induced the erroring template instantiation.

I've found the most effective way to debug these problems is to
carefully read over diffs, keeping in mind known bugs in MSVC/NVCC.
Here are a few well known pitfalls and workarounds:

* This is not actually a bug per se, but in general, code generated by MSVC
  is more sensitive to memory errors; you may have written some code
  that does a use-after-free or stack overflows; on Linux the code
  might work, but on Windows your program will crash.  ASAN may not
  catch all of these problems: stay vigilant to the possibility that
  your crash is due to a real memory problem.

* (NVCC) `c10::optional` does not work when used from device code.  Don't use
  it from kernels.  Upstream issue: https://github.com/akrzemi1/Optional/issues/58
  and our local issue #10329.

* `constexpr` generally works less well on MSVC.

  * The idiom `static_assert(f() == f())` to test if `f` is constexpr
    does not work; you'll get "error C2131: expression did not evaluate
    to a constant".  Don't use these asserts on Windows.
    (Example: `aten/src/ATen/core/intrusive_ptr.h`)

* (NVCC) Code you access inside a `static_assert` will eagerly be
  evaluated as if it were device code, and so you might get an error
  that the code is "not accessible".

```
class A {
  static A singleton_;
  static constexpr inline A* singleton() {
    return &singleton_;
  }
};
static_assert(std::is_same(A*, decltype(A::singelton()))::value, "hmm");
```

* The compiler will run out of heap if you attempt to compile files that
  are too large.  Splitting such files into separate files helps.
  (Example: `THTensorMath`, `THTensorMoreMath`, `THTensorEvenMoreMath`.)

### Running Clang-Tidy

[Clang-Tidy](https://clang.llvm.org/extra/clang-tidy/index.html) is a C++
linter and static analysis tool based on the clang compiler. We run clang-tidy
in our CI to make sure that new C++ code is safe, sane and efficient. See our
[.travis.yml](https://github.com/pytorch/pytorch/blob/master/.travis.yml) file
for the simple commands we use for this.

To run clang-tidy locally, follow these steps:

1. Install clang-tidy. First, check if you already have clang-tidy by simply
writing `clang-tidy` in your terminal. If you don't yet have clang-tidy, you
should be able to install it easily with your package manager, e.g. by writing
`apt-get install clang-tidy` on Ubuntu. See https://apt.llvm.org for details on
how to install the latest version. Note that newer versions of clang-tidy will
have more checks than older versions. In our CI, we run clang-tidy-6.0.

2. Use our driver script to run clang-tidy over any changes relative to some
   git revision (you may want to replace `HEAD~1` with `HEAD` to pick up
   uncommitted changes). Changes are picked up based on a `git diff` with the
   given revision:
  ```sh
  $ python tools/clang_tidy.py -d build -p torch/csrc --diff 'HEAD~1'
  ```

Above, it is assumed you are in the PyTorch root folder. `path/to/build` should
be the path to where you built PyTorch from source, e.g. `build` in the PyTorch
root folder if you used `setup.py build`. You can use `-c <clang-tidy-binary>`
to change the clang-tidy this script uses. Make sure you have PyYaml installed,
which is in PyTorch's `requirements.txt`.

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
