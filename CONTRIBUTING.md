Thank you for your interest in contributing to PyTorch!
If you're a new contributor, please first take a read through our
[Contributing Guide](https://github.com/pytorch/pytorch/wiki/The-Ultimate-Guide-to-PyTorch-Contributions), specifically the [Submitting a Change](https://github.com/pytorch/pytorch/wiki/The-Ultimate-Guide-to-PyTorch-Contributions#submitting-a-change) section
that walks through the process of contributing a change to PyTorch.

The rest of this document (CONTRIBUTING.md) covers some of the more technical
aspects of contributing to PyTorch.

# Table of Contents

<!-- toc -->

- [Developing PyTorch](#developing-pytorch)
  - [Tips and Debugging](#tips-and-debugging)
- [Nightly Checkout & Pull](#nightly-checkout--pull)
- [Codebase structure](#codebase-structure)
- [Unit testing](#unit-testing)
  - [Python Unit Testing](#python-unit-testing)
  - [Better local unit tests with `pytest`](#better-local-unit-tests-with-pytest)
  - [Local linting](#local-linting)
    - [Running `pyrefly`](#running-pyrefly)
  - [C++ Unit Testing](#c-unit-testing)
  - [Run Specific CI Jobs](#run-specific-ci-jobs)
- [Merging your Change](#merging-your-change)
- [Writing documentation](#writing-documentation)
  - [Docstring type formatting](#docstring-type-formatting)
  - [Building documentation](#building-documentation)
    - [Tips](#tips)
    - [Building C++ Documentation](#building-c-documentation)
  - [Previewing changes locally](#previewing-changes-locally)
  - [Previewing documentation on PRs](#previewing-documentation-on-prs)
  - [Adding documentation tests](#adding-documentation-tests)
- [Profiling with `py-spy`](#profiling-with-py-spy)
- [Managing multiple build trees](#managing-multiple-build-trees)
- [C++ development tips](#c-development-tips)
  - [Build only what you need](#build-only-what-you-need)
  - [Code completion and IDE support](#code-completion-and-ide-support)
  - [Make no-op build fast](#make-no-op-build-fast)
    - [Use Ninja](#use-ninja)
    - [Use CCache](#use-ccache)
    - [Use a faster linker](#use-a-faster-linker)
    - [Use pre-compiled headers](#use-pre-compiled-headers)
    - [Workaround for header dependency bug in nvcc](#workaround-for-header-dependency-bug-in-nvcc)
  - [Rebuild few files with debug information](#rebuild-few-files-with-debug-information)
  - [C++ frontend development tips](#c-frontend-development-tips)
  - [GDB integration](#gdb-integration)
  - [C++ stacktraces](#c-stacktraces)
- [CUDA development tips](#cuda-development-tips)
- [Windows development tips](#windows-development-tips)
  - [Known MSVC (and MSVC with NVCC) bugs](#known-msvc-and-msvc-with-nvcc-bugs)
  - [Building on legacy code and CUDA](#building-on-legacy-code-and-cuda)
- [Pre-commit tidy/linting hook](#pre-commit-tidylinting-hook)
- [Building PyTorch with ASAN](#building-pytorch-with-asan)
  - [Getting `ccache` to work](#getting-ccache-to-work)
  - [Why this stuff with `LD_PRELOAD` and `LIBASAN_RT`?](#why-this-stuff-with-ld_preload-and-libasan_rt)
  - [Why LD_PRELOAD in the build function?](#why-ld_preload-in-the-build-function)
  - [Why no leak detection?](#why-no-leak-detection)
- [Caffe2 notes](#caffe2-notes)
- [CI failure tips](#ci-failure-tips)
  - [Which commit is used in CI?](#which-commit-is-used-in-ci)
- [Dev Infra Office Hours](#dev-infra-office-hours)

<!-- tocstop -->

## Developing PyTorch

Follow the instructions for [installing PyTorch from source](https://github.com/pytorch/pytorch#from-source). If you get stuck when developing PyTorch on your machine, check out the [tips and debugging](#tips-and-debugging) section below for common solutions.

### Tips and Debugging

* If you want to have no-op incremental rebuilds (which are fast), see [Make no-op build fast](#make-no-op-build-fast) below.

* When installing with `python -m pip install -e . -v --no-build-isolation` (in contrast to `python -m pip install . -v --no-build-isolation`) Python runtime will use
  the current local source-tree when importing `torch` package. (This is done by creating [`.egg-link`](https://wiki.python.org/moin/PythonPackagingTerminology#egg-link) file in `site-packages` folder)
  This way you do not need to repeatedly install after modifying Python files (`.py`).
  However, you would need to reinstall if you modify Python interface (`.pyi`, `.pyi.in`) or non-Python files (`.cpp`, `.cc`, `.cu`, `.h`, ...).


  One way to avoid running `python -m pip install -e . -v --no-build-isolation` every time one makes a change to C++/CUDA/ObjectiveC files on Linux/Mac,
  is to create a symbolic link from `build` folder to `torch/lib`, for example, by issuing following:
  ```bash
  pushd torch/lib; sh -c "ln -sf ../../build/lib/libtorch_cpu.* ."; popd
  ```
  Afterwards rebuilding a library (for example to rebuild `libtorch_cpu.so` issue `ninja torch_cpu` from `build` folder),
  would be sufficient to make change visible in `torch` package.


  To reinstall, first uninstall all existing PyTorch installs. You may need to run `pip
  uninstall torch` multiple times. You'll know `torch` is fully
  uninstalled when you see `WARNING: Skipping torch as it is not
  installed`. (You should only have to `pip uninstall` a few times, but
  you can always `uninstall` with `timeout` or in a loop if you're feeling
  lazy.)

  ```bash
  pip uninstall torch
  ```

  Next run `python setup.py clean`. After that, you can install in editable mode again.

* If you run into errors when running `python -m pip install -e . -v --no-build-isolation`, here are some debugging steps:
  1. Run `printf '#include <stdio.h>\nint main() { printf("Hello World");}'|clang -x c -; ./a.out` to make sure
  your CMake works and can compile this simple Hello World program without errors.
  2. Nuke your `build` directory. The `setup.py` script compiles binaries into the `build` folder and caches many
  details along the way, which saves time the next time you build. If you're running into issues, you can always
  `rm -rf build` from the toplevel `pytorch` directory and start over.
  3. If you have made edits to the PyTorch repo, commit any change you'd like to keep and clean the repo with the
  following commands (note that clean _really_ removes all untracked files and changes.):
      ```bash
      git submodule deinit -f .
      git clean -xdf
      python setup.py clean
      git submodule update --init --recursive
      python -m pip install --group dev
      python -m pip install --no-build-isolation -v -e .
      ```
  4. The main step within `python -m pip install -e . -v --no-build-isolation` is running `make` from the `build` directory. If you want to
    experiment with some environment variables, you can pass them into the command:
      ```bash
      ENV_KEY1=ENV_VAL1[, ENV_KEY2=ENV_VAL2]* CMAKE_FRESH=1 python -m pip install --no-build-isolation -v -e .
      ```
  5. Try installing PyTorch without build isolation by adding `--no-build-isolation` to the `pip install` command.
  This will use the current environment's packages instead of creating a new isolated environment for the build.
      ```bash
      python -m pip install --no-build-isolation -v -e .
      ```


* If you run into issue running `git submodule update --init --recursive`. Please try the following:
  - If you encounter an error such as
    ```
    error: Submodule 'third_party/pybind11' could not be updated
    ```
    check whether your Git local or global config file contains any `submodule.*` settings. If yes, remove them and try again.
    (please reference [this doc](https://git-scm.com/docs/git-config#Documentation/git-config.txt-submoduleltnamegturl) for more info).

  - If you encounter an error such as
    ```
    fatal: unable to access 'https://github.com/pybind/pybind11.git': could not load PEM client certificate ...
    ```
    this is likely that you are using HTTP proxying and the certificate expired. To check if the certificate is valid, run
    `git config --global --list` and search for config like `http.proxysslcert=<cert_file>`. Then check certificate valid date by running
    ```bash
    openssl x509 -noout -in <cert_file> -dates
    ```

  - If you encounter an error that some third_party modules are not checked out correctly, such as
    ```
    Could not find .../pytorch/third_party/pybind11/CMakeLists.txt
    ```
    remove any `submodule.*` settings in your local git config (`.git/config` of your pytorch repo) and try again.
* If you're a Windows contributor, please check out [Best Practices](https://github.com/pytorch/pytorch/wiki/Best-Practices-to-Edit-and-Compile-Pytorch-Source-Code-On-Windows).
* For help with any part of the contributing process, please don’t hesitate to utilize our Zoom office hours! See details [here](https://github.com/pytorch/pytorch/wiki/Dev-Infra-Office-Hours)

## Nightly Checkout & Pull

The `tools/nightly.py` script is provided to ease pure Python development of
PyTorch. This uses `venv` and `git` to check out the nightly development
version of PyTorch and installs pre-built binaries into the current repository.
This is like a development or editable install, but without needing the ability
to compile any C++ code.

You can use this script to check out a new nightly branch with the following:

```bash
./tools/nightly.py checkout -b my-nightly-branch
source venv/bin/activate  # or `. .\venv\Scripts\activate` on Windows
```

To install the nightly binaries built with CUDA, you can pass in the flag `--cuda`:

```bash
./tools/nightly.py checkout -b my-nightly-branch --cuda
source venv/bin/activate  # or `. .\venv\Scripts\activate` on Windows
```

To install the nightly binaries built with ROCm, you can pass in the flag `--rocm`:

```bash
./tools/nightly.py checkout -b my-nightly-branch --rocm
source venv/bin/activate  # or `. .\venv\Scripts\activate` on Windows
```

You can also use this tool to pull the nightly commits into the current branch:

```bash
./tools/nightly.py pull
source venv/bin/activate  # or `. .\venv\Scripts\activate` on Windows
```

To create the virtual environment with a specific Python interpreter, you can
pass in the `--python` argument:

```bash
./tools/nightly.py --python /path/to/python3.12
source venv/bin/activate  # or `. .\venv\Scripts\activate` on Windows
```

Pulling will recreate a fresh virtual environment and reinstall the development
dependencies as well as the nightly binaries into the repo directory.

## Codebase structure

* [c10](c10) - Core library files that work everywhere, both server
  and mobile. We are slowly moving pieces from [ATen/core](aten/src/ATen/core)
  here. This library is intended only to contain essential functionality,
  and appropriate to use in settings where binary size matters. (But
  you'll have a lot of missing functionality if you try to use it
  directly.)
* [aten](aten) - C++ tensor library for PyTorch (no autograd support)
  * [src](aten/src) - [README](aten/src/README.md)
    * [ATen](aten/src/ATen)
      * [core](aten/src/ATen/core) - Core functionality of ATen. This
        is migrating to top-level c10 folder.
      * [native](aten/src/ATen/native) - Modern implementations of
        operators. If you want to write a new operator, here is where
        it should go. Most CPU operators go in the top level directory,
        except for operators which need to be compiled specially; see
        cpu below.
        * [cpu](aten/src/ATen/native/cpu) - Not actually CPU
          implementations of operators, but specifically implementations
          which are compiled with processor-specific instructions, like
          AVX. See the [README](aten/src/ATen/native/cpu/README.md) for more
          details.
        * [cuda](aten/src/ATen/native/cuda) - CUDA implementations of
          operators.
        * [mps](aten/src/ATen/native/mps) - MPS implementations of
          operators for Apple's Metal GPU family.
        * [sparse](aten/src/ATen/native/sparse) - CPU and CUDA
          implementations of COO sparse tensor operations
        * [mkl](aten/src/ATen/native/mkl) [mkldnn](aten/src/ATen/native/mkldnn)
          [miopen](aten/src/ATen/native/miopen) [cudnn](aten/src/ATen/native/cudnn)
          - implementations of operators which simply bind to some
            backend library.
        * [quantized](aten/src/ATen/native/quantized/) - Quantized tensor (i.e. QTensor) operation implementations. [README](aten/src/ATen/native/quantized/README.md) contains details including how to implement native quantized operations.
* [torch](torch) - The actual PyTorch library. Everything that is not
  in [csrc](torch/csrc) is a Python module, following the PyTorch Python
  frontend module structure.
  * [csrc](torch/csrc) - C++ files composing the PyTorch library. Files
    in this directory tree are a mix of Python binding code, and C++
    heavy lifting. Consult `setup.py` for the canonical list of Python
    binding files; conventionally, they are often prefixed with
    `python_`. [README](torch/csrc/README.md)
    * [jit](torch/csrc/jit) - Compiler and frontend for TorchScript JIT
      frontend. [README](torch/csrc/jit/README.md)
    * [autograd](torch/csrc/autograd) - Implementation of reverse-mode automatic differentiation. [README](torch/csrc/autograd/README.md)
    * [api](torch/csrc/api) - The PyTorch C++ frontend.
    * [distributed](torch/csrc/distributed) - Distributed training
      support for PyTorch.
* [tools](tools) - Code generation scripts for the PyTorch library.
  See [README](tools/README.md) of this directory for more details.
* [torchgen](torchgen) - contains the logic and tooling for generating PyTorch's low-level C++ and Python bindings from operator definitions, typically specified in native_functions.yaml
* [test](test) - Python unit tests for PyTorch Python frontend.
  * [test_torch.py](test/test_torch.py) - Basic tests for PyTorch
    functionality.
  * [test_autograd.py](test/test_autograd.py) - Tests for non-NN
    automatic differentiation support.
  * [test_nn.py](test/test_nn.py) - Tests for NN operators and
    their automatic differentiation.
  * [test_jit.py](test/test_jit.py) - Tests for the JIT compiler
    and TorchScript.
  * ...
  * [cpp](test/cpp) - C++ unit tests for PyTorch C++ frontend.
    * [api](test/cpp/api) - [README](test/cpp/api/README.md)
    * [jit](test/cpp/jit) - [README](test/cpp/jit/README.md)
    * [tensorexpr](test/cpp/tensorexpr) - [README](test/cpp/tensorexpr/README.md)
  * [expect](test/expect) - Automatically generated "expect" files
    which are used to compare against expected output.
  * [onnx](test/onnx) - Tests for ONNX export functionality,
    using both PyTorch and Caffe2.
* [caffe2](caffe2) - The Caffe2 library.
  * [core](caffe2/core) - Core files of Caffe2, e.g., tensor, workspace,
    blobs, etc.
  * ...
* [.circleci](.circleci) - CircleCI configuration management. [README](.circleci/README.md)

## Unit testing

### Python Unit Testing

**Prerequisites**:
The following packages should be installed with `pip`:
- `expecttest` and `hypothesis` - required to run tests
- `pyrefly` - recommended for type checking. [Pyrefly](https://pyrefly.org/)
- `pytest` - recommended to run tests more selectively
Running
```
pip install --group dev
```
will install these dependencies for you.

All PyTorch test suites are located in the `test` folder and start with
`test_`. Run the entire test
suite with

```bash
python test/run_test.py
```

or run individual test suites using the command `python test/FILENAME.py`,
where `FILENAME` represents the file containing the test suite you wish
to run.

For example, to run all the TorchScript JIT tests (located at
`test/test_jit.py`), you would run:

```bash
python test/test_jit.py
```

You can narrow down what you're testing even further by specifying the
name of an individual test with `TESTCLASSNAME.TESTNAME`. Here,
`TESTNAME` is the name of the test you want to run, and `TESTCLASSNAME`
is the name of the class in which it is defined.

Going off the above example, let's say you want to run
`test_Sequential`, which is defined as part of the `TestJit` class
in `test/test_jit.py`. Your command would be:

```bash
python test/test_jit.py TestJit.test_Sequential
```

**Weird note:** In our CI (Continuous Integration) jobs, we actually run the tests from the `test` folder and **not** the root of the repo, since there are various dependencies we set up for CI that expects the tests to be run from the test folder. As such, there may be some inconsistencies between local testing and CI testing--if you observe an inconsistency, please [file an issue](https://github.com/pytorch/pytorch/issues/new/choose).

### Better local unit tests with `pytest`

We don't officially support `pytest`, but it works well with our
`unittest` tests and offers a number of useful features for local
developing. Install it via `pip install pytest`.

If you want to just run tests that contain a specific substring, you can
use the `-k` flag:

```bash
pytest test/test_nn.py -k Loss -v
```

The above is an example of testing a change to all Loss functions: this
command runs tests such as `TestNN.test_BCELoss` and
`TestNN.test_MSELoss` and can be useful to save keystrokes.

### Local linting

You can run the same linting steps that are used in CI locally via `make`:

```bash
make lint
```

Learn more about the linter on the [lintrunner wiki page](https://github.com/pytorch/pytorch/wiki/lintrunner)

#### Running `pyrefly`

[Pyrefly](https://pyrefly.org/) is a high-performance static type checker for Python. It provides fast type checking along with IDE features like autocomplete and instant error feedback.

PyTorch uses Pyrefly for type checking across the codebase. The configuration is managed in `pyrefly.toml` at the root of the repository.

**Getting Started with Pyrefly:**

To run type checking on the PyTorch codebase:
```bash
pyrefly check
```

For more detailed error information with summaries:
```bash
pyrefly check --summarize-errors
```

**Learn More:**
- [Pyrefly Configuration](https://pyrefly.org/en/docs/configuration/) - Detailed configuration options
- [Pyrefly IDE Features](https://pyrefly.org/en/docs/IDE-features/) - Set up Pyrefly in your editor for real-time type checking
- [Python Typing Tutorial](https://pyrefly.org/en/docs/typing-for-python-developers/) - Learn about Python type annotations

See [Guide for adding type annotations to
PyTorch](https://github.com/pytorch/pytorch/wiki/Guide-for-adding-type-annotations-to-PyTorch)
for PyTorch-specific guidance on how to set up `pyrefly` and tackle type annotation tasks in this codebase.

### C++ Unit Testing

PyTorch offers a series of tests located in the `test/cpp` folder.
These tests are written in C++ and use the Google Test testing framework.
After compiling PyTorch from source, the test runner binaries will be
written to the `build/bin` folder. The command to run one of these tests
is `./build/bin/FILENAME --gtest_filter=TESTSUITE.TESTNAME`, where
`TESTNAME` is the name of the test you'd like to run and `TESTSUITE` is
the suite that test is defined in.

For example, if you wanted to run the test `MayContainAlias`, which
is part of the test suite `ContainerAliasingTest` in the file
`test/cpp/jit/test_alias_analysis.cpp`, the command would be:

```bash
./build/bin/test_jit --gtest_filter=ContainerAliasingTest.MayContainAlias
```


### Run Specific CI Jobs

You can generate a commit that limits the CI to only run a specific job by using
`tools/testing/explicit_ci_jobs.py` like so:

```bash
# --job: specify one or more times to filter to a specific job + its dependencies
# --filter-gha: specify github actions workflows to keep
# --make-commit: commit CI changes to git with a message explaining the change
python tools/testing/explicit_ci_jobs.py --job binary_linux_manywheel_3_6m_cpu_devtoolset7_nightly_test --filter-gha '*generated*gcc5.4*' --make-commit

# Make your changes

ghstack submit
```

**NB**: It is not recommended to use this workflow unless you are also using
[`ghstack`](https://github.com/ezyang/ghstack). It creates a large commit that is
of very low signal to reviewers.

## Merging your Change
If you know the right people or team that should approve your PR (and you have the required permissions to do so), add them to the Reviewers list.

If not, leave the Reviewers section empty. Our triage squad will review your PR, add a module label, and assign it to the appropriate reviewer in a couple business days.  The reviewer will then look at your PR and respond.

Occasionally, things might fall through the cracks (sorry!). In case your PR either doesn't get assigned to a reviewer or doesn't get any response from the reviewer for 4 business days, please leave comment on the PR (mentioning the reviewer if one has been assigned). That'll get it nudged back onto people's radar.

If that still doesn't help, come see us during [our office hours](https://github.com/pytorch/pytorch/wiki/Dev-Infra-Office-Hours)

Once your PR is approved, you can merge it in by entering a comment with the content `@pytorchmergebot merge` ([what's this bot?](https://github.com/pytorch/pytorch/wiki/Bot-commands))

## Writing documentation

So you want to write some documentation and don't know where to start?
PyTorch has two main types of documentation:
- **User facing documentation**:
These are the docs that you see over at [our docs website](https://pytorch.org/docs).
- **Developer facing documentation**:
Developer facing documentation is spread around our READMEs in our codebase and in
the [PyTorch Developer Wiki](https://github.com/pytorch/pytorch/wiki).
If you're interested in adding new developer docs, please read this [page on the wiki](https://github.com/pytorch/pytorch/wiki/Where-or-how-should-I-add-documentation) on our best practices for where to put it.

The rest of this section is about user-facing documentation.

PyTorch uses [Google style](https://www.sphinx-doc.org/en/master/usage/extensions/example_google.html)
for formatting docstrings. Each line inside a docstrings block must be limited to 80 characters so that it fits into Jupyter documentation popups.


### Docstring type formatting

In addition to the standard Google Style docstring formatting rules, the following guidelines should be followed for docstring types (docstring types are the type information contained in the round brackets after the variable name):

* The "`Callable`", "`Any`", "`Iterable`", "`Iterator`", "`Generator`" types should have their first letter capitalized.

* The "`list`" and "`tuple`" types should be completely lowercase.

* Types should not be made plural. For example: `tuple of int` should be used instead of `tuple of ints`.

* The only acceptable delimiter words for types are `or` and `of`. No other non-type words should be used other than `optional`.

* The word `optional` should only be used after the types, and it is only used if the user does not have to specify a value for the variable. Default values are listed after the variable description. Example:

    ```
    my_var (int, optional): Variable description. Default: 1
    ```

* Basic Python types should match their type name so that the [Intersphinx](https://www.sphinx-doc.org/en/master/usage/extensions/intersphinx.html) extension can correctly identify them. For example:
    * Use `str` instead of `string`.
    * Use `bool` instead of `boolean`.
    * Use `dict` instead of `dictionary`.

* Square brackets should be used for the dictionary type. For example:

    ```
    my_var (dict[str, int]): Variable description.
    ```

* If a variable has two different possible types, then the word `or` should be used without a comma. Otherwise variables with 3 or more types should use commas to separate the types. Example:

    ```
    x (type1 or type2): Variable description.
    y (type1, type2, or type3): Variable description.
    ```


### Building documentation

Note that the docs will only build with Python versions <3.13. To build the documentation:

1. Build and install PyTorch

2. Install the prerequisites

```bash
cd docs
pip install -r requirements.txt
# `katex` must also be available in your PATH.
# You can either install katex globally if you have properly configured npm:
# npm install -g katex
# Or if you prefer an uncontaminated global executable environment or do not want to go through the node configuration:
# npm install katex && export PATH="$PATH:$(pwd)/node_modules/.bin"
```
> Note: if you installed `nodejs` with a different package manager then `npm` will probably install a version of `katex` that is not
compatible with your version of `nodejs` and doc builds will fail.
A combination of versions that is known to work is `node@6.13.1` and
`katex@0.13.18`. To install the latter with `npm` you can run
```npm install -g katex@0.13.18```


> Note that if you are a Facebook employee using a devserver, yarn may be more convenient to install katex:

```bash
yarn global add katex
```
> If a specific version is required you can use for example `yarn global add katex@0.13.18`.

3. Generate the documentation HTML files. The generated files will be in `docs/build/html`.

```bash
make html
```

#### Tips

The `.rst` source files live in [docs/source](docs/source). Some of the `.rst`
files pull in docstrings from PyTorch Python code (for example, via
the `autofunction` or `autoclass` directives). To vastly shorten doc build times,
it is helpful to remove the files you are not working on, only keeping the base
`index.rst` file and the files you are editing. The Sphinx build will produce
missing file warnings but will still complete. For example, to work on `jit.rst`:

```bash
cd docs/source
find . -type f | grep rst | grep -v index | grep -v jit | xargs rm

# Make your changes, build the docs, etc.

# Don't commit the deletions!
git add index.rst jit.rst
...
```

#### Building C++ Documentation

For C++ documentation (https://pytorch.org/cppdocs), we use
[Doxygen](http://www.doxygen.nl/) and then convert it to
[Sphinx](http://www.sphinx-doc.org/) via
[Breathe](https://github.com/michaeljones/breathe) and
[Exhale](https://github.com/svenevs/exhale). Check the [Doxygen
reference](https://www.doxygen.nl/manual/) for more
information on the documentation syntax.

We run Doxygen in CI (Travis) to verify that you do not use invalid Doxygen
commands. To run this check locally, run `./check-doxygen.sh` from inside
`docs/cpp/source`.

To build the documentation, follow the same steps as above, but run them from
`docs/cpp` instead of `docs`.

### Previewing changes locally

To view HTML files locally, you can open the files in your web browser. For example,
navigate to `file:///your_pytorch_folder/docs/build/html/index.html` in a web
browser.

If you are developing on a remote machine, you can set up an SSH tunnel so that
you can access the HTTP server on the remote machine from your local machine. To map
remote port 8000 to local port 8000, use either of the following commands.

```bash
# For SSH
ssh my_machine -L 8000:my_machine:8000

# For Eternal Terminal
et my_machine -t="8000:8000"
```

Then navigate to `localhost:8000` in your web browser.

**Tip:**
You can start a lightweight HTTP server on the remote machine with:

```bash
python -m http.server 8000 <path_to_html_output>
```

Alternatively, you can run `rsync` on your local machine to copy the files from
your remote machine:

```bash
mkdir -p build cpp/build
rsync -az me@my_machine:/path/to/pytorch/docs/build/html build
rsync -az me@my_machine:/path/to/pytorch/docs/cpp/build/html cpp/build
```

### Previewing documentation on PRs

PyTorch will host documentation previews at `https://docs-preview.pytorch.org/pytorch/pytorch/<pr number>/index.html` once the docs GitHub Actions job has completed on your PR. You can find its link in the automated pytorchbot comment on your PR or go to the URL
directly.

### Adding documentation tests

It is easy for code snippets in docstrings and `.rst` files to get out of date. The docs
build includes the [Sphinx Doctest Extension](https://www.sphinx-doc.org/en/master/usage/extensions/doctest.html),
which can run code in documentation as a unit test. To use the extension, use
the `.. testcode::` directive in your `.rst` and docstrings.

To manually run these tests, follow steps 1 and 2 above, then run:

```bash
cd docs
make doctest
```

## Profiling with `py-spy`

Evaluating the performance impact of code changes in PyTorch can be complicated,
particularly if code changes happen in compiled code. One simple way to profile
both Python and C++ code in PyTorch is to use
[`py-spy`](https://github.com/benfred/py-spy), a sampling profiler for Python
that has the ability to profile native code and Python code in the same session.

`py-spy` can be installed via `pip`:

```bash
pip install py-spy
```

To use `py-spy`, first write a Python test script that exercises the
functionality you would like to profile. For example, this script profiles
`torch.add`:

```python
import torch

t1 = torch.tensor([[1, 1], [1, 1.]])
t2 = torch.tensor([[0, 0], [0, 0.]])

for _ in range(1000000):
    torch.add(t1, t2)
```

Since the `torch.add` operation happens in microseconds, we repeat it a large
number of times to get good statistics. The most straightforward way to use
`py-spy` with such a script is to generate a [flame
graph](http://www.brendangregg.com/flamegraphs.html):

```bash
py-spy record -o profile.svg --native -- python test_tensor_tensor_add.py
```

This will output a file named `profile.svg` containing a flame graph you can
view in a web browser or SVG viewer. Individual stack frame entries in the graph
can be selected interactively with your mouse to zoom in on a particular part of
the program execution timeline. The `--native` command-line option tells
`py-spy` to record stack frame entries for PyTorch C++ code. To get line numbers
for C++ code it may be necessary to compile PyTorch in debug mode by prepending
your `python -m pip install -e . -v --no-build-isolation` call to compile
PyTorch with `DEBUG=1`. Depending on your operating system it may also be
necessary to run `py-spy` with root privileges.

`py-spy` can also work in an `htop`-like "live profiling" mode and can be
tweaked to adjust the stack sampling rate, see the `py-spy` readme for more
details.

## Managing multiple build trees

One downside to using `python -m pip install -e . -v --no-build-isolation` is
that your development version of PyTorch will be installed globally on your
account (e.g., if you run `import torch` anywhere else, the development version
will be used).

If you want to manage multiple builds of PyTorch, you can make use of
[venv environments](https://docs.python.org/3/library/venv.html) to maintain
separate Python package environments, each of which can be tied to a
specific build of PyTorch. To set one up:

```bash
python -m venv pytorch-myfeature
source pytorch-myfeature/bin/activate  # or `& .\pytorch-myfeature\Scripts\Activate.ps1` on Windows
# if you run python now, torch will NOT be installed
python -m pip install --no-build-isolation -v -e .
```

## C++ development tips

If you are working on the C++ code, there are a few important things that you
will want to keep in mind:

1. How to rebuild only the code you are working on.
2. How to make rebuilds in the absence of changes go faster.

### Build only what you need

`python setup.py build` will build everything by default, but sometimes you are
only interested in a specific component.

- Working on a test binary? Run `(cd build && ninja bin/test_binary_name)` to
  rebuild only that test binary (without rerunning cmake). (Replace `ninja` with
  `make` if you don't have ninja installed).

On the initial build, you can also speed things up with the environment
variables `DEBUG`, `USE_DISTRIBUTED`, `USE_MKLDNN`, `USE_CUDA`, `USE_FLASH_ATTENTION`, `USE_MEM_EFF_ATTENTION`, `BUILD_TEST`, `USE_FBGEMM`, `USE_NNPACK` and `USE_QNNPACK`.

- `DEBUG=1` will enable debug builds (-g -O0)
- `REL_WITH_DEB_INFO=1` will enable debug symbols with optimizations (-g -O3)
- `USE_DISTRIBUTED=0` will disable distributed (c10d, gloo, mpi, etc.) build.
- `USE_MKLDNN=0` will disable using MKL-DNN.
- `USE_CUDA=0` will disable compiling CUDA (in case you are developing on something not CUDA related), to save compile time.
- `BUILD_TEST=0` will disable building C++ test binaries.
- `USE_FBGEMM=0` will disable using FBGEMM (quantized 8-bit server operators).
- `USE_NNPACK=0` will disable compiling with NNPACK.
- `USE_QNNPACK=0` will disable QNNPACK build (quantized 8-bit operators).
- `USE_XNNPACK=0` will disable compiling with XNNPACK.
- `USE_FLASH_ATTENTION=0` and `USE_MEM_EFF_ATTENTION=0` will disable compiling flash attention and memory efficient kernels respectively

For example:

```bash
DEBUG=1 USE_DISTRIBUTED=0 USE_MKLDNN=0 USE_CUDA=0 BUILD_TEST=0 \
    USE_FBGEMM=0 USE_NNPACK=0 USE_QNNPACK=0 USE_XNNPACK=0 \
    python -m pip install --no-build-isolation -v -e .
```

For subsequent builds (i.e., when `build/CMakeCache.txt` exists), the build
options passed for the first time will persist; please run `ccmake build/`, run
`cmake-gui build/`, or directly edit `build/CMakeCache.txt` to adapt build
options.

### Code completion and IDE support

When using `python -m pip install -e . -v --no-build-isolation`, PyTorch will generate
a `compile_commands.json` file that can be used by many editors
to provide command completion and error highlighting for PyTorch's
C++ code. You need to `pip install ninja` to generate accurate
information for the code in `torch/csrc`. More information at:
- https://sarcasm.github.io/notes/dev/compilation-database.html

### Make no-op build fast

#### Use Ninja

By default, cmake will use its Makefile generator to generate your build
system.  You can get faster builds if you install the ninja build system
with `pip install ninja`.  If PyTorch was already built, you will need
to run `python setup.py clean` once after installing ninja for builds to
succeed.

Note: Make sure to use a machine with a larger number of CPU cores, this will significantly reduce your build times.

#### Use CCache

Even when dependencies are tracked with file modification, there are many
situations where files get rebuilt when a previous compilation was exactly the
same. Using ccache in a situation like this is a real time-saver.

Before building pytorch, install ccache from your package manager of choice:

```bash
sudo apt install ccache
sudo yum install ccache
brew install ccache
```

You may also find the default cache size in ccache is too small to be useful.
The cache sizes can be increased from the command line:

```bash
# config: cache dir is ~/.ccache, conf file ~/.ccache/ccache.conf
# max size of cache
ccache -M 25Gi  # -M 0 for unlimited
# unlimited number of files
ccache -F 0
```

To check this is working, do two clean builds of pytorch in a row. The second
build should be substantially and noticeably faster than the first build. If
this doesn't seem to be the case, check the `CMAKE_<LANG>_COMPILER_LAUNCHER`
rules in `build/CMakeCache.txt`, where `<LANG>` is `C`, `CXX` and `CUDA`.
Each of these 3 variables should contain ccache, e.g.

```
//CXX compiler launcher
CMAKE_CXX_COMPILER_LAUNCHER:STRING=/usr/bin/ccache
```

If not, you can define these variables on the command line before invoking `setup.py`.

```bash
export CMAKE_C_COMPILER_LAUNCHER=ccache
export CMAKE_CXX_COMPILER_LAUNCHER=ccache
export CMAKE_CUDA_COMPILER_LAUNCHER=ccache
python -m pip install --no-build-isolation -v -e .
```

#### Use a faster linker

If you are editing a single file and rebuilding in a tight loop, the time spent linking will dominate. The system linker available in most Linux distributions (GNU `ld`) is quite slow. To improve build times, consider using a faster linker such as [mold](https://github.com/rui314/mold) or [lld](https://lld.llvm.org/).

- **mold**: A modern, high-performance linker that significantly reduces linking time. It is typically available via package managers like `apt` or `yum`. Note that `mold` requires GCC version 12 or higher.
- **lld**: A fast linker from the LLVM project. The easiest way to get `lld` is from a [LLVM release](https://releases.llvm.org/download.html).

Starting with CMake 3.29, you can specify the linker type using the [`CMAKE_LINKER_TYPE`](https://cmake.org/cmake/help/latest/variable/CMAKE_LINKER_TYPE.html) variable. For example, with `mold` installed:

```sh
CMAKE_LINKER_TYPE=MOLD python -m pip install --no-build-isolation -v -e .
```

#### Use pre-compiled headers

Sometimes there's no way of getting around rebuilding lots of files, for example
editing `native_functions.yaml` usually means 1000+ files being rebuilt. If
you're using CMake newer than 3.16, you can enable pre-compiled headers by
setting `USE_PRECOMPILED_HEADERS=1` either on first setup, or in the
`CMakeCache.txt` file.

```sh
USE_PRECOMPILED_HEADERS=1 python -m pip install --no-build-isolation -v -e .
```

This adds a build step where the compiler takes `<ATen/ATen.h>` and essentially
dumps its internal AST to a file so the compiler can avoid repeating itself for
every `.cpp` file.

One caveat is that when enabled, this header gets included in every file by default.
Which may change what code is legal, for example:
- internal functions can never alias existing names in `<ATen/ATen.h>`
- names in `<ATen/ATen.h>` will work even if you don't explicitly include it.

#### Workaround for header dependency bug in nvcc
If re-building without modifying any files results in several CUDA files being
re-compiled, you may be running into an `nvcc` bug where header dependencies are
not converted to absolute paths before reporting it to the build system. This
makes `ninja` think one of the header files has been deleted, so it runs the
build again.

A compiler-wrapper to fix this is provided in `tools/nvcc_fix_deps.py`. You can use
this as a compiler launcher, similar to `ccache`
```bash
export CMAKE_CUDA_COMPILER_LAUNCHER="python;`pwd`/tools/nvcc_fix_deps.py;ccache"
python -m pip install --no-build-isolation -v -e .
```

### Rebuild few files with debug information

While debugging a problem one often had to maintain a debug build in a separate folder.
But often only a few files needs to be rebuild with debug info to get a symbolicated backtrace or enable source debugging
One can easily solve this with the help of `tools/build_with_debinfo.py`

For example, suppose one wants to debug what is going on while tensor index is selected, which can be achieved by setting a breakpoint at `applySelect` function:
```
% lldb -o "b applySelect" -o "process launch" -- python3 -c "import torch;print(torch.rand(5)[3])"
(lldb) target create "python"
Current executable set to '/usr/bin/python3' (arm64).
(lldb) settings set -- target.run-args  "-c" "import torch;print(torch.rand(5)[3])"
(lldb) b applySelect
Breakpoint 1: no locations (pending).
WARNING:  Unable to resolve breakpoint to any actual locations.
(lldb) process launch
2 locations added to breakpoint 1
Process 87729 stopped
* thread #1, queue = 'com.apple.main-thread', stop reason = breakpoint 1.1
    frame #0: 0x00000001023d55a8 libtorch_python.dylib`at::indexing::impl::applySelect(at::Tensor const&, long long, c10::SymInt, long long, c10::Device const&, std::__1::optional<c10::ArrayRef<c10::SymInt>> const&)
libtorch_python.dylib`at::indexing::impl::applySelect:
->  0x1023d55a8 <+0>:  sub    sp, sp, #0xd0
    0x1023d55ac <+4>:  stp    x24, x23, [sp, #0x90]
    0x1023d55b0 <+8>:  stp    x22, x21, [sp, #0xa0]
    0x1023d55b4 <+12>: stp    x20, x19, [sp, #0xb0]
Target 0: (python) stopped.
Process 87729 launched: '/usr/bin/python' (arm64)
```
Which is not very informative, but can be easily remedied by rebuilding `python_variable_indexing.cpp` with debug information
```
% ./tools/build_with_debinfo.py torch/csrc/autograd/python_variable_indexing.cpp
[1 / 2] Building caffe2/torch/CMakeFiles/torch_python.dir/csrc/autograd/python_variable_indexing.cpp.o
[2 / 2] Building lib/libtorch_python.dylib
```
And afterwards:
```
% lldb -o "b applySelect" -o "process launch" -- python3 -c "import torch;print(torch.rand(5)[3])"
(lldb) target create "python"
Current executable set to '/usr/bin/python3' (arm64).
(lldb) settings set -- target.run-args  "-c" "import torch;print(torch.rand(5)[3])"
(lldb) b applySelect
Breakpoint 1: no locations (pending).
WARNING:  Unable to resolve breakpoint to any actual locations.
(lldb) process launch
2 locations added to breakpoint 1
Process 87741 stopped
* thread #1, queue = 'com.apple.main-thread', stop reason = breakpoint 1.1
    frame #0: 0x00000001024e2628 libtorch_python.dylib`at::indexing::impl::applySelect(self=0x00000001004ee8a8, dim=0, index=(data_ = 3), real_dim=0, (null)=0x000000016fdfe535, self_sizes= Has Value=true ) at TensorIndexing.h:239:7
   236         const at::Device& /*self_device*/,
   237         const std::optional<SymIntArrayRef>& self_sizes) {
   238       // See NOTE [nested tensor size for indexing]
-> 239       if (self_sizes.has_value()) {
   240         auto maybe_index = index.maybe_as_int();
   241         if (maybe_index.has_value()) {
   242           TORCH_CHECK_INDEX(
Target 0: (python) stopped.
Process 87741 launched: '/usr/bin/python3' (arm64)
```
Which is much more useful, isn't it?

### C++ frontend development tips

We have very extensive tests in the [test/cpp/api](test/cpp/api) folder. The
tests are a great way to see how certain components are intended to be used.
When compiling PyTorch from source, the test runner binary will be written to
`build/bin/test_api`. The tests use the [GoogleTest](https://github.com/google/googletest/blob/master/googletest)
framework, which you can read up about to learn how to configure the test runner. When
submitting a new feature, we care very much that you write appropriate tests.
Please follow the lead of the other tests to see how to write a new test case.

### GDB integration

If you are debugging pytorch inside GDB, you might be interested in
[pytorch-gdb](tools/gdb/pytorch-gdb.py). This script introduces some
pytorch-specific commands which you can use from the GDB prompt. In
particular, `torch-tensor-repr` prints a human-readable repr of an at::Tensor
object. Example of usage:

```
$ gdb python
GNU gdb (GDB) 9.2
[...]
(gdb) # insert a breakpoint when we call .neg()
(gdb) break at::Tensor::neg
Function "at::Tensor::neg" not defined.
Make breakpoint pending on future shared library load? (y or [n]) y
Breakpoint 1 (at::Tensor::neg) pending.

(gdb) run
[...]
>>> import torch
>>> t = torch.tensor([1, 2, 3, 4], dtype=torch.float64)
>>> t
tensor([1., 2., 3., 4.], dtype=torch.float64)
>>> t.neg()

Thread 1 "python" hit Breakpoint 1, at::Tensor::neg (this=0x7ffb118a9c88) at aten/src/ATen/core/TensorBody.h:3295
3295    inline at::Tensor Tensor::neg() const {
(gdb) # the default repr of 'this' is not very useful
(gdb) p this
$1 = (const at::Tensor * const) 0x7ffb118a9c88
(gdb) p *this
$2 = {impl_ = {target_ = 0x55629b5cd330}}
(gdb) torch-tensor-repr *this
Python-level repr of *this:
tensor([1., 2., 3., 4.], dtype=torch.float64)
```

GDB tries to automatically load `pytorch-gdb` thanks to the
[.gdbinit](.gdbinit) at the root of the pytorch repo. However, auto-loadings is disabled by default, because of security reasons:

```bash
$ gdb
warning: File "/path/to/pytorch/.gdbinit" auto-loading has been declined by your `auto-load safe-path' set to "$debugdir:$datadir/auto-load".
To enable execution of this file add
        add-auto-load-safe-path /path/to/pytorch/.gdbinit
line to your configuration file "/home/YOUR-USERNAME/.gdbinit".
To completely disable this security protection add
        set auto-load safe-path /
line to your configuration file "/home/YOUR-USERNAME/.gdbinit".
For more information about this security protection see the
"Auto-loading safe path" section in the GDB manual.  E.g., run from the shell:
        info "(gdb)Auto-loading safe path"
(gdb)
```

As gdb itself suggests, the best way to enable auto-loading of `pytorch-gdb`
is to add the following line to your `~/.gdbinit` (i.e., the `.gdbinit` file
which is in your home directory, **not** `/path/to/pytorch/.gdbinit`):

```bash
add-auto-load-safe-path /path/to/pytorch/.gdbinit
```

### C++ stacktraces
Set `TORCH_SHOW_CPP_STACKTRACES=1` to get the C++ stacktrace when an error occurs in Python.

## CUDA development tips

If you are working on the CUDA code, here are some useful CUDA debugging tips:

1. `CUDA_DEVICE_DEBUG=1` will enable CUDA device function debug symbols (`-g -G`).
    This will be particularly helpful in debugging device code. However, it will
    slow down the build process for about 50% (compared to only `DEBUG=1`), so use wisely.
2. `cuda-gdb` and `cuda-memcheck` are your best CUDA debugging friends. Unlike`gdb`,
   `cuda-gdb` can display actual values in a CUDA tensor (rather than all zeros).
3. CUDA supports a lot of C++11/14 features such as, `std::numeric_limits`, `std::nextafter`,
   `std::tuple` etc. in device code. Many of such features are possible because of the
   [--expt-relaxed-constexpr](https://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html#constexpr-functions)
   nvcc flag. There is a known [issue](https://github.com/ROCm/hip/issues/374)
   that ROCm errors out on device code, which uses such stl functions.
4. A good performance metric for a CUDA kernel is the
   [Effective Memory Bandwidth](https://devblogs.nvidia.com/how-implement-performance-metrics-cuda-cc/).
   It is useful for you to measure this metric whenever you are writing/optimizing a CUDA
   kernel. Following script shows how we can measure the effective bandwidth of CUDA `uniform_`
   kernel.
   ```python
   import torch
   from torch.utils.benchmark import Timer
   size = 128*512
   nrep = 100
   nbytes_read_write = 4 # this is number of bytes read + written by a kernel. Change this to fit your kernel.

   for i in range(10):
       a=torch.empty(size).cuda().uniform_()
       torch.cuda.synchronize()
       out = a.uniform_()
       torch.cuda.synchronize()
       t = Timer(stmt="a.uniform_()", globals=globals())
       res = t.blocked_autorange()
       timec = res.median
       print("uniform, size, elements", size, "forward", timec, "bandwidth (GB/s)", size*(nbytes_read_write)*1e-9/timec)
       size *=2
   ```

  See more cuda development tips [here](https://github.com/pytorch/pytorch/wiki/CUDA-basics)

## Windows development tips

For building from source on Windows, consult
[our documentation](https://pytorch.org/docs/stable/notes/windows.html) on it.

Occasionally, you will write a patch which works on Linux, but fails CI on Windows.
There are a few aspects in which MSVC (the Windows compiler toolchain we use) is stricter
than Linux, which are worth keeping in mind when fixing these problems.

1. Symbols are NOT exported by default on Windows; instead, you have to explicitly
   mark a symbol as exported/imported in a header file with `__declspec(dllexport)` /
   `__declspec(dllimport)`. We have codified this pattern into a set of macros
   which follow the convention `*_API`, e.g., `TORCH_API` inside Caffe2, Aten and Torch.
   (Every separate shared library needs a unique macro name, because symbol visibility
   is on a per shared library basis. See c10/macros/Macros.h for more details.)

   The upshot is if you see an "unresolved external" error in your Windows build, this
   is probably because you forgot to mark a function with `*_API`. However, there is
   one important counterexample to this principle: if you want a *templated* function
   to be instantiated at the call site, do NOT mark it with `*_API` (if you do mark it,
   you'll have to explicitly instantiate all of the specializations used by the call
   sites.)

2. If you link against a library, this does not make its dependencies transitively
   visible. You must explicitly specify a link dependency against every library whose
   symbols you use. (This is different from Linux where in most environments,
   transitive dependencies can be used to fulfill unresolved symbols.)

3. If you have a Windows box (we have a few on EC2 which you can request access to) and
   you want to run the build, the easiest way is to just run `.ci/pytorch/win-build.sh`.
   If you need to rebuild, run `REBUILD=1 .ci/pytorch/win-build.sh`.

Even if you don't know anything about MSVC, you can use cmake to build simple programs on
Windows; this can be helpful if you want to learn more about some peculiar linking behavior
by reproducing it on a small example. Here's a simple example cmake file that defines
two dynamic libraries, one linking with the other:

```CMake
project(myproject CXX)
set(CMAKE_CXX_STANDARD 14)
add_library(foo SHARED foo.cpp)
add_library(bar SHARED bar.cpp)
# NB: don't forget to __declspec(dllexport) at least one symbol from foo,
# otherwise foo.lib will not be created.
target_link_libraries(bar PUBLIC foo)
```

You can build it with:

```bash
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

We've found the most effective way to debug these problems is to
carefully read over diffs, keeping in mind known bugs in MSVC/NVCC.
Here are a few well known pitfalls and workarounds:

* This is not actually a bug per se, but in general, code generated by MSVC
  is more sensitive to memory errors; you may have written some code
  that does a use-after-free or stack overflows; on Linux the code
  might work, but on Windows your program will crash. ASAN may not
  catch all of these problems: stay vigilant to the possibility that
  your crash is due to a real memory problem.

* `constexpr` generally works less well on MSVC.

  * The idiom `static_assert(f() == f())` to test if `f` is constexpr
    does not work; you'll get "error C2131: expression did not evaluate
    to a constant". Don't use these asserts on Windows.
    (Example: `c10/util/intrusive_ptr.h`)

* (NVCC) Code you access inside a `static_assert` will eagerly be
  evaluated as if it were device code, and so you might get an error
  that the code is "not accessible".

```cpp
class A {
  static A singleton_;
  static constexpr inline A* singleton() {
    return &singleton_;
  }
};
static_assert(std::is_same(A*, decltype(A::singleton()))::value, "hmm");
```

* The compiler will run out of heap space if you attempt to compile files that
  are too large. Splitting such files into separate files helps.
  (Example: `THTensorMath`, `THTensorMoreMath`, `THTensorEvenMoreMath`.)

* MSVC's preprocessor (but not the standard compiler) has a bug
  where it incorrectly tokenizes raw string literals, ending when it sees a `"`.
  This causes preprocessor tokens inside the literal like an`#endif`  to be incorrectly
  treated as preprocessor directives. See https://godbolt.org/z/eVTIJq as an example.

* Either MSVC or the Windows headers have a PURE macro defined and will replace
  any occurrences of the PURE token in code with an empty string. This is why
  we have AliasAnalysisKind::PURE_FUNCTION and not AliasAnalysisKind::PURE.
  The same is likely true for other identifiers that we just didn't try to use yet.

### Building on legacy code and CUDA

CUDA, MSVC, and PyTorch versions are interdependent; please install matching versions from this table:
| CUDA version | Newest supported VS version                             | PyTorch version |
| ------------ | ------------------------------------------------------- | --------------- |
| 10.1         | Visual Studio 2019 (16.X) (`_MSC_VER` < 1930)           |  1.3.0 ~ 1.7.0  |
| 10.2         | Visual Studio 2019 (16.X) (`_MSC_VER` < 1930)           |  1.5.0 ~ 1.7.0  |
| 11.0         | Visual Studio 2019 (16.X) (`_MSC_VER` < 1930)           |      1.7.0      |

Note: There's a [compilation issue](https://github.com/uxlfoundation/oneDNN/issues/812) in several Visual Studio 2019 versions since 16.7.1, so please make sure your Visual Studio 2019 version is not in 16.7.1 ~ 16.7.5

## Pre-commit tidy/linting hook

We use clang-tidy to perform additional
formatting and semantic checking of code. We provide a pre-commit git hook for
performing these checks, before a commit is created:

  ```bash
  ln -s ../../tools/git-pre-commit .git/hooks/pre-commit
  ```

If you have already committed files and
CI reports `flake8` errors, you can run the check locally in your PR branch with:

  ```bash
  flake8 $(git diff --name-only $(git merge-base --fork-point main))
  ```

You'll need to install an appropriately configured flake8; see
[Lint as you type](https://github.com/pytorch/pytorch/wiki/Lint-as-you-type)
for documentation on how to do this.

Fix the code so that no errors are reported when you re-run the above check again,
and then commit the fix.

## Building PyTorch with ASAN

[ASAN](https://github.com/google/sanitizers/wiki/AddressSanitizer) is very
useful for debugging memory errors in C++. We run it in CI, but here's how to
get the same thing to run on your local machine.

First, install LLVM 8. The easiest way is to get [prebuilt
binaries](http://releases.llvm.org/download.html#8.0.0) and extract them to
folder (later called `$LLVM_ROOT`).

Then set up the appropriate scripts. You can put this in your `.bashrc`:

```bash
LLVM_ROOT=<wherever your llvm install is>
PYTORCH_ROOT=<wherever your pytorch checkout is>

LIBASAN_RT="$LLVM_ROOT/lib/clang/8.0.0/lib/linux/libclang_rt.asan-x86_64.so"
build_with_asan()
{
  LD_PRELOAD=${LIBASAN_RT} \
  CC="$LLVM_ROOT/bin/clang" \
  CXX="$LLVM_ROOT/bin/clang++" \
  LDSHARED="clang --shared" \
  LDFLAGS="-stdlib=libstdc++" \
  CFLAGS="-fsanitize=address -fno-sanitize-recover=all -shared-libasan -pthread" \
  CXX_FLAGS="-pthread" \
  USE_CUDA=0 USE_OPENMP=0 USE_DISTRIBUTED=0 DEBUG=1 \
  python -m pip install --no-build-isolation -v -e .
}

run_with_asan()
{
  LD_PRELOAD=${LIBASAN_RT} $@
}

# you can look at build-asan.sh to find the latest options the CI uses
export ASAN_OPTIONS=detect_leaks=0:symbolize=1:strict_init_order=true
export UBSAN_OPTIONS=print_stacktrace=1:suppressions=$PYTORCH_ROOT/ubsan.supp
export ASAN_SYMBOLIZER_PATH=$LLVM_ROOT/bin/llvm-symbolizer
```

Then you can use the scripts like:

```
suo-devfair ~/pytorch ❯ build_with_asan
suo-devfair ~/pytorch ❯ run_with_asan python test/test_jit.py
```

### Getting `ccache` to work

The scripts above specify the `clang` and `clang++` binaries directly, which
bypasses `ccache`. Here's how to get `ccache` to work:

1. Make sure the ccache symlinks for `clang` and `clang++` are set up (see
   CONTRIBUTING.md)
2. Make sure `$LLVM_ROOT/bin` is available on your `$PATH`.
3. Change the `CC` and `CXX` variables in `build_with_asan()` to point
   directly to `clang` and `clang++`.

### Why this stuff with `LD_PRELOAD` and `LIBASAN_RT`?

The “standard” workflow for ASAN assumes you have a standalone binary:

1. Recompile your binary with `-fsanitize=address`.
2. Run the binary, and ASAN will report whatever errors it find.

Unfortunately, PyTorch is a distributed as a shared library that is loaded by
a third-party executable (Python). It’s too much of a hassle to recompile all
of Python every time we want to use ASAN. Luckily, the ASAN folks have a
workaround for cases like this:

1. Recompile your library with `-fsanitize=address -shared-libasan`. The
   extra `-shared-libasan` tells the compiler to ask for the shared ASAN
   runtime library.
2. Use `LD_PRELOAD` to tell the dynamic linker to load the ASAN runtime
   library before anything else.

More information can be found
[here](https://github.com/google/sanitizers/wiki/AddressSanitizerAsDso).

### Why LD_PRELOAD in the build function?

We need `LD_PRELOAD` because there is a cmake check that ensures that a
simple program builds and runs. If we are building with ASAN as a shared
library, we need to `LD_PRELOAD` the runtime library, otherwise there will
dynamic linker errors and the check will fail.

We don’t actually need either of these if we fix the cmake checks.

### Why no leak detection?

Python leaks a lot of memory. Possibly we could configure a suppression file,
but we haven’t gotten around to it.

## Caffe2 notes

In 2018, we merged Caffe2 into the PyTorch source repository. While the
steady state aspiration is that Caffe2 and PyTorch share code freely,
in the meantime there will be some separation.

There are a few "unusual" directories which, for historical reasons,
are Caffe2/PyTorch specific. Here they are:

- `CMakeLists.txt`, `Makefile`, `binaries`, `cmake`, `modules`,
  `scripts` are Caffe2-specific. Don't put PyTorch code in them without
  extra coordination.

- `mypy*`, `requirements.txt`, `setup.py`, `test`, `tools` are
  PyTorch-specific. Don't put Caffe2 code in them without extra
  coordination.

## CI failure tips

Once you submit a PR or push a new commit to a branch that is in
an active PR, CI jobs will be run automatically. Some of these may
fail and you will need to find out why, by looking at the logs.

Fairly often, a CI failure might be unrelated to your changes. You can
confirm by going to our [HUD](https://hud.pytorch.org) and seeing if the CI job
is failing upstream already. In this case, you
can usually ignore the failure. See [the following
subsection](#which-commit-is-used-in-ci) for more details.

Some failures might be related to specific hardware or environment
configurations. In this case, if you're a Meta employee, you can ssh into
the job's session to perform manual debugging following the instructions in
our [CI wiki](https://github.com/pytorch/pytorch/wiki/Debugging-using-with-ssh-for-Github-Actions).


### Which commit is used in CI?

For CI run on `main`, this repository is checked out for a given `main`
commit, and CI is run on that commit (there isn't really any other choice).

For PRs, however, it's a bit more complicated. Consider this commit graph, where
`main` is at commit `A`, and the branch for PR #42 (just a placeholder) is at
commit `B`:

```
       o---o---B (refs/pull/42/head)
      /         \
     /           C (refs/pull/42/merge)
    /           /
---o---o---o---A (merge-destination) - usually main
```

There are two possible choices for which commit to use:

1. Checkout commit `B`, the head of the PR (manually committed by the PR
   author).
2. Checkout commit `C`, the hypothetical result of what would happen if the PR
   were merged into its destination (usually `main`).

For all practical purposes, most people can think of the commit being used as
commit `B` (choice **1**).

However, if workflow files (which govern CI behavior) were modified (either by your PR or since dev branch were created ) there's
a nuance to know about:
The workflow files themselves get taken from checkpoint `C`, the merger of your
PR and the `main` branch. But only the workflow files get taken from that merged
checkpoint. Everything else (tests, code, etc) all get taken directly from your
PR's commit (commit `B`). Please note, this scenario would never affect PRs authored by `ghstack` as they would not automatically ingest the updates from default branch.


## Dev Infra Office Hours
[Dev Infra Office Hours](https://github.com/pytorch/pytorch/wiki/Dev-Infra-Office-Hours) are hosted every Friday to answer any questions regarding developer experience, Green HUD, and CI.
