# Code Coverage Tool for Pytorch

## Overview

This tool is designed for calculating code coverage for Pytorch project.
It’s an integrated tool. You can use this tool to run and generate both file-level and line-level report for C++ and Python tests. It will also be the tool we use in *CircleCI* to generate report for each master commit.

### Simple
* *Simple command to run:*
    * `python oss_coverage.py  `
* *Argument `--clean` will do all the messy clean up things for you*

### But Powerful

* *Choose your own interested folder*:
    * Default folder will be good enough in most times
    * Flexible: you can specify one or more folder(s) that you are interested in
* *Run only the test you want:*
    * By default it will run all the c++ and python tests
    * Flexible: you can specify one or more test(s) that you want to run
* *Final report:*
    * File-Level: The coverage percentage for each file you are interested in
    * Line-Level: The coverage details for each line in each file you are interested in
* *More complex but flexible options:*
    * Use different stages like *--run, --export, --summary* to achieve more flexible functionality

## How to use

This part will introduce about the arguments you can use when run this tool. The arguments are powerful, giving you full flexibility to do different work.
We have two different compilers, `gcc` and `clang`, and this tool supports both. But it is recommended to use `gcc` because it's much faster and use less disk place. The examples will also be divided to two parts, for `gcc` and `clang`.


## Examples
First step is to set some experimental value if needed:
```bash
# pytorch folder, by default all the c++ binaries are in build/bin/
export PYTORCH_FOLDER=...
# set compiler type
export COMPILER_TYPE="GCC" or export COMPILER_TYPE="CLANG"
# make sure llvm-cov is available, by default it is /usr/local/opt/llvm/bin
export LLVM_TOOL_PATH=...
```

then command will run all the tests in `build/bin/` and `test/` folder
```bash
python oss_coverage.py
```
Most times you don't want collect coverage for the entire Pytorch folder, use --interested-folder to report coverage only over the folder you want:
```bash
python oss_coverage.py --interested-folder=aten
```
Then, still in most cases, if you only run one or several test(s):
```bash
python oss_coverage.py --run-only=atest
python oss_coverage.py --run-only atest basic test_nn.py
```

### For more complex arguments and functionality
*To Be Done*

## Reference

For `gcc`
* See about how to invoke `gcov`, read [Invoking gcov](https://gcc.gnu.org/onlinedocs/gcc/Invoking-Gcov.html#Invoking-Gcov) will be helpful

For `clang`
* If you are not familiar with the procedure of generating code coverage report by using `clang`, read [Source-based Code Coverage](https://clang.llvm.org/docs/SourceBasedCodeCoverage.html) will be helpful.
