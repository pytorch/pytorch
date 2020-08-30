# Code Coverage Tool for Pytorch

## Overview

This tool is designed for calculating code coverage for Pytorch project in both fbcode and oss. But it also goes beyond Pytorch, applying to other folders in fbcode.

It’s an integrated tool. You can use this tool to build, run, and generate both file-level and line-level report with both C++ tests and Python tests.

### Simple
* *Simple command to run:*
    * `python oss_coverage.py  `
* *Argument --clean will do all the messy clean up things for you*

### But Powerful

* *Choose your own interested folder*:
    * Choose the folder you want to collect coverage for
    * Flexible: default folder is good enough, but you can choose one or more other folders
* *Run only the test you want:*
    * use --run-only to run the tests you want
    * apply to both cpp and python tests
* *Final report:*
    * File-Level: The coverage for each file you are interested in
    * Line-Level: The coverage for each line in each file you are interested in
* *More complex but flexible options:*
    * Use different stages like --build, --run, --summary to achieve more flexible functionality

## How to use

This part will introduce about the arguments you can use when run this tool. The arguments are powerful, giving you full flexibility to do different work.
If you are not familiar with the procedure of generating code coverage report by using clang, read [Source-based Code Coverage](https://clang.llvm.org/docs/SourceBasedCodeCoverage.html) will be helpful.


## Examples

First step is to set some experimental value.
```
# pytorch folder, by default all the c++ binaries are in build/bin/
export PYTORCH_FOLDER=...
# make sure llvm-cov is available, by default it is /usr/local/opt/llvm/bin
export LLVM_TOOL_PATH=...
```

then command will run all the tests in `build/bin/` and `test/` folder
```
python oss_coverage.py
```
Most times you don't want collect coverage for the entire Pytorch folder, use --interested-folder to report coverage only over the folder you want:
```
python oss_coverage.py --interested-folder=aten
```
Then, still in most cases, if you only run one or several test(s):
```
python oss_coverage.py --run-only=atest
python oss_coverage.py --run-only atest basic test_nn.py
```

### For more complex arguments and functionality
*To Be Done*
