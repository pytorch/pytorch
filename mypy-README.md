# Optional type checking with mypy

Mypy is an optional static typechecker that works with Python 3.
To use it, install the following dependencies:

```bash
# Install dependencies
pip install mypy mypy-extensions

# Run type checker in the pytorch/ directory
mypy
```

Note that the minimum version of mypy that is supported is 0.770


## What we're aiming for

1. Complete type annotations for the whole code base, and shipping those in a
   PEP 561 compatible manner (adding a `py.typed` file so the installed package
   supports typechecking.
2. Inline type annotations for all Python code where possible, _except_ if
   there are too many overloads for functions/methods - in that case a stub
   file should be preferred (what's too many is a bit of a judgement call, I'd
   suggest three per function is a reasonable threshold). Another reason we may
   have to stay with stub files is if there's a mypy or code limitation (see
   for example gh-35566 about `nn.Module`).
3. Stub files for the extension modules (e.g. `torch._C`).
4. Good type annotation test coverage, by using `check_untyped_defs=True` for
   the test suite (or adding type annotations to tests).

`mypy.ini` does not get installed, it's only meant to facility development.
Therefore the end state of `mypy.ini` should be minimal and not include any
ignores, something like:

```
[mypy]
warn_unused_configs = True
warn_redundant_casts = True
show_error_codes = True
check_untyped_defs = True

files =
    torch,
    caffe2,
    test,
    aten/src/ATen/function_wrapper.py
```


## How to go about improving type annotations

_The tracking issue for this is
https://github.com/pytorch/pytorch/issues/16574_

### Setting up and checking mypy works

Before starting, install mypy (0.770 or newer), build PyTorch with `python
setup.py develop`, and run `mypy` in the root of the repo. This should give
output like:

```
Success: no issues found in 969 source files
```

In `mypy.ini` there's a long list of `ignore_missing_imports` and
`ignore_errors` for specific modules or files. If you remove one and re-run
`mypy`, then errors should appear. For example, deleting

```
[mypy-torch._C]
ignore_missing_imports = True
```

will show (currently):

```
...
torch/utils/data/_utils/signal_handling.py:39: error: Cannot find implementation or library stub for module named 'torch._C'  [import]
Found 14 errors in 14 files (checked 969 source files)
```

_Note that mypy caching can be flaky, in particular removing
`ignore_missing_imports` has a caching bug
(https://github.com/python/mypy/issues/7777). If you don't see any errors
appear, try `rm -rf .mypy_cache` and try again._

### Picking a task

Once the above works, pick a task by checking it off here, and adding your
GitHub username behind it. Once the task is complete, you can remove the task
from this page.

Stub files to be moved to inline annotations:

- [ ] `torch/__init__.pyi`
- [ ] `torch/autograd/__init__.pyi`
- [ ] `torch/autograd/grad_mode.pyi`
- [ ] `torch/cuda/__init__.pyi`
- [ ] `torch/nn/__init__.pyi`
- [ ] `torch/nn/common_types.pyi`
- [ ] `torch/nn/functional.pyi`
- [ ] `torch/nn/modules/__init__.pyi`
- [ ] `torch/nn/modules/activation.pyi`
- [ ] `torch/nn/modules/adaptive.pyi`
- [ ] `torch/nn/modules/batchnorm.pyi`
- [ ] `torch/nn/modules/container.pyi`
- [ ] `torch/nn/modules/conv.pyi`
- [ ] `torch/nn/modules/distance.pyi`
- [ ] `torch/nn/modules/dropout.pyi`
- [ ] `torch/nn/modules/flatten.pyi`
- [ ] `torch/nn/modules/fold.pyi`
- [ ] `torch/nn/modules/instancenorm.pyi`
- [ ] `torch/nn/modules/linear.pyi`
- [ ] `torch/nn/modules/loss.pyi`
- [ ] `torch/nn/modules/module.pyi`
- [ ] `torch/nn/modules/normalization.pyi`
- [ ] `torch/nn/modules/padding.pyi`
- [ ] `torch/nn/modules/pixelshuffle.pyi`
- [ ] `torch/nn/modules/pooling.pyi`
- [ ] `torch/nn/modules/rnn.pyi`
- [ ] `torch/nn/modules/sparse.pyi`
- [ ] `torch/nn/modules/transformer.pyi`
- [ ] `torch/nn/modules/upsampling.pyi`
- [ ] `torch/nn/parallel/__init__.pyi`
- [ ] `torch/nn/parallel/common_types.pyi`
- [ ] `torch/nn/parallel/data_parallel.pyi`
- [ ] `torch/nn/parallel/distributed.pyi`
- [ ] `torch/nn/parallel/parallel_apply.pyi`
- [ ] `torch/nn/parallel/replicate.pyi`
- [ ] `torch/nn/parallel/scatter_gather.pyi`
- [ ] `torch/nn/parameter.pyi`
- [ ] `torch/nn/utils/__init__.pyi`
- [ ] `torch/nn/utils/clip_grad.pyi`
- [ ] `torch/nn/utils/convert_parameters.pyi`
- [ ] `torch/nn/utils/rnn.pyi`
- [ ] `torch/nn/utils/spectral_norm.pyi`
- [ ] `torch/nn/utils/weight_norm.pyi`
- [ ] `torch/optim/__init__.pyi`
- [ ] `torch/optim/adadelta.pyi`
- [ ] `torch/optim/adagrad.pyi`
- [ ] `torch/optim/adam.pyi`
- [ ] `torch/optim/adamax.pyi`
- [ ] `torch/optim/adamw.pyi`
- [ ] `torch/optim/asgd.pyi`
- [ ] `torch/optim/lbfgs.pyi`
- [ ] `torch/optim/lr_scheduler.pyi`
- [ ] `torch/optim/optimizer.pyi`
- [ ] `torch/optim/rmsprop.pyi`
- [ ] `torch/optim/rprop.pyi`
- [ ] `torch/optim/sgd.pyi`
- [ ] `torch/optim/sparse_adam.pyi`
- [ ] `torch/optim/swa_utils.pyi`
- [ ] `torch/utils/__init__.pyi`
- [ ] `torch/utils/data/__init__.pyi`
- [ ] `torch/utils/data/dataloader.pyi`
- [ ] `torch/utils/data/dataset.pyi`
- [ ] `torch/utils/data/distributed.pyi`
- [ ] `torch/utils/data/sampler.pyi`
- [ ] `torch/utils/hooks.pyi`


### Scripts to regenerate the above task list

```
import os

import torch


# Assumes an in-place build to find all .pyi files
rootdir = os.path.dirname(os.path.abspath(torch.__file__))

filepaths = []
for subdir, dirs, files in os.walk(rootdir):
    for f in files:
        if f.endswith(".pyi"):
            os.path.join(subdir, f)
            filepaths.append(subdir + os.sep + f)

for filepath in sorted(filepaths):
    print("- [ ] `" + filepath[len(rootdir)-5:] + "`")
```
