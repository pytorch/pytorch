# Omniglot MAML examples

In this directory we've provided some examples of traning omniglot that reproduce the experiments from [the original MAML paper](https://arxiv.org/abs/1703.03400).

They can be run via `python {filename}`.

`maml-omniglot-higher.py` uses the [facebookresearch/higher](https://github.com/facebookresearch/higher) metalearning package and is the reference implementation. It runs all of its tasks sequentially.

`maml-omniglot-transforms.py` uses an experimental vmap (and functional grad) prototype. It runs all of its tasks in parallel. In theory this should lead to some speedups, but we haven't finished implementing all the rules for vmap that would actually make training faster.

`maml-omniglot-ptonly.py` is an implementation of `maml-omniglot-transforms.py` that runs all of its tasks sequentially (and also doesn't use the higher package).

The prototype vmap used for these experiments currently run off of a branch.
We'd love some feedback on the prototype and encourage folks to try it out.
It's a bit difficult to install, but here are some options:
1. If you're on the FAIR cluster, we can share a path to a conda environment
2. We are looking into building binaries using our branch and shipping them.
