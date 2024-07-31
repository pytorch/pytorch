# AutoHeuristic
AutoHeuristic is a framework that allows to use results from autotuning to learn a heuristic as a decision tree, that can be generated to code and shipped with compiler.

Each subdirectory contains the code necessary to collect data and learn a heuristic for `flex_attention`, `mixed_mm`, `mm`, and `pad_mm`.

## How to use AutoHeuristic
In general, the following steps have to performed:
- The AutoHeursitic constructor has to be called.
- A script that runs benchmarks in order to collect training data has to be implemented.
- The train_decision.py (if you want to learn a decision tree) or train_regression.py (if you want to learn a regression tree) script has to be run in order to learn the heuristic and generate it to code.

## Where has AutoHeuristic already been used?
Take a look at the following PRs in which AutoHeuristic has enabled for various optimizations.
Looking at these examples may be helpful if you want to use AutoHeuristic yourself.
- pad_mm: https://github.com/pytorch/pytorch/pull/128643
- mixed_mm:
    - Enabling of AutoHeuristic: https://github.com/pytorch/pytorch/pull/131610
    - Script to collect data: https://github.com/pytorch/pytorch/pull/131611
    - A100 heuristic: https://github.com/pytorch/pytorch/pull/131613
    - H100 heuristic: https://github.com/pytorch/pytorch/pull/131790
- flex_attention: https://github.com/pytorch/pytorch/pull/130398
- mm (heuristic for ranking choices):
    - https://github.com/pytorch/pytorch/pull/131615
    - https://github.com/pytorch/pytorch/pull/131617
    - https://github.com/pytorch/pytorch/pull/131705
    - https://github.com/pytorch/pytorch/pull/131714
