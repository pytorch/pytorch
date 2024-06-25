PyTorch 2.0 Performance Dashboard
=================================

**Author:** `Bin Bao <https://github.com/desertfire>`__ and `Huy Do <https://github.com/huydhn>`__

PyTorch 2.0's performance is tracked nightly on this `dashboard <https://hud.pytorch.org/benchmark/compilers>`__.
The performance collection runs on 12 GCP A100 nodes every night. Each node contains a 40GB A100 Nvidia GPU and
a 6-core 2.2GHz Intel Xeon CPU. The corresponding CI workflow file can be found
`here <https://github.com/pytorch/pytorch/blob/main/.github/workflows/inductor-perf-test-nightly.yml>`__.

How to read the dashboard?
---------------------------

The landing page shows tables for all three benchmark suites we measure, ``TorchBench``, ``Huggingface``, and ``TIMM``,
and graphs for one benchmark suite with the default setting. For example, the default graphs currently show the AMP
training performance trend in the past 7 days for ``TorchBench``. Droplists on the top of that page can be
selected to view tables and graphs with different options. In addition to the pass rate, there are 3 key
performance metrics reported there: ``Geometric mean speedup``, ``Mean compilation time``, and
``Peak memory footprint compression ratio``.
Both ``Geometric mean speedup`` and ``Peak memory footprint compression ratio`` are compared against
the PyTorch eager performance, and the larger the better. Each individual performance number on those tables can be clicked,
which will bring you to a view with detailed numbers for all the tests in that specific benchmark suite.

What is measured on the dashboard?
-----------------------------------

All the dashboard tests are defined in this
`function <https://github.com/pytorch/pytorch/blob/3e18d3958be3dfcc36d3ef3c481f064f98ebeaf6/.ci/pytorch/test.sh#L305>`__.
The exact test configurations are subject to change, but at the moment, we measure both inference and training
performance with AMP precision on the three benchmark suites. We also measure different settings of TorchInductor,
including ``default``, ``with_cudagraphs (default + cudagraphs)``, and ``dynamic (default + dynamic_shapes)``.

Can I check if my PR affects TorchInductor's performance on the dashboard before merging?
-----------------------------------------------------------------------------------------

Individual dashboard runs can be triggered manually by clicking the ``Run workflow`` button
`here <https://github.com/pytorch/pytorch/actions/workflows/inductor-perf-test-nightly.yml>`__
and submitting with your PR's branch selected. This will kick off a whole dashboard run with your PR's changes.
Once it is done, you can check the results by selecting the corresponding branch name and commit ID
on the performance dashboard UI. Be aware that this is an expensive CI run. With the limited
resources, please use this functionality wisely.

How can I run any performance test locally?
--------------------------------------------

The exact command lines used during a complete dashboard run can be found in any recent CI run logs.
The `workflow page <https://github.com/pytorch/pytorch/actions/workflows/inductor-perf-test-nightly.yml>`__
is a good place to look for logs from some of the recent runs.
In those logs, you can search for lines like
``python benchmarks/dynamo/huggingface.py --performance --cold-start-latency --inference --amp --backend inductor --disable-cudagraphs --device cuda``
and run them locally if you have a GPU working with PyTorch 2.0.
``python benchmarks/dynamo/huggingface.py -h`` will give you a detailed explanation on options of the benchmarking script.
