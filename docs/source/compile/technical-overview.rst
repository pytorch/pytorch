Technical Overview
====================

**TorchDynamo** is a Python-level JIT compiler designed to make unmodified
PyTorch programs faster. TorchDynamo hooks into the frame evaluation API
in CPython (`PEP 523 <https://peps.python.org/pep-0523/>`__) to
dynamically modify Python bytecode right before it is executed. It
rewrites Python bytecode in order to extract sequences of PyTorch
operations into an `FX Graph <https://pytorch.org/docs/stable/fx.html>`__
which is then just-in-time compiled with a customizable backend.
It creates this FX Graph through bytecode analysis and is designed to
mix Python execution with compiled backends to get the best of both
worlds — usability and performance.

TorchDynamo makes it easy to experiment with different compiler
backends to make PyTorch code faster with a single line decorator
``torch._dynamo.optimize()`` which is wrapped for convenience by ``torch.compile()``

.. image:: ../_static/img/dynamo/TorchDynamo.png

`TorchInductor` is one of the backends
supported by `TorchDynamo Graph <https://pytorch.org/docs/stable/fx.html>`__
into `Triton <https://github.com/openai/triton>`__ for GPUs or
`C++/OpenMP <https://www.openmp.org/>`__ for CPUs. We have a
`training performance dashboard <https://github.com/pytorch/torchdynamo/issues/681#issuecomment-1233828468>`__
that provides performance comparison for different training backends. You can read
more in the `TorchInductor post on PyTorch
dev-discuss <https://dev-discuss.pytorch.org/t/torchinductor-a-pytorch-native-compiler-with-define-by-run-ir-and-symbolic-shapes/747>`__.

.. seealso::

   * `TorchDynamo deep-dive video <https://www.youtube.com/watch?v=egZB5Uxki0I>`__
   * `dev-discuss topics <https://dev-discuss.pytorch.org/search?q=TorchDynamo%20order%3Alatest>`__

.. toctree::
   :maxdepth: 1

   guards-overview
   best-practices-for-backends
   custom-backends
   deep-dive
