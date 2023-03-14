.. automodule:: torch
.. currentmodule:: torch

torch.compile
====================

:func::`~torch.compile` was introduced in PyTorch 2.0 `PyTorch 2.0 <https://pytorch.org/get-started/pytorch-2.0/>`__

Our default and supported backend is ::module::`~torch._inductor` with benchmarks `showing 30% to 2x speedups and 10% memory compression <https://github.com/pytorch/pytorch/issues/93794>`__
on real world models for both training and inference with a single line of code.

.. note::
    The :func:`~torch.compile` API is experimental and subject to change.

The simplest possible interesting program is the below which we go over in a lot more detail in `getting started <https://pytorch.org/docs/master/compile/get-started.html> `
showing how to use :func:`~torch.compile` to speed up inference on a variety of real world models from both TIMM and HuggingFace which we 
co-announced `here <https://pytorch.org/blog/Accelerating-Hugging-Face-and-TIMM-models/>__`

.. code:: python

   import torch
   def fn(x):
       x = torch.cos(x).cuda()
       x = torch.sin(x).cuda()
       return x
   compiled_fn = torch.compile(fn(torch.randn(10).cuda()))

::func::`torch.compile` works over ::class::`torch.nn.Module` as well as functions so you can pass in your entire training loop.

The above example was for inference but you can follow this tutorial for an `example on training <https://pytorch.org/tutorials/intermediate/torch_compile_tutorial.html>__`

Gotchas: Dynamic shape support, distributed training, export, custom backends

Optimizations
-------------

Optimizations can be passed in :func:`~torch.compile` with either a backend mode parameter or as passes. To understand what are the available options you can run
:func:`~torch._inductor.list_options()`` and :func:`~torch._inductor.list_mode_options`

The default backend is `backend="inductor"` which will likely be the most reliable and performant option for most users,
other backends are there for power users who don't mind more experimental community support.    

.. autosummary::
    :toctree: generated
    :nosignatures:

    compile

Troubleshooting and Gotchas
---------------------------

IF you experience issues with models failing to compile, running of out of memory, recompiling too often, not giving accurate results,
odds are you will find the right tool to solve your problem in our guides.

.. WARNING::
    A few features are still very much in development and not likely to work for most users
    - Dynamic shapes
    - Distributed training
    - Model export

.. toctree::
   :maxdepth: 1

   troubleshooting
   faq

Learn more
----------

If you can't wait to get started and want to learn more about the internals of the PyTorch 2.0 stack then 
please check out the references below.

.. toctree::
   :maxdepth: 1

   installation
   get-started
   technical-overview


