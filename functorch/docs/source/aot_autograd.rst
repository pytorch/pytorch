functorch.compile (experimental)
================================

AOT Autograd is an experimental feature that allows ahead of time capture of
forward and backward graphs, and allows easy integration with compilers. This
creates an easy to hack Python-based development environment to speedup training
of PyTorch models. AOT Autograd currently lives inside ``functorch.compile``
namespace.

.. warning::
    AOT Autograd is experimental and the APIs are likely to change. We are looking
    for feedback. If you are interested in using AOT Autograd and need help or have
    suggestions, please feel free to open an issue. We will be happy to help.

.. currentmodule:: functorch.compile

Compilation APIs (experimental)
-------------------------------
.. autosummary::
    :toctree: generated
    :nosignatures:

    aot_function
    aot_module
    memory_efficient_fusion

Partitioners (experimental)
---------------------------
.. autosummary::
    :toctree: generated
    :nosignatures:

    default_partition
    min_cut_rematerialization_partition

Compilers (experimental)
------------------------
.. autosummary::
    :toctree: generated
    :nosignatures:

    nop
    ts_compile
