torch.compile
====================

It's fast, it's easy to use, here are some benchmarks

Optimizations can be passed in :func:`~torch.compile` with either a backend mode parameter or as passes. To understand what are the available options you can run
:func:`~torch._inductor.list_options()`` and :func:`~torch._inductor.list_mode_options`

Gotchas: Dynamic shape support, distributed training, export, custom backends

.. autosummary::
    :toctree: generated
    :nosignatures:

    compile

.. toctree::
   :maxdepth: 1
   :hidden:

   installation
   get-started
   technical-overview
   troubleshooting
   faq
