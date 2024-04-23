Best Practices for Backends
===========================

x86 CPU
-------

Compiled workloads on modern x86 CPUs are usually optimized by Single Instruction Multiple Data (SIMD) instruction sets. SIMD is a typical parallel processing technique for high performance computing, such as deep learning model training and inference. With SIMD applied, each compute unit performs the same instruction with different allocated data at any given time slot. The most commonly deployed x86 instruction set architectures (ISAs) enabling SIMD include `AVX, AVX2, AVX-512 <https://en.wikipedia.org/wiki/Advanced_Vector_Extensions>`_ and `AMX <https://en.wikipedia.org/wiki/Advanced_Matrix_Extensions>`_.

You can check supported ISAs for your machine by using the `collect_env script <https://github.com/pytorch/pytorch/blob/main/torch/utils/collect_env.py>`_. As the script provides complete environment information for PyTorch, we can use ``grep`` to extract the line containing ISA information:

::

    python collect_env.py | grep "a[(v|m)]x"

Normally, if AVX-512 is supported, instructions start with "avx512" (like ``avx512f``, ``avx512bw``, ``avx512_vnni``) should be observed. If AMX is supported, instructions start with "amx" (like ``amx_tile``, ``amx_bf16``, ``amx_int8``) should be observed.

Specifically, with a server having AMX instructions enabled, workloads performance can be further boosted by `leveraging AMX <https://pytorch.org/tutorials/recipes/amx.html>`_.
