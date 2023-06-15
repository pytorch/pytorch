Best Practice for Backends
==========================

x86 CPU
-----------------

Compiled workloads on modern x86 CPUs are usually optimized by SIMD instruction sets. SIMD (Single Instruction Multiple Data) is a typical parallel processing technique for high performance computing, such as deep learning model training and inference. With SIMD applied, each compute unit performs the same instruction with different allocated data at any given timeslot. The most commonly deployed x86 instruction set architectures (ISAs) enabling SIMD include `AVX, AVX2, AVX-512 <https://en.wikipedia.org/wiki/Advanced_Vector_Extensions>` and `AMX <https://en.wikipedia.org/wiki/Advanced_Matrix_Extensions>`.

Which ISAs are supported by a machine can be checked with `the tutorial <https://linuxconfig.org/identify-if-my-cpu-using-32-bit-or-64-bit-instruction-set>`.

Specifically, with a server having AMX instructions enabled, workloads performance can be further boosted by `leveraging AMX <https://pytorch.org/tutorials/recipes/recipes/amx.html>`.
