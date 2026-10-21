# Copyright (c) 2026, Tri Dao.
"""Generic epilogue-GEMM runtime: identity (digests, refs, registries), the
plan/compile host layer, the ``quack::gemm_epi`` torch op, and autotune.

See AI/epilogue_transform_reorg.md for the layer map. Import submodules
directly (``quack.gemm_runtime.identity``); this package root stays empty so
the identity leaf can be imported without pulling the host layer.
"""
