"""Process-wide lock for in-process CuTeDSL compilation.

CuTeDSL's ``DSLPreprocessor`` singleton and ``cuda_dialect_init_library_once``
are not thread-safe. Inductor compiles CuTeDSL kernels from several places
(template precompile threads, lazy benchmark-time compiles, NVGEMM's in-process
fallback), so every in-process ``cute.compile`` site must hold this lock.
Subprocess compile workers take it uncontended.
"""

import threading


CUTEDSL_COMPILE_LOCK = threading.Lock()
