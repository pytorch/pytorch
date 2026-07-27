# General, DSL-agnostic utilities shared by all torch._native op families. Modules
# here must NOT import any DSL runtime (cutlass, triton, tvm_ffi) at module scope --
# they are imported during override registration / cond evaluation, which the
# lazy-DSL-import contract keeps free of the DSL toolchain (see
# test_no_dsl_imports_after_import_torch and the README's import note).

from . import capability, lazy


__all__ = ["capability", "lazy"]
