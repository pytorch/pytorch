# General, DSL-agnostic utilities shared by all torch._native op families. Modules
# here must NOT import any DSL runtime (cutlass, triton, tvm_ffi) at module scope --
# they are imported during override registration / cond evaluation, which the
# lazy-DSL-import contract keeps free of the DSL toolchain (see
# test_no_dsl_imports_after_import_torch and the README's import note).
#
# Deliberately empty: importing the submodules here would load them for anyone touching the
# package, which is the opposite of what the contract above is for. Callers name what they
# need -- `from torch._native.utils import capability`.
