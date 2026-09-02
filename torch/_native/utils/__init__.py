# General, DSL-agnostic utilities shared by all torch._native op families. Modules here must NOT
# import a DSL runtime at module scope: they load during override registration and cond evaluation,
# which the lazy-DSL-import contract keeps free of the toolchain (test_no_dsl_imports_after_
# import_torch). Deliberately empty for the same reason -- callers name what they need.
