# General, DSL-agnostic utilities shared by every torch._native op family. Modules here must
# NOT import a DSL runtime at module scope: they load during registration and cond evaluation,
# which the lazy-DSL-import contract keeps free of the toolchain. Empty for the same reason.
