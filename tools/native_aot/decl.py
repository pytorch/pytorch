"""Declaration contract for native-AOT op modules.

THE contract an op author programs against. The mechanism behind it (the
validating loader, discovery, decl_id) lives in
torchgen/native_aot_decl.py and is re-exported here, because installed
torchgen must load declarations out of tree, where tools/ is not shipped.
This file stays the documented home of the contract: torchgen, the
runtime coverage layer, cmake/Codegen.cmake and the README all name
this path.

An op opts into AOT by shipping ``torch/_native/ops/<op>/aot.py``, a
module whose scope is TORCH-FREE (torch lazily inside function bodies)
so torchgen can load it pre-build. Stdlib and torchgen are both fine at
module scope; torchgen is pure Python with no torch dependency.

A module declares either ONE op (the module itself carries the exports
below) or a FAMILY: it exports ``declarations() -> list`` of objects,
each carrying the same exports as attributes/methods. Table-driven
families (e.g. pointwise) build their declaration objects from the same
table that drives their JIT registration.

Required exports (module or declaration object):

  ATEN_OP: str                name of a STRUCTURED op: a base name
                              ("topk") when the base resolves to exactly
                              one structured group, or overload-qualified
                              ("gt.Tensor", "all.dim") when overloads
                              have separate structured groups. decl_id()
                              (dots -> underscores) names the stub, the
                              generated kernel, and the covers op.
  DISPATCH_KEY: str           e.g. "CUDA"
  KERNEL_MODULE: str          sibling module exporting build(spec); the
                              export tool package-imports it with the
                              built torch available (two-stage build),
                              so it may share code with the JIT wrapper
  kernel_precompile_grid() -> list[dict]
                              the artifact grid; list-valued fields
                              cross-multiply; one precompiled kernel per
                              expanded point. Field values must be
                              JSON-representable (sidecars store the
                              spec)
  covered_axes(*schema_args) -> dict
                              project a live call onto grid axes; a call
                              is covered (declines the JIT route) iff
                              some precompile point matches every
                              returned field; exceptions => uncovered
  cpp_dispatch(spec) -> str   one boolean C++ expr per precompile point:
                              given a call that passed the prelude, is
                              it served by THIS point? First match wins.
  cpp_launch(spec, launch_fn) -> str
                              C++ invoking this point's kernel via
                              launch_fn(...); no allocation, no fallback
                              (the chain's return false IS the fallback)

Emitted C++ (any of the cpp_* exports) sees ATen/core/Tensor.h, not
ATen/ATen.h -- Tensor methods all work, but calling an at:: FACTORY
(at::empty, at::zeros, at::cat, ...) or an operator like `t + t` needs
its per-op header added to FILE_TMPL in gen_aot_lib.py. The failure is a
loud "'empty' is not a member of 'at'" at build time, not a silent one.

Optional exports:

  ARCHS: tuple[str, ...]      architectures the op's kernels are valid
                              on (sm strings). Defaults to all sm90+.
                              Export skips arches outside it; codegen
                              emits a runtime device gate from
                              ARCHS intersect shipped-arches, so
                              declarations never hand-write arch
                              checks.
  cpp_dispatch_prelude() -> str | None
                              shared front half of the dispatch chain:
                              cheap universal rejects and setup (locals,
                              classifier calls) every branch reads. May
                              also `return true` for degenerate calls
                              the op serves without a kernel (e.g. an
                              empty index -> copy only), bypassing the
                              chain entirely. Absent => every
                              cpp_dispatch(spec) must be self-contained.
  cpp_helpers() -> str | None C++ shared beyond one op (family
                              classifiers); emitted once per generated
                              file.
  cpp_covers() -> str | None  fast C++ port of covered_axes matching:
                              a bool-returning body over the op's
                              FUNCTIONAL schema arguments (outputs do
                              not exist yet -- this runs at router
                              time, pre-meta()). Registered by the AOT
                              library as torch.ops._native_aot
                              .covers_<op>; the runtime coverage layer
                              prefers it over the Python path when the
                              library is loaded. Must decide the SAME
                              covered set as covered_axes + grid
                              matching; like covered_axes it may be
                              narrower than the stub's dispatch chain
                              but never wider than intended coverage.

Emission cardinality: cpp_helpers once per file, cpp_dispatch_prelude
once per op, cpp_dispatch/cpp_launch once per precompile point. The
generated stub is::

    helpers | prelude -> [if (dispatch) { launch; return true; }]* -> return false

in the op's structured impl scope (outputs allocated by meta(), device
guard held). Dispatch conditions are evaluated ASSUMING the prelude
passed; locals the prelude declares are in scope for dispatch and
launch.
"""

from torchgen.native_aot_decl import (
    AotDeclaration,
    archs_of,
    decl_id,
    discover_declarations,
    load_by_path,
    load_declaration,
    load_declarations,
)


__all__ = [
    "AotDeclaration",
    "archs_of",
    "decl_id",
    "discover_declarations",
    "load_by_path",
    "load_declaration",
    "load_declarations",
]
