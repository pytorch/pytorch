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
each carrying the same exports as attributes/methods. A table-driven family
builds its declaration objects from the same table that drives its JIT
registration.

Required exports (module or declaration object):

  ATEN_OP: str               Unique structured base or qualified overload name; an
                              unstructured hook names an exact fresh-output or inplace op.
  DISPATCH_KEY: str           e.g. "CUDA"
  KERNEL_MODULE: str          sibling module exporting build(spec); the
                              export tool package-imports it with the
                              built torch available (two-stage build),
                              so it may share code with the JIT wrapper
  ARCHS: tuple[str, ...]      candidate compile targets the op supports (sm
                              strings). For each device architecture supported
                              by the containing build, export chooses the widest
                              compatible candidate; an ``f`` family target wins
                              an equal-coverage tie. An ``a`` target is exact-only.
                              Codegen derives runtime device gates from the
                              targets actually shipped, so declarations never
                              hand-write architecture checks.
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
                              launch_fn(...). Structured outputs are
                              already allocated. With STRUCTURED=False,
                              allocate outputs and assign aot_result
                              (the schema's return type). No fallback:
                              the chain's return false IS the fallback.

For all hooks, returning false leaves data, metadata and operator-visible state
(including RNG) unchanged. Finish decline checks before mutating or enqueueing.

decl_id() names the stub, kernel, artifact directory and coverage op.

Emitted C++ uses ATen/core/Tensor.h. Add per-op headers through cpp_includes()
when calling factories or operators such as at::empty or `t + t`.

Optional exports:

  STRUCTURED: bool           True by default; False runs before the backend, with a
                              device guard but no meta(). Validate all inputs/devices;
                              on handled calls, fill fresh aot_result or mutate self.
                              Inplace hooks receive and return self as aot_result.
  EXACT: bool                Match one exact functional name even when a bare
                              name also has structured overloads, e.g. round.
  cpp_includes() -> str      Additional per-declaration C++ includes.
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
                              declaration-level set as covered_axes +
                              grid matching; codegen conjoins the
                              shipped-target and ABI gates. Like
                              covered_axes it may be narrower than the
                              stub's dispatch chain but never wider than
                              intended coverage.

Every generated op also registers torch.ops._native_aot.archs_<op>(), returning
the device compute capabilities covered by its embedded targets (e.g. sm_100f
reports [100, 103, 107]). This is independent of cpp_covers: the Python fallback
must verify that this op was embedded for the input device before subtracting any
calls from the JIT route.

Emission cardinality: cpp_helpers once per file, cpp_dispatch_prelude
once per op, cpp_dispatch/cpp_launch once per precompile point. The
generated stub is::

    helpers | prelude -> [if (dispatch) { launch; return true; }]* -> return false

in the op's structured impl scope (outputs allocated by meta(), device
guard held). Dispatch conditions are evaluated ASSUMING the prelude
passed; locals the prelude declares are in scope for dispatch and
launch. For unstructured functions the signature is the dispatcher
signature followed by a reference to aot_result.
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
