#pragma once

#include <ATen/core/operator_name.h>
#include <c10/macros/Export.h>

namespace at::impl {

// C++-side copy of the Python decomp/meta/prims/op_impl registries
// these are updated at Python registration sites
//
// The tables are kept in sync by hooks at the Python registration sites, which
// call fakeDispatchTableAdd/Remove:
//   - Decomp:   torch._decomp.decomposition_table (post-autograd
//   decompositions),
//               via torch._decomp._add_op_to_registry
//   - Meta:     torch._decomp.meta_table (Python meta registrations),
//               via torch._decomp._add_op_to_registry
//   - OpImpl:   the exact-identity tier of
//               torch._subclasses.fake_impls.op_implementations_dict, via
//               register_op_impl / _deregister_op_impl. (Predicate-based
//               op_impls are matched separately in fakeFallback and are not
//               mirrored here.)
//   - PrimMeta: prims ops that define a prim_meta_impl, via torch._prims
//
// This does not store the Python callables and is backed by c10::LeftRight
enum class FakeDispatchCategory { Decomp, Meta, OpImpl, PrimMeta };

// Record/erase that `name` belongs to `category`. Called from the Python
// registration hooks listed above.
TORCH_API void fakeDispatchTableAdd(
    FakeDispatchCategory category,
    const c10::OperatorName& name);
TORCH_API void fakeDispatchTableRemove(
    FakeDispatchCategory category,
    const c10::OperatorName& name);

// True iff `name` is registered under `category`.
TORCH_API bool fakeDispatchTableContains(
    FakeDispatchCategory category,
    const c10::OperatorName& name);

} // namespace at::impl
