#pragma once

#include <ATen/core/operator_name.h>
#include <c10/macros/Export.h>
#include <c10/util/Exception.h>

#include <cstdint>

namespace c10 {
class OperatorHandle;
} // namespace c10

namespace at::impl {

// C++-side copy of the Python decomp/meta/prims/op_impl registries, and of the
// ops FakeTensorMode decomposes under static shapes
//
// The tables are kept in sync by hooks at the Python registration sites, which
// call fakeDispatchTableAdd/Remove, except StaticShapeDecomp, filled once:
//   - Decomp:   torch._decomp.decomposition_table (post-autograd
//   decompositions),
//               via torch._decomp._add_op_to_registry
//   - Meta:     torch._decomp.meta_table (Python meta registrations),
//               via torch._decomp._add_op_to_registry
//   - OpImpl:   the exact-identity tier of
//               torch._subclasses.fake_impls.op_implementations_dict, via
//               register_op_impl / _deregister_op_impl, minus ops whose C++
//               meta kernel is used instead (see cpp_meta_supports_symint_ops).
//               (Predicate-based op_impls are not mirrored here;
//               fakeHasPythonFakeImpl re-implements the predicates.)
//   - PrimMeta: prims ops that define a prim_meta_impl, via torch._prims
//   - PythonCIA: Python CompositeImplicitAutograd overrides, via
//                OpOverload.py_impl
//   - StaticShapeDecomp: the Decomp ops that FakeTensorMode decomposes under
//                static shapes when they are not in Meta and have no sparse
//                input (decomposes_under_static_shapes), filled by
//                torch._meta_registrations once the decompositions are
//                registered (torch._decomp imports them all). Later
//                registrations and removals are not mirrored;
//                test_fake_fallback_static_shape_decomps checks the table
//                against the predicate.
//
// This does not store the Python callables and is backed by c10::LeftRight
enum class FakeDispatchCategory {
  Decomp,
  Meta,
  OpImpl,
  PrimMeta,
  PythonCIA,
  // for custom op handling registered through
  // torch.library.custom_op(...).register_fake
  CustomOpImpl,
  StaticShapeDecomp,
  NumCategories
};

static_assert(static_cast<int>(FakeDispatchCategory::NumCategories) <= 32);

inline uint32_t fakeDispatchCategoryBit(FakeDispatchCategory category) {
  TORCH_INTERNAL_ASSERT(category < FakeDispatchCategory::NumCategories);
  return 1u << static_cast<int>(category);
}

// Record/erase that `name` belongs to `category`. Called from the Python
// registration hooks listed above, and once at import for StaticShapeDecomp.
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

// The categories `name` is registered under, read at once, as a bitmask to
// test with fakeDispatchCategoriesHave
TORCH_API uint32_t fakeDispatchTableCategories(const c10::OperatorName& name);

inline bool fakeDispatchCategoriesHave(
    uint32_t categories,
    FakeDispatchCategory category) {
  return categories & fakeDispatchCategoryBit(category);
}

// Whether Python's FakeTensorMode runs op through a Python implementation
// rather than its Meta kernel, which the C++ fake fallback must reject: an op
// in the OpImpl, CustomOpImpl, PrimMeta or PythonCIA table, or one an
// op_implementations predicate selects (constructors, the dynamic_output_shape
// and data_dependent_output tags, and _foreach_ ops with a Meta kernel), as
// _dispatch_impl checks them. Decomp and StaticShapeDecomp ops are decomposed
// only depending on the inputs, and Meta ops run their Meta kernel, so callers
// check those. alias's op_impl, which only keeps the recorded dispatch keys, is
// not registered in OpImpl, as the fallback keeps them itself.
TORCH_API bool fakeHasPythonFakeImpl(const c10::OperatorHandle& op);

} // namespace at::impl
