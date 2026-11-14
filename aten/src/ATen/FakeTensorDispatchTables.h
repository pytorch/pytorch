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
// ops FakeTensorMode decomposes under static shapes and
// cpp_meta_supports_symint_ops
//
// The tables are kept in sync by hooks at the Python registration sites, which
// call fakeDispatchTableAdd/Remove:
//   - Decomp:   torch._decomp.decomposition_table (post-autograd
//   decompositions),
//               via torch._decomp._add_op_to_registry / remove_decompositions
//   - Meta:     torch._decomp.meta_table (Python meta registrations),
//               via torch._decomp._add_op_to_registry / remove_decompositions
//   - OpImpl:   the exact-identity tier of
//               torch._subclasses.fake_impls.op_implementations_dict, via
//               register_op_impl / _deregister_op_impl, except aten::alias.
//               (Predicate-based op_impls are not mirrored here;
//               FakeFallbackKernel.cpp re-implements the built-in predicates.)
//   - PrimMeta: prims ops that define a prim_meta_impl, via torch._prims
//   - PythonCIA: Python CompositeImplicitAutograd overrides, via mutations of
//                OpOverload.py_kernels
//   - StaticShapeDecomp: default static-shape decomposition eligibility,
//                initialized after torch._meta_registrations imports finish,
//                then updated by register_decomposition /
//                remove_decompositions. Callers still check Meta membership and
//                sparse inputs. Direct dictionary edits and mode-method
//                overrides are not mirrored; Python's first-dispatch
//                classification cache is unchanged.
//   - CppMetaSupportsSymint: cpp_meta_supports_symint_ops, initialized once by
//                torch._meta_registrations. Later set mutations are not
//                mirrored.
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
  CppMetaSupportsSymint,
  NumCategories
};

static_assert(static_cast<int>(FakeDispatchCategory::NumCategories) <= 32);

inline uint32_t fakeDispatchCategoryBit(FakeDispatchCategory category) {
  TORCH_INTERNAL_ASSERT(category < FakeDispatchCategory::NumCategories);
  return 1u << static_cast<int>(category);
}

// Record/erase that `name` belongs to `category`. Called from the Python
// registration hooks listed above, including initialization at import.
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

} // namespace at::impl
