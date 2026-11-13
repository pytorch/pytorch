#include <ATen/FakeTensorDispatchTables.h>
#include <ATen/core/dispatch/Dispatcher.h>

#include <algorithm>

namespace {

bool contains_tensor_types(const c10::TypePtr& type) {
  if (type->isSubtypeOf(*c10::TensorType::get())) {
    return true;
  }
  const auto contained = type->containedTypes();
  return std::any_of(contained.begin(), contained.end(), contains_tensor_types);
}

// Same as Python's _is_tensor_constructor in fake_impls.py
bool is_tensor_constructor(const c10::FunctionSchema& schema) {
  const auto& arguments = schema.arguments();
  return std::none_of(
             arguments.begin(),
             arguments.end(),
             [](const c10::Argument& a) {
               return contains_tensor_types(a.type());
             }) &&
      schema.returns().size() == 1 &&
      schema.returns()[0].type() == c10::TensorType::get();
}

// See cppFakeNotImplemented; categories is the op's fakeDispatchTableCategories
bool cpp_fake_not_implemented(const c10::OperatorHandle& op, uint32_t categories) {
  using at::impl::FakeDispatchCategory;
  for (const auto category :
       {FakeDispatchCategory::OpImpl,
        FakeDispatchCategory::CustomOpImpl,
        FakeDispatchCategory::PrimMeta,
        FakeDispatchCategory::PythonCIA}) {
    if (at::impl::fakeDispatchCategoriesHave(categories, category)) {
      return true;
    }
  }
  return op.hasTag(at::Tag::dynamic_output_shape) ||
      op.hasTag(at::Tag::data_dependent_output) ||
      is_tensor_constructor(op.schema()) ||
      (op.operator_name().name.starts_with("aten::_foreach_") &&
       op.hasComputedKernelForDispatchKey(c10::DispatchKey::Meta));
}

} // namespace

namespace at::impl {

bool cppFakeNotImplemented(const c10::OperatorHandle& op) {
  return cpp_fake_not_implemented(op, fakeDispatchTableCategories(op.operator_name()));
}

} // namespace at::impl
