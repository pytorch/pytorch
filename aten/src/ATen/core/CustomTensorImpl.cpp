#include <ATen/core/CustomTensorImpl.h>

#include <ATen/core/ATenDispatch.h>
#include <torch/csrc/autograd/variable.h>
#include <torch/csrc/jit/script/function_schema_parser.h>

namespace at {

c10::intrusive_ptr<TensorImpl> CustomTensorImpl::shallow_copy_and_detach(
    const c10::VariableVersion& version_counter,
    bool allow_tensor_metadata_change) const {
  auto impl = c10::make_intrusive<CustomTensorImpl>(
    tag_,
    storage_,
    type_set(),
    dtype(),
    device_type()
  );
  return impl;
}

void customDispatch(const c10::FunctionSchema& schema, torch::jit::Stack* stack);

// Wrapper to register custom dispatch
struct RegisterCustomTensorDispatch {
  RegisterCustomTensorDispatch() {
    globalATenDispatch().registerFallbackBoxedOp(
        c10::TensorTypeId::CustomTensorId, &customDispatch);
  }
};

static RegisterCustomTensorDispatch r;

// tag_name -> tag
std::unordered_map<std::string, size_t>& getTagNameMap() {
  static std::unordered_map<std::string, size_t> tag_name_map;
  return tag_name_map;
}
// tag -> (schema -> impl)
static std::unordered_map<size_t, std::unordered_map<std::string, Impl*>> tag_impls;

namespace native {

at::Tensor to_custom(const Tensor& self, std::string type) {
  TORCH_CHECK(getTagNameMap().find(type) != getTagNameMap().end());
  size_t tag = getTagNameMap()[type];
  auto& m = tag_impls[tag];
  auto impl = m["to_custom"];
  TORCH_CHECK(impl);
  torch::jit::Stack s;
  torch::jit::push(s, self);
  torch::jit::push(s, type);
  const static char* schema = "aten::to_custom(Tensor self, str type) -> Tensor";
  static auto fs = torch::jit::parseSchema(schema);
  impl(fs, &s);
  auto tensor = torch::jit::pop(s).toTensor();
  return tensor;
}

// This is called in the case of non-Custom tensors
// For custom tensors, register "from_custom" just as you would "to_custom"
at::Tensor from_custom(const Tensor& self) {
  return self;
}

} // namespace native

void customDispatch(const FunctionSchema& schema, torch::jit::Stack* stack) {
  size_t tag = -1; // meaningless, must be overwritten

  for (const auto& iv : *stack) {
    if (iv.isTensor()) {
      auto t = iv.toTensor();
      auto cst = static_cast<CustomTensorImpl*>(t.unsafeGetTensorImpl());
      tag = cst->tag();
      break;
    }
  }
  TORCH_CHECK(tag_impls.find(tag) != tag_impls.end());

  auto& m = tag_impls[tag];
  auto impl = m[schema.name()];
  impl(schema, stack);
}

RegisterCustomTensorImpl::RegisterCustomTensorImpl(std::string name, std::string method, Impl* impl) {
  if (getTagNameMap().find(name) == getTagNameMap().end()) {
    getTagNameMap()[name] = getTagNameMap().size();
  }
  auto tag = getTagNameMap()[name];
  auto& m = tag_impls[tag];
  TORCH_CHECK(m.find(method) == m.end());
  m[method] = impl;
}

} // namespace at
