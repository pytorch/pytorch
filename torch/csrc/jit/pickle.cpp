#include <torch/csrc/WindowsTorchApiMacro.h>
#include <ATen/core/ivalue.h>
#include <torch/csrc/jit/pickler.h>


namespace torch {
namespace jit {

TORCH_API std::string pickle(
    const std::vector<IValue>& ivalues,
    std::vector<at::Tensor>* tensor_table) {
  std::stringstream ss;
  Pickler pickler(ss, tensor_table);

  if (tensor_table == nullptr) {
    // No tensor table provided, so tensors will be stored directly in the blob.
    // Add torch.save metadata so these tensors can be de-serialized later
    pickler.pushMetadata();
  }

  pickler.start();

  bool wrap_in_tuple = ivalues.size() > 0;

  if (wrap_in_tuple) {
    pickler.startTuple();
  }
  for (const auto& ivalue : ivalues) {
    pickler.addIValue(ivalue);
  }
  if (wrap_in_tuple) {
    pickler.endTuple();
  }
  pickler.finish();

  return ss.str();
}

TORCH_API std::vector<IValue> unpickle(
    std::istream& in,
    std::vector<at::Tensor>* tensor_table,
    ClassResolver class_resolver) {
  Unpickler unpickler(in, tensor_table, std::move(class_resolver));
  return unpickler.parse_ivalue_list();
}

TORCH_API std::vector<IValue> unpickle(
    const char* data,
    size_t size,
    std::vector<at::Tensor>* tensor_table,
    ClassResolver class_resolver) {
  std::stringstream ss;
  ss << std::string(data, size);
  Unpickler unpickler(ss, tensor_table, std::move(class_resolver));
  return unpickler.parse_ivalue_list();
}

TORCH_API std::vector<IValue> unpickle(
    const void* data,
    size_t size,
    std::vector<at::Tensor>* tensor_table,
    ClassResolver class_resolver) {
  return unpickle(
      reinterpret_cast<const char*>(data),
      size,
      tensor_table,
      std::move(class_resolver));
}

} // namespace jit
} // namespace torch
