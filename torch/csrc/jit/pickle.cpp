#include <torch/csrc/WindowsTorchApiMacro.h>
#include <ATen/core/ivalue.h>
#include <torch/csrc/jit/pickler.h>


namespace torch {
namespace jit {

std::string Pickle(
    std::vector<IValue> ivalues,
    std::vector<at::Tensor>* tensor_table) {
  std::stringstream ss;

  Pickler pickler(ss, tensor_table);
  pickler.start();
  pickler.startTuple();
  for (const auto& ivalue : ivalues) {
    pickler.addIValue(ivalue);
  }
  pickler.endTuple();
  pickler.finish();

  return ss.str();
}

std::vector<IValue> Unpickle(
    std::istream& in,
    std::vector<at::Tensor>* tensor_table,
    ClassResolver class_resolver) {
  // TODO: don't double copy here
  Unpickler unpickler(in, tensor_table, class_resolver);
  return unpickler.parse_ivalue_list();
}

std::vector<IValue> Unpickle(
    const char* data,
    size_t size,
    std::vector<at::Tensor>* tensor_table,
    ClassResolver class_resolver) {
  // TODO: don't double copy here
  std::stringstream ss;
  ss << std::string(data, size);
  Unpickler unpickler(ss, tensor_table, class_resolver);
  return unpickler.parse_ivalue_list();
}

std::vector<IValue> Unpickle(
    const void* data,
    size_t size,
    std::vector<at::Tensor>* tensor_table,
    ClassResolver class_resolver) {
  // TODO: don't double copy here
  return Unpickle(
      reinterpret_cast<const char*>(data), size, tensor_table, class_resolver);
}

} // namespace jit
} // namespace torch
