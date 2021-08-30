#include <c10/util/Exception.h>
#include <torch/csrc/jit/mobile/model_tracer/TensorUtils.h>

namespace torch {
namespace jit {
namespace mobile {
void for_each_tensor_in_ivalue(
    c10::IValue& iv,
    std::function<void(::at::Tensor&)> const& func) {
  const bool is_leaf_type = iv.isString() || iv.isNone() || iv.isScalar() ||
      iv.isDouble() || iv.isInt() || iv.isBool() || iv.isDevice() ||
      iv.isIntList() || iv.isDoubleList() || iv.isBoolList();
  if (is_leaf_type) {
    // Do Nothing.
    return;
  }

  if (iv.isTensor()) {
    ::at::Tensor& t = iv.toTensor();
    func(t);
  } else if (iv.isTuple()) {
    c10::intrusive_ptr<at::ivalue::Tuple> tup_ptr = iv.toTuple();
    for (auto& e : tup_ptr->elements()) {
      for_each_tensor_in_ivalue(e, func);
    }
  } else if (iv.isList()) {
    c10::List<c10::IValue> l = iv.toList();
    for (auto&& i : l) {
      c10::IValue item = i;
      for_each_tensor_in_ivalue(item, func);
    }
  } else if (iv.isGenericDict()) {
    c10::Dict<c10::IValue, c10::IValue> dict = iv.toGenericDict();
    for (auto& it : dict) {
      ::c10::IValue v = it.value();
      for_each_tensor_in_ivalue(v, func);
    }
  } else {
    AT_ERROR("Unhandled type of IValue. Got ", iv.tagKind());
  }
}
} // namespace mobile
} // namespace jit
} // namespace torch
