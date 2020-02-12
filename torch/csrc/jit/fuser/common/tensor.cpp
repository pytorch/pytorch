#include <torch/csrc/jit/fuser/common/tensor.h>

namespace torch {
namespace jit {
namespace fuser {

namespace {
DataType aten_opt_type_map(
    const c10::optional<at::ScalarType>& scalar_type) {
  return scalar_type.has_value() ? 
      aten_to_data_type(scalar_type.value()) : DataType::Null;
}

c10::optional<TensorContiguity> infer_contiguity_from_tensor_type(
    const std::shared_ptr<c10::TensorType>& tensor_type) {
  if (!tensor_type->isComplete()) {
    return c10::nullopt;
  } else {
    return TensorContiguity(
        *(tensor_type->sizes().concrete_sizes()),
        *(tensor_type->strides().concrete_sizes()));
  }
}

} // namespace

/*
* Tensor member definitions
*/

Tensor::Tensor(const std::shared_ptr<c10::TensorType>& tensor_type)
  : Val(ValType::Tensor, aten_opt_type_map(tensor_type->scalarType())),
    contiguity_(infer_contiguity_from_tensor_type(tensor_type)) {
}

Tensor::Tensor(const std::shared_ptr<Value>& jit_value)
: Tensor(jit_value->type()->cast<c10::TensorType>()) {
}

bool Tensor::hasContiguityInfo() {
  return contiguity_.has_value();
}

const c10::optional<TensorContiguity>& Tensor::getContiguityInfo() {
  return contiguity_;
}

}}}
