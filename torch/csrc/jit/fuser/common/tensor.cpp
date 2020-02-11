#include <torch/csrc/jit/fuser/common/tensor.h>

namespace torch {
namespace jit {
namespace fuser {

/*
* Tensor member definitions
*/

Tensor::Tensor(const std::shared_ptr<c10::TensorType>& tensor_type)
: Val(ValType::Tensor, DataType::Null) {
  std::runtime_error("Not implemented yet.");
}

/*
  // TODO: protocol between codegen and JIT is not set in stone yet.
  // Issue 1:
  //   Profiling executor promises static shape information, but the defaul executor cannot guaranttee this.
  //   We inevitably would need to support dim only tensor.
  // Issue 2:
  //   Our codegen is trying to be flexible. We want to do codegen based on contiguity and broadcast information,
  //   which requires corresponding support from profiling executor. This means requiring static sizes and strides
  //   are too restricting for our need and should be updated.
  assert(tensor_type && tensor_type->isComplete());

  scalar_type_ = tensor_type->scalarType();

  auto dim = tensor_type->dim().value();
  sizes_ = VectorInts(dim);
  strides_ = VectorInts(dim);
  for (int i = 0; i < dim; i++) {
    sizes_.value()[i] = *(tensor_type->sizes()[i]);
    strides_.value()[i] = *(tensor_type->strides()[i]);
  }
}

Tensor::Tensor(const std::shared_ptr<Value>& jit_value)
: Tensor(jit_value->type()->cast<c10::TensorType>()) {
}
*/

}}}