#pragma once

// ${generated_comment}

$th_headers

#include "ATen/Tensor.h"
#include "ATen/TensorImpl.h"
#include "ATen/TensorMethods.h"

namespace at {

struct ${Tensor} final : public TensorImpl {
public:
  ${Tensor}(THTensor * tensor);
  virtual Scalar localScalar() override;
  virtual std::unique_ptr<Storage> storage() override;
  static const char * typeString();
};

namespace detail {
  // This is just a temporary function to help out code generation.
  // Eventually, the codegen code should construct tensors using
  // a new Tensor constructor that takes scalar type and backend,
  // but I haven't written this yet.
  ${Tensor}* new_${Tensor}();
}

} // namespace at
