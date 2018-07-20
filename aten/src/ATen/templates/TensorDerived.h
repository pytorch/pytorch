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
  virtual ~${Tensor}();
  virtual const char * toString() const override;
  virtual IntList sizes() const override;
  virtual IntList strides() const override;
  virtual int64_t dim() const override;
  virtual Scalar localScalar() override;
  virtual void * unsafeGetTH(bool retain) override;
  virtual std::unique_ptr<Storage> storage() override;
  virtual void release_resources() override;
  static const char * typeString();

  THTensor * tensor;
};

namespace detail {
  // This is just a temporary function to help out code generation.
  // Eventually, the codegen code should construct tensors using
  // a new Tensor constructor that takes scalar type and backend,
  // but I haven't written this yet.
  ${Tensor}* new_${Tensor}();
}

} // namespace at
