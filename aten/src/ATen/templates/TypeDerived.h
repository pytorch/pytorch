#pragma once

// ${generated_comment}

#include <ATen/CPUTypeDefault.h>
#include <ATen/Context.h>
#include <ATen/Utils.h>

$extra_cuda_headers

#ifdef _MSC_VER
#ifdef Type
#undef Type
#endif
#endif

namespace at {

struct ${Type} final : public ${DeviceType}TypeDefault {
  explicit ${Type}();
  virtual Backend backend() const override;
  virtual const char * toString() const override;
  virtual TypeID ID() const override;

  // example
  // virtual Tensor * add(Tensor & a, Tensor & b) override;
  ${type_derived_method_declarations}

 private:
  ScalarType infer_scalar_type(const Tensor & t) const {
    return t.scalar_type();
  }
  ScalarType infer_scalar_type(const TensorList & tl) const {
    TORCH_CHECK(tl.size() > 0, "expected a non-empty list of Tensors");
    return tl[0].scalar_type();
  }
};

} // namespace at
