#pragma once

// ${generated_comment}

#include "ATen/CPUTypeDefault.h"
#include "ATen/Context.h"
#include "ATen/CheckGenerator.h"

$extra_cuda_headers

#ifdef _MSC_VER
#ifdef Type
#undef Type
#endif
#endif

namespace at {

struct ${Type} final : public ${DenseBackend}TypeDefault {
  explicit ${Type}();
  virtual ScalarType scalarType() const override;
  virtual caffe2::TypeMeta typeMeta() const override;
  virtual Backend backend() const override;
  virtual const char * toString() const override;
  virtual size_t elementSizeInBytes() const override;
  virtual TypeID ID() const override;

  // example
  // virtual Tensor * add(Tensor & a, Tensor & b) override;

  virtual Tensor & s_copy_(Tensor & self, const Tensor & src, bool non_blocking) const override;
  virtual Tensor & _s_copy_from(const Tensor & self, Tensor & dst, bool non_blocking) const override;
  ${type_derived_method_declarations}
};

} // namespace at
