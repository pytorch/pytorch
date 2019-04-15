#pragma once
#include <ATen/${Backend}Type.h>

namespace at {

struct CAFFE2_API ${Type} : public ${Backend}Type {
  explicit ${Type}();

  virtual ScalarType scalarType() const override;
  virtual caffe2::TypeMeta typeMeta() const override;
  virtual const char * toString() const override;
  virtual TypeID ID() const override;
};

} // namespace at
