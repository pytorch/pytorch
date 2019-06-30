#pragma once

#include <ATen/TypeDefault.h>
#include <ATen/Utils.h>

#ifdef _MSC_VER
#ifdef Type
#undef Type
#endif
#endif

namespace at {

struct UndefinedType final : public TypeDefault {
  explicit UndefinedType();
  virtual const char * toString() const override;
  virtual TypeID ID() const override;
};

} // namespace at
