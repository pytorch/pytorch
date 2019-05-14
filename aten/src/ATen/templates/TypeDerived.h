#pragma once

// ${generated_comment}

#include <ATen/CPUTypeDefault.h>
#include <ATen/Context.h>
#include <ATen/CheckGenerator.h>

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
};

} // namespace at
