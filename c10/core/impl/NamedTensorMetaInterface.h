#pragma once

#include <c10/macros/Macros.h>
#include <c10/util/Exception.h>

#include <memory>
#include <numeric>

namespace c10 {

struct C10_API NamedTensorMetaInterface {
  virtual ~NamedTensorMetaInterface() = default;
  virtual std::unique_ptr<NamedTensorMetaInterface> clone() const {
    TORCH_INTERNAL_ASSERT(
        false, "Not implemented: NamedTensorMetaInterface::clone");
  };
  virtual int64_t slow_dim() const {
    TORCH_INTERNAL_ASSERT(
        false, "Not implemented: NamedTensorMetaInterface::slow_dim");
  };
};

} // namespace c10
