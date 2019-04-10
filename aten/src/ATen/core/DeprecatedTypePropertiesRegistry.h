#pragma once

// In order to preserve bc, we make DeprecatedTypeProperties instances unique
// just like they are for Type.

#include <c10/core/Backend.h>
#include <c10/core/ScalarType.h>

namespace at {

struct DeprecatedTypeProperties;

struct CAFFE2_API DeprecatedTypePropertiesDeleter {
  void operator()(DeprecatedTypeProperties * ptr);
};

class CAFFE2_API DeprecatedTypePropertiesRegistry {
 public:
  using DeprecatedTypePropertiesUniquePtr =
      std::unique_ptr<DeprecatedTypeProperties, DeprecatedTypePropertiesDeleter>;

  DeprecatedTypePropertiesRegistry();

  DeprecatedTypeProperties& getDeprecatedTypeProperties(Backend p, ScalarType s, bool is_variable) const;

private:
  DeprecatedTypePropertiesUniquePtr registry
    [static_cast<int>(Backend::NumOptions)]
    [static_cast<int>(ScalarType::NumOptions)]
    [2];  // is_variable
};

CAFFE2_API DeprecatedTypePropertiesRegistry& globalDeprecatedTypePropertiesRegistry();

} // namespace at
