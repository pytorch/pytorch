#pragma once

// In order to preserve bc, we make DeprecatedTypeProperties instances unique
// just like they are for Type.

#include <c10/core/Backend.h>
#include <c10/core/ScalarType.h>
#include <ATen/core/DeprecatedTypeProperties.h>

namespace at {

struct CAFFE2_API DeprecatedTypePropertiesDeleter {
  void operator()(DeprecatedTypeProperties * ptr) {
      delete ptr;
  }
};

class CAFFE2_API DeprecatedTypePropertiesRegistry {
 public:
  using DeprecatedTypePropertiesUniquePtr =
      std::unique_ptr<DeprecatedTypeProperties, DeprecatedTypePropertiesDeleter>;

  DeprecatedTypePropertiesRegistry() {
    for (int b = 0; b < static_cast<int>(Backend::NumOptions); ++b) {
      for (int s = 0; s < static_cast<int>(ScalarType::NumOptions); ++s) {
        registry[b][s] = DeprecatedTypePropertiesUniquePtr{
            new DeprecatedTypeProperties(static_cast<Backend>(b), static_cast<ScalarType>(s)),
            DeprecatedTypePropertiesDeleter()
        };
      }
    }
  }

  DeprecatedTypeProperties& getDeprecatedTypeProperties(Backend p, ScalarType s) {
    return *registry[static_cast<int>(p)][static_cast<int>(s)];
  }

private:
  DeprecatedTypePropertiesUniquePtr registry
    [static_cast<int>(Backend::NumOptions)]
    [static_cast<int>(ScalarType::NumOptions)];
};

CAFFE2_API DeprecatedTypePropertiesRegistry& globalDeprecatedTypePropertiesRegistry();

} // namespace at
