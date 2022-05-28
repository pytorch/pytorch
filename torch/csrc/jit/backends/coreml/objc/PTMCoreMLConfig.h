#import <Foundation/Foundation.h>

namespace torch {
namespace jit {
namespace mobile {
namespace coreml {

static constexpr int SUPPORTED_COREML_VER = 4;

struct CoreMLConfig {
 public:
  CoreMLConfig() = delete;

  CoreMLConfig(NSDictionary* dict)
      : coreMLVersion_([dict[@"spec_ver"] intValue]),
        backend_([dict[@"backend"] lowercaseString]),
        allow_low_precision_([dict[@"allow_low_precision"] boolValue]) {}

  int64_t coreMLVersion() const {
    return coreMLVersion_;
  }

  NSString* backend() const {
    return backend_;
  }

  bool allowLowPrecision() const {
    return allow_low_precision_;
  }

 private:
  int64_t coreMLVersion_ = SUPPORTED_COREML_VER;
  NSString* backend_ = @"CPU";
  bool allow_low_precision_ = true;
};

} // namespace
}
}
}
