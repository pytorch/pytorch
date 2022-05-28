#import <Foundation/Foundation.h>

namespace torch {
namespace jit {
namespace mobile {
namespace coreml {

enum TensorType {
  Float,
  Double,
  Int,
  Long,
  Undefined,
};

struct TensorSpec {
 public:
  TensorSpec() = delete;

  TensorSpec(NSArray<NSString*>* spec) {
    name_ = spec[0];
    dtype_ = (TensorType)spec[1].intValue;
  }

  NSString* name() {
    return name_;
  }

  TensorType dtype() {
    return dtype_;
  }

 private:
  NSString* name_ = @"";
  TensorType dtype_ = TensorType::Float;
};

} // namespace
}
}
}
