#import <ATen/ATen.h>
#import <CoreML/CoreML.h>

#include <vector>

struct PTMCoreMLFeatureSpecs {
  NSString* name;
  at::Tensor tensor;
};

API_AVAILABLE(ios(11.0), macos(10.13))
@interface PTMCoreMLFeatureProvider : NSObject<MLFeatureProvider>

- (instancetype)initWithFeatureSpecs:
                    (const std::vector<PTMCoreMLFeatureSpecs>&)specs
                       coreMLVersion:(NSUInteger)ver;

@end
