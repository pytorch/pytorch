#import <torch/csrc/jit/backends/coreml/objc/PTMCoreMLFeatureProvider.h>

#import <CoreML/CoreML.h>

#include <vector>

NS_ASSUME_NONNULL_BEGIN

@interface PTMCoreMLExecutor : NSObject

+ (nullable id<MLFeatureProvider>)
          forward:(const std::vector<PTMCoreMLFeatureSpecs>&)inputs
            model:(MLModel*)model
    coreMLVersion:(NSUInteger)coreMLVersion
            error:(NSError**)error;

@end

NS_ASSUME_NONNULL_END
