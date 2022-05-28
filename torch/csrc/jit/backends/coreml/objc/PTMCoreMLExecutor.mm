#import <torch/csrc/jit/backends/coreml/objc/PTMCoreMLExecutor.h>
#import <torch/csrc/jit/backends/coreml/objc/PTMCoreMLError.h>

#import <CoreML/CoreML.h>

@implementation PTMCoreMLExecutor

+ (id<MLFeatureProvider>)forward:(const std::vector<PTMCoreMLFeatureSpecs>&)inputs model:(MLModel *)model coreMLVersion:(NSUInteger)coreMLVersion error:(NSError **)error {
  if (!@available(iOS 11.0, macOS 10.13, *)) {
    *error = [NSError errorWithDomain:kPTMCoreMLErrorDomain code:PTMCoreMLErrorCodeOSVersion userInfo:nil];
    return nil;
  }

  PTMCoreMLFeatureProvider *inputFeature =
    [[PTMCoreMLFeatureProvider alloc] initWithFeatureSpecs:inputs coreMLVersion:coreMLVersion];
  if (inputFeature == nil) {
    return nil;
  }

  MLPredictionOptions *options = [[MLPredictionOptions alloc] init];
  id<MLFeatureProvider> outputFeature = [model predictionFromFeatures:inputFeature options:options error:error];

  if (*error) {
    return nil;
  }

  return outputFeature;
}

@end
