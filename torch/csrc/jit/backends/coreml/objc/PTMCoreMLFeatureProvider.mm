#import <torch/csrc/jit/backends/coreml/objc/PTMCoreMLFeatureProvider.h>

@implementation PTMCoreMLFeatureProvider {
  NSUInteger _coremlVersion;
  std::vector<PTMCoreMLFeatureSpecs> _specs;
}

@synthesize featureNames = _featureNames;

- (instancetype)initWithFeatureSpecs:(const std::vector<PTMCoreMLFeatureSpecs>&)specs coreMLVersion:(NSUInteger)ver {
  if (self = [super init]) {
    _coremlVersion = ver;
    _specs = specs;
    NSMutableArray *names = [NSMutableArray new];
    for (auto& spec : _specs) {
      [names addObject:spec.name];
    }
    _featureNames = [[NSSet alloc] initWithArray:names];
  }
  return self;
}

- (nullable MLFeatureValue *)featureValueForName:(NSString *)featureName {

  for (auto& spec : _specs) {
    if (![spec.name isEqualToString:featureName]) {
      continue;
    }

    TORCH_CHECK(spec.tensor.dtype() == c10::kFloat);

    NSMutableArray *shape = [NSMutableArray new];
    for (auto& dim : spec.tensor.sizes().vec()) {
      [shape addObject:@(dim)];
    }

    NSMutableArray *strides = [NSMutableArray new];
    for (auto& step : spec.tensor.strides().vec()) {
      [strides addObject:@(step)];
    }

    NSError* error = nil;
    MLMultiArray *mlArray =
      [[MLMultiArray alloc]
        initWithDataPointer:spec.tensor.data_ptr<float>()
                      shape:shape
                    dataType:MLMultiArrayDataTypeFloat32
                    strides:strides
                deallocator:(^(void* bytes){})
                      error:&error];
    return [MLFeatureValue featureValueWithMultiArray:mlArray];
  }

  return nil;
}

@end
