#import <torch/csrc/jit/backends/coreml/objc/PTMCoreMLFeatureProvider.h>

@implementation PTMCoreMLFeatureProvider {
  NSMutableDictionary *_featureValuesForName;
}

@synthesize featureNames = _featureNames;

- (instancetype)initWithFeatureNames:(NSSet<NSString *> *)featureNames {
  if (self = [super init]) {
    _featureNames = featureNames;
    _featureValuesForName = [NSMutableDictionary dictionary];
  }
  return self;
}

- (void)clearInputTensors {
  @synchronized(_featureValuesForName) {
    [_featureValuesForName removeAllObjects];
  }
}

- (void)setInputTensor:(const at::Tensor&)tensor forFeatureName:(NSString *)name {
  NSMutableArray *shape = [NSMutableArray new];
  for (auto& dim : tensor.sizes().vec()) {
    [shape addObject:@(dim)];
  }

  NSMutableArray *strides = [NSMutableArray new];
  for (auto& step : tensor.strides().vec()) {
    [strides addObject:@(step)];
  }

  NSError* error = nil;
  MLMultiArray *mlArray =
    [[MLMultiArray alloc]
     initWithDataPointer:tensor.mutable_data_ptr<float>()
     shape:shape
     dataType:MLMultiArrayDataTypeFloat32
     strides:strides
     deallocator:(^(void* bytes){})
     error:&error];
  MLFeatureValue *value = [MLFeatureValue featureValueWithMultiArray:mlArray];
  if (value) {
    @synchronized(_featureValuesForName) {
      _featureValuesForName[name] = value;
    }
  }
}

- (nullable MLFeatureValue *)featureValueForName:(NSString *)featureName {
  @synchronized(_featureValuesForName) {
    return _featureValuesForName[featureName];
  }
}

@end
