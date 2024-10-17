#import <ATen/ATen.h>
#import <CoreML/CoreML.h>

NS_ASSUME_NONNULL_BEGIN

@interface PTMCoreMLFeatureProvider : NSObject<MLFeatureProvider>

- (instancetype)initWithFeatureNames:(NSSet<NSString*>*)featureNames;

- (void)clearInputTensors;

- (void)setInputTensor:(const at::Tensor&)tensor forFeatureName:(NSString*)name;

@end

NS_ASSUME_NONNULL_END
