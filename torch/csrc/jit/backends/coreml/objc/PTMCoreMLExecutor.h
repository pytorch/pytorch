#import <torch/csrc/jit/backends/coreml/objc/PTMCoreMLFeatureProvider.h>

#import <CoreML/CoreML.h>

NS_ASSUME_NONNULL_BEGIN

@interface PTMCoreMLExecutor : NSObject

- (instancetype)initWithModel:(MLModel*)model
                 featureNames:(NSArray<NSString*>*)featureNames;

- (void)setInputs:(c10::impl::GenericList)inputs;

- (id<MLFeatureProvider>)forward;

@end

NS_ASSUME_NONNULL_END
