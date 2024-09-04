#import <torch/csrc/jit/backends/coreml/objc/PTMCoreMLFeatureProvider.h>

#import <CoreML/CoreML.h>

NS_ASSUME_NONNULL_BEGIN

@interface PTMCoreMLExecutor : NSObject

@property(atomic, strong) MLModel* model;

- (instancetype)initWithFeatureNames:(NSArray<NSString*>*)featureNames;

- (void)setInputs:(c10::impl::GenericList)inputs;

- (id<MLFeatureProvider>)forward:(NSError**)error;

@end

NS_ASSUME_NONNULL_END
