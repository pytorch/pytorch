#import <CoreML/CoreML.h>
#include <torch/script.h>

#include <string>
#include <vector>

struct PTMCoreMLFeatureSpecs {
  NSString* name;
  at::Tensor tensor;
};

API_AVAILABLE(ios(11.0), macos(10.13))
@interface PTMCoreMLFeatureProvider : NSObject<MLFeatureProvider>
- (instancetype)initWithFeatureSpecs:
                    (const std::vector<PTMCoreMLFeatureSpecs>&)specs
                       CoreMLVersion:(NSUInteger)ver;
@end

API_AVAILABLE(ios(11.0), macos(10.13))
@interface PTMCoreMLExecutor : NSObject

@property(nonatomic, copy) NSString* backend;
@property(nonatomic, assign) BOOL allowLowPrecision;
@property(nonatomic, assign) NSUInteger coreMLVersion;

+ (BOOL)isAvailable;
+ (void)setModelCacheDirectory:(NSString*)dir;
+ (NSString*)modelCacheDirectory;

- (BOOL)compileMLModel:(const std::string&)modelSpecs
            identifier:(const std::string&)identifier;
- (id<MLFeatureProvider>)forwardWithInputs:
    (const std::vector<PTMCoreMLFeatureSpecs>&)inputs;
- (BOOL)cleanup;

@end
