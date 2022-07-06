#import <CoreML/CoreML.h>

#include <string>

NS_ASSUME_NONNULL_BEGIN

@interface PTMCoreMLCompiler : NSObject

+ (void)setModelCacheDirectory:(const std::string&)dir;

+ (NSString*)modelCacheDirectory;

+ (nullable MLModel*)compileMLModel:(const std::string&)modelSpecs
                         identifier:(const std::string&)identifier
                            backend:(const std::string&)backend
                  allowLowPrecision:(BOOL)allowLowPrecision;

@end

NS_ASSUME_NONNULL_END
