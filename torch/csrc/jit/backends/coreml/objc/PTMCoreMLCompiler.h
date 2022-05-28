#import <CoreML/CoreML.h>

#include <string>

NS_ASSUME_NONNULL_BEGIN

@interface PTMCoreMLCompiler : NSObject

+ (void)setModelCacheDirectory:(NSString*)dir;

+ (NSString*)modelCacheDirectory;

+ (nullable NSDictionary*)parseSpecs:(const char*)json error:(NSError**)error;

+ (nullable MLModel*)compileMLModel:(const std::string&)modelSpecs
                         identifier:(const std::string&)identifier
                            backend:(NSString*)backend
                  allowLowPrecision:(BOOL)allowLowPrecision
                              error:(NSError**)error;

@end

NS_ASSUME_NONNULL_END
