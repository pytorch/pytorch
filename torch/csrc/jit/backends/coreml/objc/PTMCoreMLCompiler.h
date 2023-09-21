#import <CoreML/CoreML.h>

#include <string>

NS_ASSUME_NONNULL_BEGIN

@interface PTMCoreMLCompiler : NSObject

+ (void)setCacheDirectory:(const std::string&)dir;

+ (NSString*)cacheDirectory;

+ (BOOL)compileModel:(const std::string&)modelSpecs modelID:(const std::string&)modelID;

+ (nullable MLModel*)loadModel:(const std::string)modelID
                       backend:(const std::string)backend
             allowLowPrecision:(BOOL)allowLowPrecision;

@end

NS_ASSUME_NONNULL_END
