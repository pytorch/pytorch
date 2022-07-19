#import <CoreML/CoreML.h>

#include <string>

NS_ASSUME_NONNULL_BEGIN

@interface PTMCoreMLCompiler : NSObject

+ (void)setModelCacheDirectory:(const std::string&)dir;

+ (NSString*)modelCacheDirectory;

+ (NSURL*)compileModel:(const std::string&)modelSpecs
               modelID:(const std::string&)modelID;

+ (nullable MLModel*)loadCPUModelAtURL:(NSURL*)modelURL;

+ (nullable MLModel*)loadModelAtURL:(NSURL*)modelURL
                            backend:(const std::string&)backend
                  allowLowPrecision:(BOOL)allowLowPrecision;

@end

NS_ASSUME_NONNULL_END
