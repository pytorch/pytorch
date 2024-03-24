#ifdef BUILD_LITE_INTERPRETER

#import <Foundation/Foundation.h>

NS_ASSUME_NONNULL_BEGIN

@interface Benchmark : NSObject

+ (BOOL)setup:(NSDictionary* )config;
+ (NSString* )run;

@end

NS_ASSUME_NONNULL_END
#endif
