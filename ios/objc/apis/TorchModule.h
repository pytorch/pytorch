#import <Foundation/Foundation.h>

NS_ASSUME_NONNULL_BEGIN

@class TorchIValue;
@interface TorchModule : NSObject

+ (nullable TorchModule*)loadTorchscriptModel:(NSString*)modelPath;
- (nullable TorchIValue*)forward:(NSArray<TorchIValue*>*)values;
- (nullable TorchIValue*)run_method:(NSString*)methodName withInputs:(NSArray<TorchIValue*>*)inputs;

@end

NS_ASSUME_NONNULL_END
