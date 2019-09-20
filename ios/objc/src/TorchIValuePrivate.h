#import <LibTorch/LibTorch.h>
#import "TorchIValue.h"

NS_ASSUME_NONNULL_BEGIN

@interface TorchIValue ()

- (at::IValue)toIValue;
+ (TorchIValue*)newWithIValue:(const at::IValue&)value;

@end

NS_ASSUME_NONNULL_END
