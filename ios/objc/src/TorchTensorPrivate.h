#import <LibTorch/LibTorch.h>
#import "TorchTensor.h"

NS_ASSUME_NONNULL_BEGIN

@interface TorchTensor ()

- (at::Tensor)toTensor;
+ (TorchTensor*)newWithTensor:(const at::Tensor&)tensor;

@end

NS_ASSUME_NONNULL_END
