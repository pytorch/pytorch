#include <ATen/Tensor.h>
#import <ATen/native/metal/MetalCommandBuffer.h>
#import <Metal/Metal.h>
#import <MetalPerformanceShaders/MetalPerformanceShaders.h>

@interface MPSImage (Tensor)

+ (MPSImage*)imageFromCPUTensor:(const at::Tensor&)tensor;
- (at::Tensor)toCPUTensor;

+ (MPSImage*)imageFromFp16Array:(const uint16_t*)src
                          Sizes:(const std::vector<int64_t>&)sizes;
- (std::vector<uint16_t>)toFp16Array;

+ (MPSImage*)imageFromSize:(const std::vector<int64_t>&)size;
+ (MPSTemporaryImage*)temporaryImageFromSize:(const std::vector<int64_t>&)size
                               commandBuffer:(MetalCommandBuffer*)cmdBuffer;

- (std::vector<int64_t>)sizes;
- (int64_t)readCount;
- (BOOL)isTemporaryImage;
- (void)markRead;
- (void)recycle;

@end

@interface MPSImage (Shaders)

+ (MPSImage*)imageFromImage:(MPSImage*)image;

+ (MPSTemporaryImage*)temporaryImageFromImage:(MPSImage*)image
                                CommandBuffer:(MetalCommandBuffer*)cb;

+ (MPSImage*)imageFromTemporaryImage:(MPSTemporaryImage*)image
                       CommandBuffer:(MetalCommandBuffer*)cb
                  waitUntilCompleted:(BOOL)b;

+ (MPSImage*)imageFromHost:(const float*)src
                     Sizes:(const std::vector<int64_t>&)sizes;

+ (MPSTemporaryImage*)temporaryImageFromHost:(const float*)src
                                       Sizes:(const std::vector<int64_t>&)sizes
                               CommandBuffer:(MetalCommandBuffer*)cb;

+ (void)copyToHost:(float*)dst FromImage:(MPSImage*)image;

@end
