#import <Foundation/Foundation.h>
#import <Metal/Metal.h>
#import <MetalPerformanceShaders/MetalPerformanceShaders.h>

@interface MetalCommandBuffer : NSObject
@property(nonatomic, weak, readonly) NSThread* thread;
@property(nonatomic, strong, readonly) id<MTLCommandBuffer> buffer;

+ (MetalCommandBuffer*)newBuffer;
+ (MetalCommandBuffer*)currentBuffer;
- (void)synchronize;

- (void)add:(MPSTemporaryImage*)image;
- (void)remove:(MPSTemporaryImage*)image;

@end
