#import <Foundation/Foundation.h>
#import <Metal/Metal.h>
#import <MetalPerformanceShaders/MetalPerformanceShaders.h>

@protocol PTMetalCommandBufferDelegate<NSObject>
@optional
- (void)prepareForSynchronization;
@end

@interface MetalCommandBuffer : NSObject
@property(nonatomic, weak, readonly) NSThread* thread;
@property(nonatomic, strong, readonly) id<MTLCommandBuffer> buffer;

+ (MetalCommandBuffer*)newBuffer;
+ (MetalCommandBuffer*)currentBuffer;
- (void)addDelegate:(id<PTMetalCommandBufferDelegate>)delegate;
- (void)removeDelegate:(id<PTMetalCommandBufferDelegate>)delegate;
- (void)synchronize;
- (void)add:(MPSTemporaryImage*)image;
- (void)remove:(MPSTemporaryImage*)image;

@end
