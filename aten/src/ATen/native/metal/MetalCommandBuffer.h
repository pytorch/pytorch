#import <MetalPerformanceShaders/MetalPerformanceShaders.h>

@protocol PTMetalCommandBuffer<NSObject>
@optional
- (void)beginSynchronization;
- (void)endSynchronization:(NSError*)error;
@end

@interface MetalCommandBuffer : NSObject
@property(nonatomic, strong, readonly) id<MTLCommandBuffer> buffer;
@property(nonatomic, assign, readonly) BOOL valid;

+ (MetalCommandBuffer*)newBuffer;
+ (MetalCommandBuffer*)currentBuffer;
- (void)addSubscriber:(id<PTMetalCommandBuffer>)subscriber;
- (void)removeSubscriber:(id<PTMetalCommandBuffer>)subscriber;
- (void)commit;
- (void)add:(MPSTemporaryImage*)image;
- (void)remove:(MPSTemporaryImage*)image;

@end
