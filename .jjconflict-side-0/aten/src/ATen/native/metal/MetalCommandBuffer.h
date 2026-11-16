#include <c10/macros/Macros.h>

C10_DIAGNOSTIC_PUSH_AND_IGNORED_IF_DEFINED("-Wdeprecated-declarations")
#import <MetalPerformanceShaders/MetalPerformanceShaders.h>
C10_DIAGNOSTIC_POP()

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
