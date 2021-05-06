#import <Foundation/Foundation.h>
#import <Metal/Metal.h>
#import <MetalPerformanceShaders/MetalPerformanceShaders.h>

API_AVAILABLE(ios(10.0), macos(10.13))
@interface MPSCNNContext : NSObject
@property(nonatomic, strong, readonly) id<MTLDevice> device;
@property(nonatomic, strong, readonly) id<MTLCommandQueue> commandQueue;
@property(nonatomic, strong, readonly) id<MTLLibrary> library;

+ (instancetype)sharedInstance;
- (BOOL)available;
- (id<MTLComputePipelineState>)pipelineState:(NSString*)kernel;
- (id<MTLComputePipelineState>)specializedPipelineState:(NSString*)kernel
                                              Constants:(NSArray<NSNumber*>*)
                                                            constants;

@end
