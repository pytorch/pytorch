#import <Foundation/Foundation.h>
#import <Metal/Metal.h>
#import <MetalPerformanceShaders/MetalPerformanceShaders.h>
#include <string>

API_AVAILABLE(ios(11.0), macos(10.13))
@interface MetalContext : NSObject
@property(nonatomic, strong, readonly) id<MTLDevice> device;
@property(nonatomic, strong, readonly) id<MTLCommandQueue> commandQueue;
@property(nonatomic, strong, readonly) id<MTLLibrary> library;

+ (instancetype)sharedInstance;
- (BOOL)available;
- (id<MTLComputePipelineState>)pipelineState:(const std::string&)kernel;
- (id<MTLComputePipelineState>)specializedPipelineState:(const std::string&)kernel
                                              Constants:(NSArray<NSNumber*>*)
                                                            constants;
- (id<MTLBuffer>)emptyMTLBuffer:(int64_t) size;

@end
