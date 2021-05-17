#import <Foundation/Foundation.h>
#import <Metal/Metal.h>
#import <MetalPerformanceShaders/MetalPerformanceShaders.h>
#include <string>

API_AVAILABLE(ios(10.0), macos(10.13))
// TODO[T79947194]: Convert this class to C++
@interface MPSCNNContext : NSObject
@property(nonatomic, strong, readonly) id<MTLDevice> device;
@property(nonatomic, strong, readonly) id<MTLCommandQueue> commandQueue;
@property(nonatomic, strong, readonly) id<MTLLibrary> library;

+ (instancetype)sharedInstance;
- (BOOL)available;
- (id<MTLComputePipelineState>)pipelineState:(const std::string&)kernel;
- (id<MTLComputePipelineState>)specializedPipelineState:(const std::string&)kernel
                                              Constants:(NSArray<NSNumber*>*)
                                                            constants;

@end
