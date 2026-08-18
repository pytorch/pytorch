#include <c10/macros/Macros.h>

C10_DIAGNOSTIC_PUSH_AND_IGNORED_IF_DEFINED("-Wdeprecated-declarations")
#import <Metal/Metal.h>
#import <MetalPerformanceShaders/MetalPerformanceShaders.h>
C10_DIAGNOSTIC_POP()
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
