#include <c10/macros/Macros.h>

C10_DIAGNOSTIC_PUSH_AND_IGNORED_IF_DEFINED("-Wdeprecated-declarations")
#import <Metal/Metal.h>
#import <MetalPerformanceShaders/MetalPerformanceShaders.h>
C10_DIAGNOSTIC_POP()

@protocol MPSCNNOp<NSObject>

@property(nonatomic, strong) MPSCNNKernel* kernel;

- (void)encode:(id<MTLCommandBuffer>)cb
         sourceImage:(MPSImage*)src
    destinationImage:(MPSImage*)dst;

@end

@protocol MPSCNNShaderOp<NSObject>

+ (id<MPSCNNShaderOp>)newWithTextures:(NSArray<MPSImage*>*)textures
                                 Args:(NSArray<NSNumber*>*)args;
- (void)encode:(id<MTLCommandBuffer>)cb;

@end
