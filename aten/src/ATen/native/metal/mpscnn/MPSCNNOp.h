#import <Metal/Metal.h>
#import <MetalPerformanceShaders/MetalPerformanceShaders.h>

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
