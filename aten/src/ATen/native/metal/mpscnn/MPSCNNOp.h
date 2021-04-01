#import <Metal/Metal.h>
#import <MetalPerformanceShaders/MetalPerformanceShaders.h>

#if (defined(__ARM_NEON__) || defined(__ARM_NEON))
typedef float16_t fp16;
#else
typedef uint16_t fp16;
#endif

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
