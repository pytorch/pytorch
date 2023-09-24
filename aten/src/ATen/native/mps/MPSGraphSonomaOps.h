#pragma once
#include <MetalPerformanceShadersGraph/MetalPerformanceShadersGraph.h>

@interface MPSGraph (SonomaOps)

#if !defined(__MAC_14_0) && \
    (!defined(MAC_OS_X_VERSION_14_0) || (MAC_OS_X_VERSION_MIN_REQUIRED < MAC_OS_X_VERSION_14_0))

typedef NS_ENUM(NSUInteger, MPSGraphFFTScalingMode)
{
    MPSGraphFFTScalingModeNone          = 0L,
    MPSGraphFFTScalingModeSize          = 1L,
    MPSGraphFFTScalingModeUnitary       = 2L,
};
#endif

-(MPSGraphTensor * _Nonnull) fastFourierTransformWithTensor:(MPSGraphTensor *) tensor
                                              axes:(NSArray<NSNumber *> *) axes
                                        descriptor:(MPSGraphFFTDescriptor *) descriptor
                                              name:(NSString * _Nullable) name;

-(MPSGraphTensor * _Nonnull) fastFourierTransformWithTensor:(MPSGraphTensor *) tensor
                                        axesTensor:(MPSGraphTensor *) axesTensor
                                        descriptor:(MPSGraphFFTDescriptor *) descriptor
                                              name:(NSString * _Nullable) name;

-(MPSGraphTensor * _Nonnull) realToHermiteanFFTWithTensor:(MPSGraphTensor *) tensor
                                            axes:(NSArray<NSNumber *> *) axes
                                      descriptor:(MPSGraphFFTDescriptor *) descriptor
                                            name:(NSString * _Nullable) name;

-(MPSGraphTensor * _Nonnull) realToHermiteanFFTWithTensor:(MPSGraphTensor *) tensor
                                      axesTensor:(MPSGraphTensor *) axesTensor
                                      descriptor:(MPSGraphFFTDescriptor *) descriptor
                                            name:(NSString * _Nullable) name;

-(MPSGraphTensor * _Nonnull) HermiteanToRealFFTWithTensor:(MPSGraphTensor *) tensor
                                            axes:(NSArray<NSNumber *> *) axes
                                      descriptor:(MPSGraphFFTDescriptor *) descriptor
                                            name:(NSString * _Nullable) name;

-(MPSGraphTensor * _Nonnull) HermiteanToRealFFTWithTensor:(MPSGraphTensor *) tensor
                                      axesTensor:(MPSGraphTensor *) axesTensor
                                      descriptor:(MPSGraphFFTDescriptor *) descriptor
                                            name:(NSString * _Nullable) name;

@end
