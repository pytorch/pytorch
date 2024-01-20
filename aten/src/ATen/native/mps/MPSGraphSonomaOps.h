#pragma once
#include <MetalPerformanceShadersGraph/MetalPerformanceShadersGraph.h>
#include <MetalPerformanceShadersGraph/MPSGraphFourierTransformOps.h>

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

-(MPSGraphTensor * _Nonnull) fastFourierTransformWithTensor:(MPSGraphTensor * _Nonnull) tensor
                                                      axes:(NSArray<NSNumber *> * _Nonnull) axes
                                                descriptor:(MPSGraphFFTDescriptor * _Nonnull) descriptor
                                                      name:(NSString * _Nullable) name;

-(MPSGraphTensor * _Nonnull) fastFourierTransformWithTensor:(MPSGraphTensor * _Nonnull) tensor
                                                axesTensor:(MPSGraphTensor * _Nonnull) axesTensor
                                                descriptor:(MPSGraphFFTDescriptor * _Nonnull) descriptor
                                                      name:(NSString * _Nullable) name;

-(MPSGraphTensor * _Nonnull) realToHermiteanFFTWithTensor:(MPSGraphTensor * _Nonnull) tensor
                                                      axes:(NSArray<NSNumber *> * _Nonnull) axes
                                                descriptor:(MPSGraphFFTDescriptor * _Nonnull) descriptor
                                                      name:(NSString * _Nullable) name;

-(MPSGraphTensor * _Nonnull) realToHermiteanFFTWithTensor:(MPSGraphTensor * _Nonnull) tensor
                                                axesTensor:(MPSGraphTensor * _Nonnull) axesTensor
                                                descriptor:(MPSGraphFFTDescriptor * _Nonnull) descriptor
                                                      name:(NSString * _Nullable) name;

-(MPSGraphTensor * _Nonnull) HermiteanToRealFFTWithTensor:(MPSGraphTensor * _Nonnull) tensor
                                                      axes:(NSArray<NSNumber *> * _Nonnull) axes
                                                descriptor:(MPSGraphFFTDescriptor * _Nonnull) descriptor
                                                      name:(NSString * _Nullable) name;

-(MPSGraphTensor * _Nonnull) HermiteanToRealFFTWithTensor:(MPSGraphTensor * _Nonnull) tensor
                                                axesTensor:(MPSGraphTensor * _Nonnull) axesTensor
                                                descriptor:(MPSGraphFFTDescriptor * _Nonnull) descriptor
                                                      name:(NSString * _Nullable) name;

@end
