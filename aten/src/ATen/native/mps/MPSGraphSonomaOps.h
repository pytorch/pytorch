#pragma once
#include <MetalPerformanceShadersGraph/MetalPerformanceShadersGraph.h>

#if !defined(__MAC_14_0) && \
    (!defined(MAC_OS_X_VERSION_14_0) || (MAC_OS_X_VERSION_MIN_REQUIRED < MAC_OS_X_VERSION_14_0))

typedef NS_ENUM(NSUInteger, MPSGraphFFTScalingMode)
{
    MPSGraphFFTScalingModeNone          = 0L,
    MPSGraphFFTScalingModeSize          = 1L,
    MPSGraphFFTScalingModeUnitary       = 2L,
};

@interface FakeMPSGraphFFTDescriptor : NSObject<NSCopying> 
@property (readwrite, nonatomic) BOOL inverse;
@property (readwrite, nonatomic) MPSGraphFFTScalingMode scalingMode;
@property (readwrite, nonatomic) BOOL roundToOddHermitean;
+(nullable instancetype) descriptor;
@end

@compatibility_alias MPSGraphFFTDescriptor FakeMPSGraphFFTDescriptor;
#endif

@interface MPSGraph (SonomaOps)

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
