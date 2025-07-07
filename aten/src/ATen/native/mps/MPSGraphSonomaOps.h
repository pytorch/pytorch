#pragma once

#include <MetalPerformanceShadersGraph/MetalPerformanceShadersGraph.h>

#if !defined(__MAC_14_0) && (!defined(MAC_OS_X_VERSION_14_0) || (MAC_OS_X_VERSION_MIN_REQUIRED < MAC_OS_X_VERSION_14_0))

typedef NS_ENUM(NSUInteger, MPSGraphFFTScalingMode) {
  MPSGraphFFTScalingModeNone = 0L,
  MPSGraphFFTScalingModeSize = 1L,
  MPSGraphFFTScalingModeUnitary = 2L,
};

@interface FakeMPSGraphFFTDescriptor : NSObject<NSCopying>
@property(readwrite, nonatomic) BOOL inverse;
@property(readwrite, nonatomic) MPSGraphFFTScalingMode scalingMode;
@property(readwrite, nonatomic) BOOL roundToOddHermitean;
+ (nullable instancetype)descriptor;
@end

@compatibility_alias MPSGraphFFTDescriptor FakeMPSGraphFFTDescriptor;

@interface MPSGraph (SonomaOps)
- (MPSGraphTensor* _Nonnull)conjugateWithTensor:(MPSGraphTensor* _Nonnull)tensor name:(NSString* _Nullable)name;

- (MPSGraphTensor* _Nonnull)realPartOfTensor:(MPSGraphTensor* _Nonnull)tensor name:(NSString* _Nullable)name;

- (MPSGraphTensor* _Nonnull)fastFourierTransformWithTensor:(MPSGraphTensor* _Nonnull)tensor
                                                      axes:(NSArray<NSNumber*>* _Nonnull)axes
                                                descriptor:(MPSGraphFFTDescriptor* _Nonnull)descriptor
                                                      name:(NSString* _Nullable)name;

- (MPSGraphTensor* _Nonnull)realToHermiteanFFTWithTensor:(MPSGraphTensor* _Nonnull)tensor
                                                    axes:(NSArray<NSNumber*>* _Nonnull)axes
                                              descriptor:(MPSGraphFFTDescriptor* _Nonnull)descriptor
                                                    name:(NSString* _Nullable)name;

- (MPSGraphTensor* _Nonnull)HermiteanToRealFFTWithTensor:(MPSGraphTensor* _Nonnull)tensor
                                                    axes:(NSArray<NSNumber*>* _Nonnull)axes
                                              descriptor:(MPSGraphFFTDescriptor* _Nonnull)descriptor
                                                    name:(NSString* _Nullable)name;
@end

// define BFloat16 enums for MacOS13
#define MPSDataTypeBFloat16 ((MPSDataType)(MPSDataTypeAlternateEncodingBit | MPSDataTypeFloat16))

// define Metal version
#define MTLLanguageVersion3_1 ((MTLLanguageVersion)((3 << 16) + 1))
#endif
