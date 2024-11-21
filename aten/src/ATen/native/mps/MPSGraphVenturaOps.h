#pragma once
#include <MetalPerformanceShadersGraph/MetalPerformanceShadersGraph.h>

// TODO: Remove me when moved to MacOS 13
#if !defined(__MAC_13_2) && (!defined(MAC_OS_X_VERSION_13_2) || (MAC_OS_X_VERSION_MIN_REQUIRED < MAC_OS_X_VERSION_13_2))

@interface FakeMPSGraphConvolution3DOpDescriptor : NSObject<NSCopying>

@property(readwrite, nonatomic) NSUInteger strideInX;
@property(readwrite, nonatomic) NSUInteger strideInY;
@property(readwrite, nonatomic) NSUInteger strideInZ;
@property(readwrite, nonatomic) NSUInteger dilationRateInX;
@property(readwrite, nonatomic) NSUInteger dilationRateInY;
@property(readwrite, nonatomic) NSUInteger dilationRateInZ;

@property(readwrite, nonatomic) NSUInteger paddingLeft;
@property(readwrite, nonatomic) NSUInteger paddingRight;
@property(readwrite, nonatomic) NSUInteger paddingTop;
@property(readwrite, nonatomic) NSUInteger paddingBottom;
@property(readwrite, nonatomic) NSUInteger paddingFront;
@property(readwrite, nonatomic) NSUInteger paddingBack;

@property(readwrite, nonatomic) MPSGraphPaddingStyle paddingStyle;
@property(readwrite, nonatomic) MPSGraphTensorNamedDataLayout dataLayout;
@property(readwrite, nonatomic) MPSGraphTensorNamedDataLayout weightsLayout;

@property(readwrite, nonatomic) NSUInteger groups;

@end

@compatibility_alias MPSGraphConvolution3DOpDescriptor FakeMPSGraphConvolution3DOpDescriptor;

#endif

@interface MPSGraph (VenturaOps)

#if !defined(__MAC_13_0) && (!defined(MAC_OS_X_VERSION_13_0) || (MAC_OS_X_VERSION_MIN_REQUIRED < MAC_OS_X_VERSION_13_0))

typedef NS_ENUM(NSUInteger, MPSGraphResizeNearestRoundingMode) {
  MPSGraphResizeNearestRoundingModeRoundPreferCeil = 0L,
  MPSGraphResizeNearestRoundingModeRoundPreferFloor = 1L,
  MPSGraphResizeNearestRoundingModeCeil = 2L,
  MPSGraphResizeNearestRoundingModeFloor = 3L,
  MPSGraphResizeNearestRoundingModeRoundToEven = 4L,
  MPSGraphResizeNearestRoundingModeRoundToOdd = 5L,
};

// Define complex enums for MacOS 12
#define MPSDataTypeComplexBit 0x01000000
#define MPSDataTypeComplexFloat32 ((MPSDataType)(MPSDataTypeFloatBit | MPSDataTypeComplexBit | 64))
#define MPSDataTypeComplexFloat16 ((MPSDataType)(MPSDataTypeFloatBit | MPSDataTypeComplexBit | 32))
#endif

- (MPSGraphTensor* _Nonnull)convolution3DWithSourceTensor:(MPSGraphTensor* _Nonnull)source
                                            weightsTensor:(MPSGraphTensor* _Nonnull)weights
                                               descriptor:(MPSGraphConvolution3DOpDescriptor* _Nonnull)descriptor
                                                     name:(NSString* _Nullable)name;

- (MPSGraphTensor* _Nonnull)
    convolution3DDataGradientWithIncomingGradientTensor:(MPSGraphTensor* _Nonnull)incomingGradient
                                          weightsTensor:(MPSGraphTensor* _Nonnull)weights
                                            outputShape:(MPSShape* _Nonnull)outputShape
                           forwardConvolutionDescriptor:
                               (MPSGraphConvolution3DOpDescriptor* _Nonnull)forwardConvolutionDescriptor
                                                   name:(NSString* _Nullable)name;

- (MPSGraphTensor* _Nonnull)
    convolution3DWeightsGradientWithIncomingGradientTensor:(MPSGraphTensor* _Nonnull)incomingGradient
                                              sourceTensor:(MPSGraphTensor* _Nonnull)source
                                               outputShape:(MPSShape* _Nonnull)outputShape
                              forwardConvolutionDescriptor:
                                  (MPSGraphConvolution3DOpDescriptor* _Nonnull)forwardConvolutionDescriptor
                                                      name:(NSString* _Nullable)name;

- (MPSGraphTensor* _Nonnull)cumulativeSumWithTensor:(MPSGraphTensor* _Nonnull)tensor
                                               axis:(NSInteger)axis
                                               name:(NSString* _Nullable)name;

- (MPSGraphTensor* _Nonnull)sortWithTensor:(MPSGraphTensor* _Nonnull)tensor
                                      axis:(NSInteger)axis
                                      name:(NSString* _Nullable)name;

- (MPSGraphTensor* _Nonnull)sortWithTensor:(MPSGraphTensor* _Nonnull)tensor
                                      axis:(NSInteger)axis
                                descending:(BOOL)descending
                                      name:(NSString* _Nullable)name;

- (MPSGraphTensor* _Nonnull)sortWithTensor:(MPSGraphTensor* _Nonnull)tensor
                                axisTensor:(MPSGraphTensor* _Nonnull)axisTensor
                                descending:(BOOL)descending
                                      name:(NSString* _Nullable)name;

- (MPSGraphTensor* _Nonnull)sortWithTensor:(MPSGraphTensor* _Nonnull)tensor
                                axisTensor:(MPSGraphTensor* _Nonnull)axisTensor
                                      name:(NSString* _Nullable)name;

- (MPSGraphTensor* _Nonnull)argSortWithTensor:(MPSGraphTensor* _Nonnull)tensor
                                         axis:(NSInteger)axis
                                         name:(NSString* _Nullable)name;

- (MPSGraphTensor* _Nonnull)argSortWithTensor:(MPSGraphTensor* _Nonnull)tensor
                                         axis:(NSInteger)axis
                                   descending:(BOOL)descending
                                         name:(NSString* _Nullable)name;

- (MPSGraphTensor* _Nonnull)argSortWithTensor:(MPSGraphTensor* _Nonnull)tensor
                                   axisTensor:(MPSGraphTensor* _Nonnull)axisTensor
                                   descending:(BOOL)descending
                                         name:(NSString* _Nullable)name;

- (MPSGraphTensor* _Nonnull)argSortWithTensor:(MPSGraphTensor* _Nonnull)tensor
                                   axisTensor:(MPSGraphTensor* _Nonnull)axisTensor
                                         name:(NSString* _Nullable)name;

- (MPSGraphTensor* _Nonnull)inverseOfTensor:(MPSGraphTensor* _Nonnull)inputTensor name:(NSString* _Nullable)name;

- (MPSGraphTensor* _Nonnull)resizeNearestWithTensor:(MPSGraphTensor* _Nonnull)imagesTensor
                                         sizeTensor:(MPSGraphTensor* _Nonnull)size
                                nearestRoundingMode:(MPSGraphResizeNearestRoundingMode)nearestRoundingMode
                                       centerResult:(BOOL)centerResult
                                       alignCorners:(BOOL)alignCorners
                                             layout:(MPSGraphTensorNamedDataLayout)layout
                                               name:(NSString* _Nullable)name;

- (MPSGraphTensor* _Nonnull)resizeNearestWithTensor:(MPSGraphTensor* _Nonnull)imagesTensor
                                         sizeTensor:(MPSGraphTensor* _Nonnull)size
                                  scaleOffsetTensor:(MPSGraphTensor* _Nonnull)scaleOffset
                                nearestRoundingMode:(MPSGraphResizeNearestRoundingMode)nearestRoundingMode
                                             layout:(MPSGraphTensorNamedDataLayout)layout
                                               name:(NSString* _Nullable)name;

- (MPSGraphTensor* _Nonnull)resizeBilinearWithTensor:(MPSGraphTensor* _Nonnull)imagesTensor
                                          sizeTensor:(MPSGraphTensor* _Nonnull)size
                                        centerResult:(BOOL)centerResult
                                        alignCorners:(BOOL)alignCorners
                                              layout:(MPSGraphTensorNamedDataLayout)layout
                                                name:(NSString* _Nullable)name;

- (MPSGraphTensor* _Nonnull)resizeBilinearWithTensor:(MPSGraphTensor* _Nonnull)imagesTensor
                                          sizeTensor:(MPSGraphTensor* _Nonnull)size
                                   scaleOffsetTensor:(MPSGraphTensor* _Nonnull)scaleOffset
                                              layout:(MPSGraphTensorNamedDataLayout)layout
                                                name:(NSString* _Nullable)name;

- (MPSGraphTensor* _Nonnull)resizeNearestWithGradientTensor:(MPSGraphTensor* _Nonnull)gradient
                                                      input:(MPSGraphTensor* _Nonnull)input
                                        nearestRoundingMode:(MPSGraphResizeNearestRoundingMode)nearestRoundingMode
                                               centerResult:(BOOL)centerResult
                                               alignCorners:(BOOL)alignCorners
                                                     layout:(MPSGraphTensorNamedDataLayout)layout
                                                       name:(NSString* _Nullable)name;

- (MPSGraphTensor* _Nonnull)resizeNearestWithGradientTensor:(MPSGraphTensor* _Nonnull)gradient
                                                      input:(MPSGraphTensor* _Nonnull)input
                                          scaleOffsetTensor:(MPSGraphTensor* _Nonnull)scaleOffset
                                        nearestRoundingMode:(MPSGraphResizeNearestRoundingMode)nearestRoundingMode
                                                     layout:(MPSGraphTensorNamedDataLayout)layout
                                                       name:(NSString* _Nullable)name;

- (MPSGraphTensor* _Nonnull)resizeBilinearWithGradientTensor:(MPSGraphTensor* _Nonnull)gradient
                                                       input:(MPSGraphTensor* _Nonnull)input
                                                centerResult:(BOOL)centerResult
                                                alignCorners:(BOOL)alignCorners
                                                      layout:(MPSGraphTensorNamedDataLayout)layout
                                                        name:(NSString* _Nullable)name;

- (MPSGraphTensor* _Nonnull)resizeBilinearWithGradientTensor:(MPSGraphTensor* _Nonnull)gradient
                                                       input:(MPSGraphTensor* _Nonnull)input
                                           scaleOffsetTensor:(MPSGraphTensor* _Nonnull)scaleOffset
                                                      layout:(MPSGraphTensorNamedDataLayout)layout
                                                        name:(NSString* _Nullable)name;

- (MPSGraphTensor* _Nonnull)sampleGridWithSourceTensor:(MPSGraphTensor* _Nonnull)source
                                      coordinateTensor:(MPSGraphTensor* _Nonnull)coordinates
                                                layout:(MPSGraphTensorNamedDataLayout)layout
                                  normalizeCoordinates:(BOOL)normalizeCoordinates
                                   relativeCoordinates:(BOOL)relativeCoordinates
                                          alignCorners:(BOOL)alignCorners
                                           paddingMode:(MPSGraphPaddingMode)paddingMode
                                          samplingMode:(MPSGraphResizeMode)samplingMode
                                         constantValue:(double)constantValue
                                                  name:(NSString* _Nullable)name;

- (MPSGraphTensor* _Nonnull)sampleGridWithSourceTensor:(MPSGraphTensor* _Nonnull)source
                                      coordinateTensor:(MPSGraphTensor* _Nonnull)coordinates
                                                layout:(MPSGraphTensorNamedDataLayout)layout
                                  normalizeCoordinates:(BOOL)normalizeCoordinates
                                   relativeCoordinates:(BOOL)relativeCoordinates
                                          alignCorners:(BOOL)alignCorners
                                           paddingMode:(MPSGraphPaddingMode)paddingMode
                                   nearestRoundingMode:(MPSGraphResizeNearestRoundingMode)nearestRoundingMode
                                         constantValue:(double)constantValue
                                                  name:(NSString* _Nullable)name;
- (MPSGraphTensor* _Nonnull)truncateWithTensor:(MPSGraphTensor* _Nonnull)tensor name:(NSString* _Nullable)name;

@end
