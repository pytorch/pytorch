#pragma once

#import <MetalPerformanceShadersGraph/MPSGraph.h>

NS_ASSUME_NONNULL_BEGIN

// ScatterGather.mm and TriangularOps.mm call to those
// But they can not be found in https://developer.apple.com/documentation/metalperformanceshadersgraph/mpsgraph?language=objc
// Nor in MacOSX12.3.sdk headers
// TODO: Do not call to undocumented methods

@interface MPSGraph(CoolMethods)
-(MPSGraphTensor *) gatherAlongAxisWithUpdatesTensor:(MPSGraphTensor *) updateTensor
                                       indicesTensor:(MPSGraphTensor *) indicesTensor
                                                axis:(NSInteger)dim
                                                name:(NSString * _Nullable)name;

-(MPSGraphTensor *) getCoordinateValueWithShapeTensor:(MPSGraphTensor *) shapeTensor
                                           axisTensor:(MPSGraphTensor *) axisTensor
                                                 name:(NSString * _Nullable) name;

-(MPSGraphTensor *) scatterAlongAxisWithDataTensor:(MPSGraphTensor *) dataTensor
                                     updatesTensor:(MPSGraphTensor *) updatesTensor
                                     indicesTensor:(MPSGraphTensor *) indicesTensor
                                              axis:(NSInteger) axis
                                              mode:(MPSGraphScatterMode) mode
                                              name:(NSString * _Nullable)name;
@end

NS_ASSUME_NONNULL_END
