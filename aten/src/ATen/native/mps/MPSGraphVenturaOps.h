#pragma once
#include <MetalPerformanceShadersGraph/MetalPerformanceShadersGraph.h>

// TODO: Remove me when moved to MacOS 13
@interface MPSGraph (VenturaOps)
- (MPSGraphTensor *)cumulativeSumWithTensor:(MPSGraphTensor *)tensor
                                       axis:(NSInteger)axis
                                       name:(NSString *)name;
@end
