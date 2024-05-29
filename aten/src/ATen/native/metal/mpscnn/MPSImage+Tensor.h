#import <MetalPerformanceShaders/MetalPerformanceShaders.h>

#include <vector>

@interface MPSImage (Tensor)

- (std::vector<int64_t>)sizes;
- (int64_t)readCount;
- (BOOL)isTemporaryImage;
- (void)markRead;
- (void)recycle;

@end
