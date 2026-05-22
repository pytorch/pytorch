#include <c10/macros/Macros.h>

C10_DIAGNOSTIC_PUSH_AND_IGNORED_IF_DEFINED("-Wdeprecated-declarations")
#import <MetalPerformanceShaders/MetalPerformanceShaders.h>
C10_DIAGNOSTIC_POP()

#include <vector>

@interface MPSImage (Tensor)

- (std::vector<int64_t>)sizes;
- (int64_t)readCount;
- (BOOL)isTemporaryImage;
- (void)markRead;
- (void)recycle;

@end
