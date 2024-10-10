#import <ATen/native/metal/mpscnn/MPSImage+Tensor.h>

@implementation MPSImage (Tensor)

- (std::vector<int64_t>)sizes {
  int64_t N = self.numberOfImages;
  int64_t C = self.featureChannels;
  int64_t H = self.height;
  int64_t W = self.width;
  return {N, C, H, W};
}

- (BOOL)isTemporaryImage {
  return [self isKindOfClass:[MPSTemporaryImage class]];
}

- (void)markRead {
  if ([self isTemporaryImage]) {
    MPSTemporaryImage* tmpImage = (MPSTemporaryImage*)self;
    if (tmpImage.readCount > 0) {
      tmpImage.readCount -= 1;
    }
  }
}

- (void)recycle {
  if ([self isTemporaryImage]) {
    MPSTemporaryImage* tmpImage = (MPSTemporaryImage*)self;
    if (tmpImage.readCount > 0) {
      tmpImage.readCount = 0;
    }
  }
}

- (int64_t)readCount {
  if ([self isTemporaryImage]) {
    MPSTemporaryImage* tmpImage = (MPSTemporaryImage*)self;
    return (int64_t)tmpImage.readCount;
  }
  return -1;
}

@end
