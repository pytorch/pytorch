#import <ATen/native/metal/MetalCommandBuffer.h>
#import <ATen/native/metal/mpscnn/MPSCNNContext.h>
#import <ATen/native/metal/mpscnn/MPSImage+Tensor.h>

#include <mutex>

NSString* thread_local_storage_key = @"PTMetalCommandBuffer";
@implementation MetalCommandBuffer {
  NSMutableArray* _images;
  std::mutex _mutex;
}

+ (MetalCommandBuffer*)newBuffer {
  MetalCommandBuffer* cb = [MetalCommandBuffer new];
  cb->_buffer = [[MPSCNNContext sharedInstance].commandQueue commandBuffer];
  cb->_thread = [NSThread currentThread];
  cb->_images = [NSMutableArray new];
  return cb;
}

+ (MetalCommandBuffer*)currentBuffer {
  NSThread* thd = [NSThread currentThread];
  thd.name = thread_local_storage_key;
  NSMutableDictionary* dict = [thd threadDictionary];
  MetalCommandBuffer* cb = dict[thread_local_storage_key];
  if (!cb) {
    cb = [MetalCommandBuffer new];
    cb->_buffer = [[MPSCNNContext sharedInstance].commandQueue commandBuffer];
    cb->_thread = thd;
    cb->_images = [NSMutableArray new];
    dict[thread_local_storage_key] = cb;
  }
  return cb;
}

- (void)flush {
  [[_thread threadDictionary] removeObjectForKey:thread_local_storage_key];
}

- (void)add:(MPSTemporaryImage*)image {
  if (![image isTemporaryImage]) {
    return;
  }
  std::lock_guard<std::mutex> g(_mutex);
  [_images addObject:image];
}

- (void)remove:(MPSTemporaryImage*)image {
  if (![image isTemporaryImage]) {
    return;
  }
  std::lock_guard<std::mutex> g(_mutex);
  [_images removeObject:image];
}

- (void)synchronize {
  if (_buffer.status == 0) {
    // recycle all temporary images manually before flushing the command buffer
    [self recycle];
    [_buffer commit];
    [_buffer waitUntilCompleted];
    [[_thread threadDictionary] removeObjectForKey:thread_local_storage_key];
  }
}

- (void)recycle {
  for (MPSTemporaryImage* image in _images) {
    [image recycle];
  }
}

- (BOOL)isEqual:(id)object {
  if (![object isKindOfClass:[MetalCommandBuffer class]]) {
    return NO;
  }
  MetalCommandBuffer* mc = (MetalCommandBuffer*)object;
  return (_thread == mc.thread && _buffer == mc.buffer);
}

@end
