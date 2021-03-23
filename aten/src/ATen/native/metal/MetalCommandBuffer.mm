#import <ATen/native/metal/MetalCommandBuffer.h>
#import <ATen/native/metal/mpscnn/MPSCNNContext.h>
#import <ATen/native/metal/mpscnn/MPSImage+Tensor.h>

#include <mutex>

NSString* thread_local_storage_key = @"PTMetalCommandBuffer";
@implementation MetalCommandBuffer {
  NSMutableArray* _images;
  NSMutableArray<id<PTMetalCommandBufferDelegate>>* _delegates;
  std::mutex _mutex;
}

+ (MetalCommandBuffer*)newBuffer {
  MetalCommandBuffer* cb = [MetalCommandBuffer new];
  cb->_buffer = [[MPSCNNContext sharedInstance].commandQueue commandBuffer];
  cb->_thread = [NSThread currentThread];
  cb->_images = [NSMutableArray new];
  cb->_delegates = [NSMutableArray new];
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
    cb->_delegates = [NSMutableArray new];
    dict[thread_local_storage_key] = cb;
  }
  return cb;
}

- (void)addDelegate:(id<PTMetalCommandBufferDelegate>)delegate {
  if ([_delegates containsObject:delegate]) {
    [_delegates removeObject:delegate];
  }
  [_delegates addObject:delegate];
}

- (void)removeDelegate:(id<PTMetalCommandBufferDelegate>)delegate {
  [_delegates removeObject:delegate];
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
  [self prepare];
  [self flush];
  [self cleanup];
}

- (void)flush {
  if (_buffer.status == 0) {
    [_buffer commit];
    [_buffer waitUntilCompleted];
  }
}

- (void)prepare {
  for (id<PTMetalCommandBufferDelegate> delegate in _delegates) {
    if ([delegate respondsToSelector:@selector(prepareForSynchronization)]) {
      [delegate prepareForSynchronization];
    };
  }
  // recycle all temporary images manually before flushing the command buffer
  for (MPSTemporaryImage* image in _images) {
    [image recycle];
  }
}

- (void)cleanup {
  [_images removeAllObjects];
  [_delegates removeAllObjects];
  [[_thread threadDictionary] removeObjectForKey:thread_local_storage_key];
}

- (BOOL)isEqual:(id)object {
  if (![object isKindOfClass:[MetalCommandBuffer class]]) {
    return NO;
  }
  MetalCommandBuffer* mc = (MetalCommandBuffer*)object;
  return (_thread == mc.thread && _buffer == mc.buffer);
}

@end
