#import <ATen/native/metal/MetalCommandBuffer.h>
#import <ATen/native/metal/mpscnn/MPSCNNContext.h>
#import <ATen/native/metal/mpscnn/MPSImage+Tensor.h>

#include <mutex>

NSString* thread_local_storage_key = @"PTMetalCommandBuffer";
@implementation MetalCommandBuffer {
  NSMutableArray* _images;
  NSMutableSet<id<PTMetalCommandBufferDelegate>>* _delegates;
}

+ (MetalCommandBuffer*)newBuffer {
  MetalCommandBuffer* cb = [MetalCommandBuffer new];
  cb->_buffer = [[MPSCNNContext sharedInstance].commandQueue commandBuffer];
  cb->_images = [NSMutableArray new];
  cb->_delegates = [NSMutableSet new];
  [NSThread currentThread].threadDictionary[thread_local_storage_key] = cb;
  return cb;
}

+ (MetalCommandBuffer*)currentBuffer {
  NSThread* thd = [NSThread currentThread];
  thd.name = thread_local_storage_key;
  NSMutableDictionary* dict = [thd threadDictionary];
  MetalCommandBuffer* cb = dict[thread_local_storage_key];
  if (!cb) {
    cb = [MetalCommandBuffer newBuffer];
    dict[thread_local_storage_key] = cb;
  }
  return cb;
}

- (void)addDelegate:(id<PTMetalCommandBufferDelegate>)delegate {
  if (delegate) {
    [_delegates addObject:delegate];
  }
}

- (void)removeDelegate:(id<PTMetalCommandBufferDelegate>)delegate {
  if (delegate) {
    [_delegates removeObject:delegate];
  }
}

- (void)add:(MPSTemporaryImage*)image {
  if (![image isTemporaryImage]) {
    return;
  }
  [_images addObject:image];
}

- (void)remove:(MPSTemporaryImage*)image {
  if (![image isTemporaryImage]) {
    return;
  }
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
  _buffer = nil;
  [[NSThread currentThread].threadDictionary removeObjectForKey:thread_local_storage_key];
}

- (BOOL)isEqual:(id)object {
  if (![object isKindOfClass:[MetalCommandBuffer class]]) {
    return NO;
  }
  MetalCommandBuffer* mc = (MetalCommandBuffer*)object;
  return _buffer == mc.buffer;
}

@end
