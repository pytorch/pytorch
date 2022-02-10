#import <ATen/native/metal/MetalCommandBuffer.h>
#import <ATen/native/metal/MetalContext.h>
#import <ATen/native/metal/mpscnn/MPSImage+Tensor.h>

NSString* thread_local_storage_key = @"PTMetalCommandBuffer";
@implementation MetalCommandBuffer {
  NSMutableArray* _images;
  NSMutableSet<id<PTMetalCommandBuffer>>* _delegates;
}

+ (MetalCommandBuffer*)newBuffer {
  MetalCommandBuffer* cb = [MetalCommandBuffer new];
  cb->_buffer = [[MetalContext sharedInstance].commandQueue commandBuffer];
  cb->_images = [NSMutableArray new];
  cb->_delegates = [NSMutableSet new];
  return cb;
}

+ (MetalCommandBuffer*)currentBuffer {
  NSThread* thd = [NSThread currentThread];
  thd.name = thread_local_storage_key;
  NSMutableDictionary* dict = [thd threadDictionary];
  MetalCommandBuffer* cb = dict[thread_local_storage_key];
  if (!cb || !cb.valid) {
    cb = [MetalCommandBuffer newBuffer];
    // The command buffer should only be retained by the thread-local storage.
    dict[thread_local_storage_key] = cb;
  }
  return cb;
}

- (BOOL)valid {
  return _buffer != nil && _buffer.status == 0;
}

- (void)addSubscriber:(id<PTMetalCommandBuffer>)subscriber {
  if (subscriber) {
    [_delegates addObject:subscriber];
  }
}
- (void)removeSubscriber:(id<PTMetalCommandBuffer>)subscriber {
  if (subscriber) {
    [_delegates removeObject:subscriber];
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

- (void)commit {
  if (_buffer.status == 0) {
    [self beginSynchronization];
    [_buffer commit];
    [_buffer waitUntilCompleted];
    [self endSynchronization];
  }
}

- (void)beginSynchronization {
  for (id<PTMetalCommandBuffer> delegate in _delegates) {
    if ([delegate respondsToSelector:@selector(beginSynchronization)]) {
      [delegate beginSynchronization];
    };
  }
  // recycle all temporary images manually before flushing the command buffer
  for (MPSTemporaryImage* image in _images) {
    [image recycle];
  }
}

- (void)endSynchronization {
  for (id<PTMetalCommandBuffer> delegate in _delegates) {
    if ([delegate respondsToSelector:@selector(endSynchronization:)]) {
      [delegate endSynchronization:_buffer.error];
    };
  }
  [_delegates removeAllObjects];
  [_images removeAllObjects];
  _buffer = nil;
  [[NSThread currentThread].threadDictionary
      removeObjectForKey:thread_local_storage_key];
}

- (BOOL)isEqual:(id)object {
  if (![object isKindOfClass:[MetalCommandBuffer class]]) {
    return NO;
  }
  MetalCommandBuffer* mc = (MetalCommandBuffer*)object;
  return _buffer == mc.buffer;
}

@end
