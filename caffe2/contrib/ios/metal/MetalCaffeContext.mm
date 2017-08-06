// Copyright 2004-present Facebook. All Rights Reserved.

#import "MetalCaffeContext.h"
#import "MetalContext.h"

namespace caffe2 {
CAFFE_KNOWN_TYPE(TensorMetal);

static NSMutableDictionary<NSNumber *, id<MTLBuffer>> *buffer_cache = nil;

// class MetalAllocator
MetalAllocator::MetalAllocator(id<MTLDevice> _device) : device(_device) {
  buffer_cache = [NSMutableDictionary<NSNumber *, id<MTLBuffer>> dictionary];
}

MetalAllocator::~MetalAllocator() {
  for (id key in buffer_cache) {
    id<MTLBuffer> buffer = buffer_cache[key];
    [buffer_cache removeObjectForKey:key];
  }
}

void *MetalAllocator::New(size_t nbytes) {
  id<MTLBuffer> buffer = [device newBufferWithLength:nbytes options:MTLResourceCPUCacheModeDefaultCache];
  void *data = [buffer contents];
  NSNumber *key = @((unsigned long long)data);
  buffer_cache[key] = buffer;
  return data;
}

void MetalAllocator::Delete(void *data) {
  NSNumber *key = @((unsigned long long)data);
  id<MTLBuffer> buffer = buffer_cache[key];
  [buffer_cache removeObjectForKey:key];
  buffer = nil;
}

id<MTLBuffer> MetalAllocator::Buffer(void *data) {
  NSNumber *key = @((unsigned long long)data);
  return buffer_cache[key];
}

// the Metal Allocator
static MetalAllocator *MetalAllocatorInstance = NULL;

// Get the Metal Allocator
MetalAllocator *GetMetalAllocator() {
  if (MetalAllocatorInstance == NULL) {
    MetalAllocatorInstance = new MetalAllocator([MetalContext getContext].device);
  }
  CAFFE_ENFORCE(MetalAllocatorInstance != NULL);
  return MetalAllocatorInstance;
}

// class MetalCaffeContext
void *MetalCaffeContext::New(size_t nbytes) { return MetalAllocatorInstance->New(nbytes); }

void MetalCaffeContext::Delete(void *data) { MetalAllocatorInstance->Delete(data); }
}
