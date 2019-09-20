#import "TorchTensor.h"
#import <LibTorch/LibTorch.h>
#import "TorchTensorPrivate.h"

#define DEFINE_TENSOR_TYPES(_) \
  _(Byte)                      \
  _(Char)                      \
  _(Int)                       \
  _(Float)                     \
  _(Long)                      \
  _(Undefined)

static inline c10::ScalarType scalarTypeFromTensorType(TorchTensorType type) {
  switch (type) {
#define DEFINE_CASE(x)     \
  case TorchTensorType##x: \
    return c10::ScalarType::x;
    DEFINE_TENSOR_TYPES(DEFINE_CASE)
#undef DEFINE_CASE
  }
  return c10::ScalarType::Undefined;
}

static inline TorchTensorType tensorTypeFromScalarType(c10::ScalarType type) {
  switch (type) {
#define DEFINE_CASE(x)     \
  case c10::ScalarType::x: \
    return TorchTensorType##x;
    DEFINE_TENSOR_TYPES(DEFINE_CASE)
#undef DEFINE_CASE
    default:
      return TorchTensorTypeUndefined;
  }
}

@implementation TorchTensor {
  at::Tensor _impl;
}

- (TorchTensorType)dtype {
  return tensorTypeFromScalarType(_impl.scalar_type());
}

- (NSArray<NSNumber*>*)sizes {
  NSMutableArray* shapes = [NSMutableArray new];
  auto dims = _impl.sizes();
  for (int i = 0; i < dims.size(); ++i) {
    [shapes addObject:@(dims[i])];
  }
  return [shapes copy];
}

- (int64_t)numel {
  return _impl.numel();
}

- (void*)data {
  return _impl.unsafeGetTensorImpl()->storage().data();
}

- (int64_t)dim {
  return _impl.dim();
}

+ (nullable TorchTensor*)newWithData:(void*)data
                                Size:(NSArray<NSNumber*>*)size
                                Type:(TorchTensorType)type {
  if (!data) {
    return nil;
  }
  std::vector<int64_t> dimsVec;
  for (auto i = 0; i < size.count; ++i) {
    int64_t dim = size[i].integerValue;
    dimsVec.push_back(dim);
  }
  @try {
    at::Tensor tensor = torch::from_blob((void*)data, dimsVec, scalarTypeFromTensorType(type));
    return [TorchTensor newWithTensor:tensor];
  } @catch (NSException* exception) {
    @throw exception;
    NSLog(@"%@", exception);
  }
  return nil;
}

- (NSString*)description {
  NSString* size = @"[";
  for (NSNumber* num in self.sizes) {
    size = [size stringByAppendingString:[NSString stringWithFormat:@"%ld ", num.integerValue]];
  }
  size = [size stringByAppendingString:@"]"];
  return [NSString stringWithFormat:@"[%s %@]", _impl.toString().c_str(), size];
}

- (TorchTensor*)objectAtIndexedSubscript:(NSUInteger)idx {
  auto tensor = _impl[idx];
  return [TorchTensor newWithTensor:tensor];
}

- (void)setObject:(id)obj atIndexedSubscript:(NSUInteger)idx {
  NSAssert(NO, @"Tensors are immutable");
}

#pragma mark NSCopying

- (instancetype)copyWithZone:(NSZone*)zone {
  // tensors are immutable
  return self;
}

- (at::Tensor)toTensor {
  return at::Tensor(_impl);
}

+ (TorchTensor*)newWithTensor:(const at::Tensor&)tensor {
  TorchTensor* torchTensor = [TorchTensor new];
  torchTensor->_impl = at::Tensor(tensor);
  return torchTensor;
}

@end
