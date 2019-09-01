#import "TorchTensor.h"
#import "TorchTensor+Internal.h"
#import <Pytorch/Pytorch.h>

#define CHECK_IMPL(x) NSCAssert(x!=nil,@"impl is nil!");
#define CHECK_IMPL_(x) \
    CHECK_IMPL(x) \
    if (!x) { return nil; }

#define DEFINE_TENSOR_TYPES(_) \
    _(Byte) \
    _(Int) \
    _(Float) \
    _(Long) \
    _(Undefined)

#define DEFINE_TENSOR_SCALAR_TYPES(_) \
    _(Int) \
    _(Float) \
    _(Long) \

static inline c10::ScalarType scalarTypeFromTensorType(TorchTensorType type) {
    switch(type){
#define DEFINE_CASE(x) case TorchTensorType##x: return c10::ScalarType::x;
        DEFINE_TENSOR_TYPES(DEFINE_CASE)
#undef DEFINE_CASE
    }
    return c10::ScalarType::Undefined;
}

static inline TorchTensorType tensorTypeFromScalarType(c10::ScalarType type) {
    switch(type){
#define DEFINE_CASE(x) case c10::ScalarType::x: return TorchTensorType##x;
        DEFINE_TENSOR_TYPES(DEFINE_CASE)
#undef DEFINE_CASE
        default: return TorchTensorTypeUndefined;
    }
}

@implementation TorchTensor {
    std::shared_ptr<at::Tensor> _impl;
}

- (TorchTensorType)type {
    CHECK_IMPL(_impl)
    if(!_impl){
        return TorchTensorTypeUndefined;
    }
    return tensorTypeFromScalarType(_impl->scalar_type());
}

- (BOOL)quantized{
    return _impl ? _impl->is_quantized() : NO;
}

- (int64_t)numel{
    CHECK_IMPL(_impl);
    if(!_impl) {
        return NSNotFound;
    }
    return _impl->numel();
}

- (void* )data {
    CHECK_IMPL_(_impl);
    return _impl->unsafeGetTensorImpl()->storage().data();
}

- (int64_t)dim {
    CHECK_IMPL(_impl);
    if(!_impl) {
        return NSNotFound;
    }
    return _impl->dim();
}

+ (TorchTensor* )newWithType:(TorchTensorType)type Size:(NSArray<NSNumber* >*)size Data:(void* _Nullable)data {
    return [self newWithType:type Size:size Data:data Quantized:NO];
}


+ (TorchTensor* )newWithType:(TorchTensorType)type Size:(NSArray<NSNumber* >*)size Data:(void* _Nullable)data Quantized:(BOOL) quantized {
    if (!data || size.count == 0){
        return nil;
    }
    std::vector<int64_t> dimsVec;
    for (auto i = 0; i < size.count; ++i) {
        int64_t dim = size[i].integerValue;
        dimsVec.push_back(dim);
    }
    at::Tensor tensor = torch::from_blob( (void* )data, dimsVec,scalarTypeFromTensorType(type));
    if (quantized) {
        tensor = at::quantize_linear(tensor, 1, 0, at::kQInt8);
    }
    return [TorchTensor newWithTensor:tensor];
}

- (NSString* )description {
    CHECK_IMPL_(_impl);
    NSString* size = @"[";
    for(NSNumber* num in self.size) {
        size = [size stringByAppendingString:[NSString stringWithFormat:@"%ld", num.integerValue]];
    }
    size = [size stringByAppendingString:@"]"];
    return [NSString stringWithFormat:@"[%s %@]",_impl -> toString().c_str(),size];
}

- (TorchTensor* )objectAtIndexedSubscript:(NSUInteger)idx {
    CHECK_IMPL_(_impl)
    auto tensor = (*_impl)[idx];
    return [TorchTensor newWithTensor:tensor];
}

- (void)setObject:(id)obj atIndexedSubscript:(NSUInteger)idx {
    NSAssert(NO, @"Tensors are immutable");
}

- (instancetype)copyWithZone:(NSZone *)zone {
    //tensors are immutable
    return self;
}

@end

@implementation TorchTensor (Internal)

- (at::Tensor)toTensor {
    CHECK_IMPL(_impl);
    return at::Tensor(*_impl);
}

+ (TorchTensor* )newWithTensor:(const at::Tensor& ) tensor{
    std::shared_ptr<at::Tensor> impl = std::make_shared<at::Tensor>(tensor);
    if(!impl) {
        return nil;
    }
    TorchTensor* t = [TorchTensor new];
    NSMutableArray* shapes = [NSMutableArray new];
    auto dims = tensor.sizes();
    for (int i=0; i<dims.size(); ++i){
        [shapes addObject:@(dims[i])];
    }
    t->_size = [shapes copy];
    t->_impl = std::move(impl);
    
    return t;
}

@end

@implementation TorchTensor (Operations)

- (NSNumber* )item {
    CHECK_IMPL_(_impl)
    if( self.numel > 1 ){
        return nil;
    }
    switch (self.type) {
#define DEFINE_CASE(x) case TorchTensorType##x: return @(_impl->item().to##x());
            DEFINE_TENSOR_SCALAR_TYPES(DEFINE_CASE)
#undef DEFINE_CASE
        default:
            return nil;
    }
}

- (TorchTensor* )to:(TorchTensorType) type {
    CHECK_IMPL_(_impl)
    c10::ScalarType scalarType = scalarTypeFromTensorType(type);
    auto tensor = _impl->to(scalarType);
    return [TorchTensor newWithTensor:tensor];
}

- (TorchTensor* )permute:(NSArray<NSNumber* >*) dims {
    CHECK_IMPL_(_impl)
    std::vector<int64_t> dimsVec;
    for (auto i = 0; i < dims.count; ++i) {
        int64_t dim = dims[i].integerValue;
        dimsVec.push_back(dim);
    }
    auto newTensor =  _impl->permute(dimsVec);
    newTensor.options();
    return [TorchTensor newWithTensor:newTensor];
}

- (TorchTensor* )view:(NSArray<NSNumber* >*)size {
    CHECK_IMPL_(_impl)
    std::vector<int64_t> views;
    for(NSNumber* n in size){
        views.push_back(n.unsignedIntegerValue);
    }
    auto newTensor = _impl->view(views);
    return [TorchTensor newWithTensor:newTensor];
}

@end
