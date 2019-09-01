#include <vector>
#import "TorchIValue.h"
#import "TorchIValue+Internal.h"
#import "TorchTensor.h"
#import "TorchTensor+Internal.h"
#import <Pytorch/Pytorch.h>

#define IVALUE_TYPE(_) \
    _(None) \
    _(Bool) \
    _(Int) \
    _(Double) \
    _(Tensor) \
    _(BoolList) \
    _(IntList) \
    _(DoubleList)\
    _(TensorList)

#define IVALUE_SCALAR_TYPE(_) \
    _(Bool, bool, bool) \
    _(Int, int, int64_t) \
    _(Double, double,double) \

@implementation TorchIValue {
    std::shared_ptr<at::IValue> _impl;
}

+ (instancetype)newWithNone{
    auto impl = std::make_shared<at::IValue>();
    if(!impl) {
        return nil;
    }
    TorchIValue* value = [TorchIValue new];
    value->_type = TorchIValueTypeNone;
    value->_impl = std::move(impl);
    return value;
}

#define DEFINE_IVALUE_WITH_SCALAR_TYPE(type) \
+ (instancetype) newWith##type:(NSNumber* )value{\
if(![value isKindOfClass:[NSNumber class]]){ return nil; }\
return [self newWithType:TorchIValueType##type Data:value]; \
}

DEFINE_IVALUE_WITH_SCALAR_TYPE(Bool)
DEFINE_IVALUE_WITH_SCALAR_TYPE(Int)
DEFINE_IVALUE_WITH_SCALAR_TYPE(Double)

#define DEFINE_IVALUE_WITH_SCALAR_TYPE_LIST(type) \
+ (instancetype) newWith##type##List:(NSArray<NSNumber*>* )list{\
if(![list isKindOfClass:[NSArray class]]){ return nil; }\
return [self newWithType:TorchIValueType##type##List Data:list]; \
}

DEFINE_IVALUE_WITH_SCALAR_TYPE_LIST(Bool)
DEFINE_IVALUE_WITH_SCALAR_TYPE_LIST(Int)
DEFINE_IVALUE_WITH_SCALAR_TYPE_LIST(Double)

+ (instancetype) newWithTensor:(TorchTensor* )tensor {
    if(![tensor isKindOfClass:[TorchTensor class]]){
        return nil;
    }
    auto t = tensor.toTensor;
    at::IValue atIValue(t);
    auto impl = std::make_shared<at::IValue>(atIValue);
    if(!impl) {
        return nil;
    }
    TorchIValue* value = [TorchIValue new];
    value->_type = TorchIValueTypeTensor;
    value->_impl = std::move(impl);
    return value;
}

+ (instancetype) newWithTensorList:(NSArray<TorchTensor*>* )list {
    if(![list isKindOfClass:[NSArray<TorchTensor* > class]]){
        return nil;
    }
    c10::List<at::Tensor> tensorList;
    for(TorchTensor* tensor in list){
        auto t = tensor.toTensor;
        tensorList.push_back(t);
    }
    at::IValue atIValue(tensorList);
    auto impl = std::make_shared<at::IValue>(atIValue);
    if(!impl) {
        return nil;
    }
    TorchIValue* value = [TorchIValue new];
    value->_type = TorchIValueTypeTensorList;
    value->_impl = std::move(impl);
    return value;
}

+ (instancetype) newWithType:(TorchIValueType)type Data:(id _Nullable)data {
    TorchIValue* value = [TorchIValue new];
    value->_type = type;
    at::IValue atIValue = {};
    switch (type) {
    #define  DEFINE_CASE(x,y,z) case TorchIValueType##x: {atIValue = at::IValue([(NSNumber* )data y##Value]);break;}
        IVALUE_SCALAR_TYPE(DEFINE_CASE)
    #undef DEFINE_CASE

    #define  DEFINE_CASE(x,y,z) case TorchIValueType##x##List: {\
    c10::List<z> list; \
    for(NSNumber* number in data){ list.push_back(number.y##Value); }\
    atIValue = list; break; }
        IVALUE_SCALAR_TYPE(DEFINE_CASE)
    #undef DEFINE_CASE
        default:
            break;
    }
    auto impl = std::make_shared<at::IValue>(atIValue);
    value->_impl = std::move(impl);
    return value->_impl ? value : nil;
}

#define DEFINE_TO_SCALAR_TYPE(Type) \
- (NSNumber* )to##Type {\
if(!_impl || !_impl->is##Type()) { return nil; }\
return @(_impl->to##Type()); \
}

DEFINE_TO_SCALAR_TYPE(Bool);
DEFINE_TO_SCALAR_TYPE(Int);
DEFINE_TO_SCALAR_TYPE(Double);

#define DEFINE_TO_SCALAR_TYPE_LIST(Type) \
- (NSArray<NSNumber* >* )to##Type##List {\
if(!_impl || !_impl->is##Type##List()) { return nil; }\
auto list = _impl->to##Type##List(); \
NSMutableArray<NSNumber* >* tmp = [NSMutableArray new]; \
for(int i=0; i<list.size(); ++i) { [tmp addObject:@(list.get(i))]; } \
return [tmp copy];\
}

DEFINE_TO_SCALAR_TYPE_LIST(Bool);
DEFINE_TO_SCALAR_TYPE_LIST(Int);
DEFINE_TO_SCALAR_TYPE_LIST(Double);

- (TorchTensor* )toTensor {
   if (!_impl || !_impl->isTensor()) {
       return nil;
   }
   at::Tensor tensor = _impl->toTensor();
   return [TorchTensor newWithTensor:tensor];
}

- (NSArray<TorchTensor *> *)toTensorList{
    if (!_impl || !_impl->isTensorList()) {
        return nil;
    }
    auto list = _impl->toTensorList();
    NSMutableArray* ret = [NSMutableArray new];
    for(int i=0; i<list.size(); ++i){
        TorchTensor* tensor = [TorchTensor newWithTensor:list.get(i)];
        [ret addObject:tensor];
    }
    return [ret copy];
}

@end

@implementation TorchIValue (Internal)

- (at::IValue )toIValue {
    if(_impl){
        return at::IValue(*_impl);
    }
    return {};
}

+ (TorchIValue* )newWithIValue:(const at::IValue& )v {
    TorchIValue* value = [TorchIValue new];
    
    #define DEFINE_IF(x)\
        if(v.is##x()) { value->_type = TorchIValueType##x; }
        IVALUE_TYPE(DEFINE_IF)
    #undef DEFINE_IF

    auto impl = std::make_shared<at::IValue>(v);
    if(!impl){
        return nil;
    }
    
    value->_impl = std::move(impl);
    return value;
}

@end

