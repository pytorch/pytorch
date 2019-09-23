#import <LibTorch/LibTorch.h>
#import "TorchIValue.h"
#import "TorchIValuePrivate.h"
#import "TorchTensor.h"
#import "TorchTensorPrivate.h"

#define DEFINE_IVALUE_SCALAR_TYPE_VALUE(_) \
  _(Bool, bool, bool)                      \
  _(Int, int, int64_t)                     \
  _(Double, double, double)                \

@implementation TorchIValue {
  at::IValue _impl;
}

#define NEW_VALUE(type1, type2, type3)                  \
  +(instancetype)newWith##type1##Value : (type3)value { \
    TorchIValue* ret = [TorchIValue new];               \
    ret->_impl = at::IValue(value);                     \
    return ret;                                         \
  }

DEFINE_IVALUE_SCALAR_TYPE_VALUE(NEW_VALUE)

#define NEW_LIST(type1, type2, type3)                               \
  +(instancetype)newWith##type1##List : (NSArray<NSNumber*>*)list { \
    TorchIValue* value = [TorchIValue new];                         \
    c10::List<type3> type2##Array;                                  \
    for (NSNumber * number in list) {                               \
      type2##Array.push_back(number.type2##Value);                  \
    }                                                               \
    value->_impl = at::IValue(type2##Array);                        \
    return value;                                                   \
  }

DEFINE_IVALUE_SCALAR_TYPE_VALUE(NEW_LIST)

+ (instancetype)newWithStringValue:(NSString* )value {
    TorchIValue* ret = [TorchIValue new];
    ret->_impl = at::IValue(std::string(value.UTF8String));
    return ret;
}

+ (instancetype)newWithStringList:(NSArray<NSString* >* )list {
    TorchIValue* ret = [TorchIValue new];
    c10::List<std::string> strArray;
    for(NSString* str in list) {
        strArray.push_back(std::string(str.UTF8String));
    }
    ret->_impl = strArray;
    return ret;
}

+ (instancetype)newWithTensor:(TorchTensor*)tensor {
  TorchIValue* value = [TorchIValue new];
  value->_impl = at::IValue(tensor.toTensor);
  return value;
}

+ (instancetype)newWithTensorList:(NSArray<TorchTensor*>*)list {
  c10::List<at::Tensor> tensorList;
  for (TorchTensor* tensor in list) {
    auto atTensor = tensor.toTensor;
    tensorList.push_back(atTensor);
  }
  TorchIValue* value = [TorchIValue new];
  value->_impl = at::IValue(tensorList);
  return value;
}

#define DEFINE_IS_SCALAR_TYPE(type) \
  -(BOOL)is##type {                 \
    return _impl.is##type();        \
  }
DEFINE_IS_SCALAR_TYPE(Tensor)
DEFINE_IS_SCALAR_TYPE(Bool)
DEFINE_IS_SCALAR_TYPE(Double)
DEFINE_IS_SCALAR_TYPE(Int)
DEFINE_IS_SCALAR_TYPE(String)
DEFINE_IS_SCALAR_TYPE(TensorList)
DEFINE_IS_SCALAR_TYPE(BoolList)
DEFINE_IS_SCALAR_TYPE(DoubleList)
DEFINE_IS_SCALAR_TYPE(IntList)

#define TO_VALUE(type1, type2, type3)                   \
  -(type3)to##type1 {                                   \
    NSAssert(_impl.is##type1(), @"Type doesn't match"); \
    return _impl.to##type1();                           \
  }
DEFINE_IVALUE_SCALAR_TYPE_VALUE(TO_VALUE)

#define TO_LIST(type1, type2, type3)                       \
  -(NSArray<NSNumber*>*)to##type1##List {                  \
    if (!_impl.is##type1##List()) {                        \
      return nil;                                          \
    }                                                      \
    auto list = _impl.to##type1##List();                   \
    NSMutableArray<NSNumber*>* tmp = [NSMutableArray new]; \
    for (int i = 0; i < list.size(); ++i) {                \
      [tmp addObject:@(list.get(i))];                      \
    }                                                      \
    return [tmp copy];                                     \
  }
DEFINE_IVALUE_SCALAR_TYPE_VALUE(TO_LIST)

- (TorchTensor*)toTensor {
  if (!_impl.isTensor()) {
    return nil;
  }
  at::Tensor tensor = _impl.toTensor();
  return [TorchTensor newWithTensor:tensor];
}

- (NSArray<TorchTensor*>*)toTensorList {
  if (!_impl.isTensorList()) {
    return nil;
  }
  auto list = _impl.toTensorList();
  NSMutableArray* ret = [[NSMutableArray alloc] init];
  for (int i = 0; i < list.size(); ++i) {
    TorchTensor* tensor = [TorchTensor newWithTensor:list.get(i)];
    [ret addObject:tensor];
  }
  return [ret copy];
}

- (NSString* )toString {
    if(!_impl.isString()){
        return nil;
    }
    auto str = (*_impl.toString()).string();
    return [[NSString alloc]initWithCString:str.c_str() encoding:NSUTF8StringEncoding];
}

- (at::IValue)toIValue {
  return at::IValue(_impl);
}

+ (TorchIValue*)newWithIValue:(const at::IValue&)value {
    TorchIValue* torchIValue = [TorchIValue new];
    torchIValue->_impl = at::IValue(value);
    return torchIValue;
}

@end
