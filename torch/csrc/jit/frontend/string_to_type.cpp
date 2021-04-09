#include <ATen/core/jit_type.h>

namespace torch {
namespace jit {
using namespace c10;
const std::unordered_map<std::string, TypePtr>& string_to_type_lut() {
  static std::unordered_map<std::string, TypePtr> map = {
      {"Tensor", TensorType::get()},
      // Dtype constraints are not constrained in compilation. Therefore, we map
      // all tensor subclasses with different dtypes to a same underlying
      // Tensor. But, we give warning about possible dtype change whenever user
      // uses any of the tensor subclasses such as LongTensor.
      {"LongTensor", TensorType::get()},
      {"DoubleTensor", TensorType::get()},
      {"FloatTensor", TensorType::get()},
      {"IntTensor", TensorType::get()},
      {"ShortTensor", TensorType::get()},
      {"HalfTensor", TensorType::get()},
      {"CharTensor", TensorType::get()},
      {"ByteTensor", TensorType::get()},
      {"BoolTensor", TensorType::get()},
      {"int", IntType::get()},
      {"float", FloatType::get()},
      {"bool", BoolType::get()},
      {"complex", ComplexType::get()},
      {"str", StringType::get()},
      {"Device", DeviceObjType::get()},
      {"Stream", StreamObjType::get()},
      // technically this is not a python type but we need it when
      // parsing serialized methods that use implicit conversions to Scalar
      {"number", NumberType::get()},
      {"None", NoneType::get()},
      {"NoneType", NoneType::get()},
      {"Any", AnyType::get()},
      {"Capsule", CapsuleType::get()},
      {"list", AnyListType::get()},
      {"tuple", AnyTupleType::get()}};
  return map;
}
} // namespace jit
} // namespace torch
