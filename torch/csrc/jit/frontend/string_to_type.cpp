#include <ATen/core/jit_type.h>

namespace torch {
namespace jit {
using namespace c10;
const std::unordered_map<std::string, TypePtr>& string_to_type_lut() {
  static std::unordered_map<std::string, TypePtr> map = {
      {"Tensor", TensorType::get()},
      {"int", IntType::get()},
      {"float", FloatType::get()},
      {"bool", BoolType::get()},
      {"str", StringType::get()},
      {"Device", DeviceObjType::get()},
      // technically this is not a python type but we need it when
      // parsing serialized methods that use implicit conversions to Scalar
      {"number", NumberType::get()},
      {"None", NoneType::get()},
      {"Any", AnyType::get()},
      {"Capsule", CapsuleType::get()},
      {"list", AnyListType::get()},
      {"tuple", AnyTupleType::get()}};
  return map;
}
} // namespace jit
} // namespace torch
