#pragma once

#include <ATen/core/jit_type.h>

namespace torch {
namespace jit {
const std::unordered_map<std::string, c10::TypePtr>& string_to_type_lut();
} // namespace jit
} // namespace torch

namespace c10 {

template <>
struct TORCH_API TypeParser::TypeFactory<c10::DynamicType> {
  template <typename T, typename... Args>
  static TypePtr create(Args&&... args) {
    return std::make_shared<DynamicType>(
        DynamicTypeTrait<T>::tagValue,
        DynamicType::Arguments(
            ArrayRef<TypePtr>({std::forward<Args>(args)...})));
  }
};

template <>
struct TORCH_API TypeParser::TypeFactory<c10::Type> {
  template <typename T, typename... Args>
  static TypePtr create(Args&&... args) {
    return T::create(std::forward<Args>(args)...);
  }
};

template <typename T>
TypePtr TypeParser::parseNonSimple(const std::string& token) {
  if (token == "List") {
    return CreateSingleElementType<ListType>();
  } else if (token == "Optional") {
    return parseSingleElementType(DynamicType::Tag::Optional);
  } else if (token == "Future") {
    return CreateSingleElementType<FutureType>();
  } else if (token == "Dict") {
    expectChar('[');
    auto key = parse<T>();
    expectChar(',');
    auto val = parse<T>();
    expectChar(']');
    return TypeFactory<T>::template create<DictType>(
        std::move(key), std::move(val));
  } else if (token == "Tuple") {
    std::vector<TypePtr> types;
    expectChar('[');
    while (cur() != "]") {
      types.emplace_back(parse<T>());
      if (cur() != "]") {
        expectChar(',');
      }
    }
    expect("]");
    return TupleType::create(types);
  }
  return nullptr;
}

template <typename T>
TypePtr TypeParser::parse() {
  std::string token = next();
  auto simpleTypeIt = torch::jit::string_to_type_lut().find(token);
  if (simpleTypeIt != torch::jit::string_to_type_lut().end()) {
    if (cur() != "]" && cur() != "," && cur() != "") {
      TORCH_CHECK(
          false, "Simple type ", token, " is followed by ", "invalid chars.");
    }
    contained_types_.insert(token);
    return simpleTypeIt->second;
  } else if (getNonSimpleType().find(token) != getNonSimpleType().end()) {
    contained_types_.insert(token);
    return parseNonSimple<T>(token);
  } else if (token == "__torch__") {
    expectChar('.');
    if (cur() == "torch") {
      // torch bind class starts with __torch__.torch.classes
      return parseTorchbindClassType();
    } else {
      // other class starts with __torch__ following by custom names
      return parseCustomType();
    }
  } else {
    TORCH_CHECK(
        false,
        "Type ",
        token,
        " is not supported in the parser, ",
        "or the token is in wrong format.");
  }
  return nullptr;
}

} // namespace c10
