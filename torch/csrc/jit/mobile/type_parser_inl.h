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
  static TypePtr createTuple(std::vector<TypePtr> types) {
    return std::make_shared<DynamicType>(
        DynamicType::Tag::Tuple, DynamicType::Arguments(types));
  }
  static TypePtr createNamedTuple(
      const std::string& name,
      const std::vector<c10::string_view>& fields,
      const std::vector<TypePtr>& types) {
    return std::make_shared<DynamicType>(
        DynamicType::Tag::Tuple, name, DynamicType::Arguments(fields, types));
  }
};

template <>
struct TORCH_API TypeParser::TypeFactory<c10::Type> {
  template <typename T, typename... Args>
  static TypePtr create(Args&&... args) {
    return T::create(std::forward<Args>(args)...);
  }
  static TypePtr createTuple(std::vector<TypePtr> types) {
    return TupleType::create(std::move(types));
  }
  static TypePtr createNamedTuple(
      const std::string& name,
      const std::vector<c10::string_view>& fields,
      const std::vector<TypePtr>& types) {
    return TupleType::createNamed(name, fields, types);
  }
};

template <typename T>
TypePtr TypeParser::parseNonSimple(const std::string& token) {
  if (token == "List") {
    return parseSingleElementType<T, ListType>();
  } else if (token == "Optional") {
    return parseSingleElementType<T, OptionalType>();
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
    return TypeFactory<T>::createTuple(std::move(types));
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
      return parseCustomType<T>();
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

// Custom type will be following structure:
// "qualified_named[
//   custom_type, [
//       [filed_name_1, field_type_1],
//       [filed_name_2, field_type_2]
//   ]
// ]"
template <typename T>
TypePtr TypeParser::parseCustomType() {
  c10::string_view token = cur();
  std::string qualified_name = "__torch__.";
  qualified_name.reserve(qualified_name.size() + token.size());
  qualified_name.append(token.begin(), token.end());
  next();
  while (cur() == ".") {
    qualified_name.append(next());
    qualified_name.append(next());
  }
  // After cur() moves to the next token after qualified name, if it's "[", it
  // means this custom type follow by it's class definition. Otherwise, it's a
  // barebone qualified name and needs to look up str_type_ptr_map_ to find
  // the typeptr.
  if (cur() == "[") {
    next();
    std::string type_name = next();
    // Currently only supports NamedTuple custom type, if more types need to
    // be supported, extend them here.
    if (type_name == kTypeNamedTuple) {
      contained_types_.insert(kTypeNamedTuple);
      return parseNamedTuple<T>(qualified_name);
    } else {
      TORCH_CHECK(
          false, "Custom Type ", type_name, " is not supported in the parser.");
    }
  } else {
    auto find_type = str_type_ptr_map_.find(qualified_name);
    if (find_type != str_type_ptr_map_.end()) {
      return find_type->second;
    } else {
      // When the type definition can't be found, likely two reasons
      // 1. The type list in bytecode.pkl is not in the correct order
      // 2. This custom type definition doesn't exist in bytecode.pkl type
      // table
      TORCH_CHECK(
          false, "Can't find definition for the type: ", qualified_name);
    }
    return nullptr;
  }
}

// NamedTuple custom type will be following structure:
// "qualified_named[
//   NamedTuple, [
//       [filed_name_1, field_type_1],
//       [filed_name_2, field_type_2]
//   ]
// ]"
//  Example NamedTuple type:
//  "__torch__.base_models.sparse_nn.pytorch_preproc_types.PreprocOutputType[
//     NamedTuple, [
//         [float_features, Tensor],
//         [id_list_features, List[Tensor]],
//         [label,  Tensor],
//         [weight, Tensor],
//         ]
//     ]"
template <typename T>
TypePtr TypeParser::parseNamedTuple(const std::string& qualified_name) {
  std::vector<c10::string_view> field_names;
  std::vector<TypePtr> field_types;
  std::string ns;
  expect(",");
  expect("[");
  while (cur() != "]") {
    expect("[");
    auto field_name = nextView();
    expect(",");
    TypePtr field_type = parse<T>();
    field_names.emplace_back(field_name);
    field_types.emplace_back(field_type);
    expect("]");
    if (cur() == ",") {
      next();
    }
  }
  return TypeFactory<T>::createNamedTuple(
      qualified_name, field_names, field_types);
}

template <typename M, typename T>
TypePtr TypeParser::parseSingleElementType() {
  expectChar('[');
  auto result = TypeFactory<M>::template create<T>(parse<M>());
  expectChar(']');
  return result;
}

} // namespace c10
