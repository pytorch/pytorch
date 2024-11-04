#include <torch/csrc/jit/mobile/type_parser.h>

#include <ATen/core/jit_type.h>
#include <ATen/core/type_factory.h>
#include <c10/util/string_view.h>
#include <torch/csrc/jit/frontend/parser_constants.h>
#include <torch/custom_class.h>

using torch::jit::valid_single_char_tokens;

namespace c10 {

namespace {

// Torchbind custom class always starts with the follow prefix, so use it as
// an identifier for torchbind custom class type
static constexpr const char* kTypeTorchbindCustomClass =
    "__torch__.torch.classes";
static constexpr const char* kTypeNamedTuple = "NamedTuple";

bool isSpecialChar(char a) {
  for (const char* c = valid_single_char_tokens; *c; c++) {
    if (a == *c)
      return true;
  }
  return false;
}
} // namespace

TypeParser::TypeParser(std::string pythonStr)
    : pythonStr_(std::move(pythonStr)), start_(0) {
  lex();
}

TypeParser::TypeParser(std::vector<std::string>& pythonStrs)
    : start_(0), pythonStrs_(pythonStrs) {}

// For the Python string list parsing, the order of the Python string matters.
// In bytecode, the order of the type list correspondings to the order of
// instruction. In nested type, the lowest level type will be at the beginning
// of the type list. It is possible to parse it without worrying about
// ordering, but it also introduces 1) extra cost to process nested type to
// the correct order 2) lost the benifit that the instruction order is likely
// problematic if type list parsing fails.
std::vector<TypePtr> TypeParser::parseList() {
  std::vector<TypePtr> typePtrs;
  typePtrs.resize(pythonStrs_.size());
  static const c10::QualifiedName classPrefix = "__torch__.torch.classes";
  for (size_t i = 0; i < pythonStrs_.size(); i++) {
    c10::QualifiedName qn(pythonStrs_[i]);
    c10::TypePtr type_ptr;
    if (classPrefix.isPrefixOf(qn)) {
      type_ptr = torch::getCustomClass(qn.qualifiedName());
      TORCH_CHECK(
          type_ptr,
          "The implementation of class ",
          qn.qualifiedName(),
          " cannot be found.");
    } else {
      pythonStr_ = pythonStrs_[i];
      start_ = 0;
      lex();
      type_ptr = parse();
    }
    typePtrs[i] = type_ptr;
    str_type_ptr_map_[type_ptr->repr_str()] = type_ptr;
  }
  return typePtrs;
}

// The list of non-simple types supported by current parser.
const std::unordered_set<std::string>& TypeParser::getNonSimpleType() {
  static std::unordered_set<std::string> nonSimpleTypes{
      "List", "Optional", "Dict", "Tuple"};
  return nonSimpleTypes;
}

// The list of custom types supported by current parser.
const std::unordered_set<std::string>& TypeParser::getCustomType() {
  static std::unordered_set<std::string> customeTypes{
      kTypeTorchbindCustomClass, kTypeNamedTuple};
  return customeTypes;
}

// Given a PyThon str, get all contained types. It's usually used for
// compatibility check between model and runtime. For example:
// PyThon string: "Dict[int, Tuple[Tensor, Tensor, Tensor]]"
// contained type is: [Dict, int, Tuple, Tensor]
std::unordered_set<std::string> TypeParser::getContainedTypes() {
  return contained_types_;
}

template <typename T>
TypePtr TypeParser::parseSingleElementType() {
  expectChar('[');
  auto result = DynamicTypeFactory::create<T>(parse());
  expectChar(']');
  return result;
}

TypePtr TypeParser::parseNonSimple(const std::string& token) {
  if (token == "List") {
    return parseSingleElementType<ListType>();
  } else if (token == "Optional") {
    return parseSingleElementType<OptionalType>();
  } else if (token == "Dict") {
    expectChar('[');
    auto key = parse();
    expectChar(',');
    auto val = parse();
    expectChar(']');
    return DynamicTypeFactory::create<DictType>(std::move(key), std::move(val));
  } else if (token == "Tuple") {
    std::vector<TypePtr> types;
    expectChar('[');
    while (cur() != "]") {
      types.emplace_back(parse());
      if (cur() != "]") {
        expectChar(',');
      }
    }
    expect("]");
    return DynamicTypeFactory::create<TupleType>(types);
  }
  return nullptr;
}

TypePtr TypeParser::parse() {
  std::string token = next();
  const auto& baseTypes = DynamicTypeFactory::basePythonTypes();
  auto simpleTypeIt = baseTypes.find(token);
  if (simpleTypeIt != baseTypes.end()) {
    if (cur() != "]" && cur() != "," && !cur().empty()) {
      TORCH_CHECK(
          false, "Simple type ", token, " is followed by ", "invalid chars.");
    }
    contained_types_.insert(token);
    return simpleTypeIt->second;
  } else if (getNonSimpleType().find(token) != getNonSimpleType().end()) {
    contained_types_.insert(token);
    return parseNonSimple(token);
  } else if (token == "__torch__") {
    expectChar('.');
    if (cur() == "torch") {
      // torch bind class starts with __torch__.torch.classes
      return parseTorchbindClassType();
    } else {
      // other class starts with __torch__ following by custom names
      return parseCustomType();
    }
  } else if (token == "Union") {
    // TODO Union types are not supported on embedded runtime, and we need to
    // generate compiler errors for users scripting UnionTypes. Right now
    // for preserving backward compatibility we have to return a nullptr since
    // it does not get involved in type reflection.
    return nullptr;
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
TypePtr TypeParser::parseNamedTuple(const std::string& qualified_name) {
  std::vector<c10::string_view> field_names;
  std::vector<TypePtr> field_types;
  expect(",");
  expect("[");
  while (cur() != "]") {
    expect("[");
    auto field_name = nextView();
    expect(",");
    TypePtr field_type = parse();
    field_names.emplace_back(field_name);
    field_types.emplace_back(field_type);
    expect("]");
    if (cur() == ",") {
      next();
    }
  }
  return DynamicTypeFactory::createNamedTuple(
      qualified_name, field_names, field_types);
}

// Custom type will be following structure:
// "qualified_named[
//   custom_type, [
//       [filed_name_1, field_type_1],
//       [filed_name_2, field_type_2]
//   ]
// ]"
TypePtr TypeParser::parseCustomType() {
  std::string_view token = cur();
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
      return parseNamedTuple(qualified_name);
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

TypePtr TypeParser::parseTorchbindClassType() {
  static constexpr std::array<const char*, 4> expected_atoms = {
      "torch", ".", "classes", "."};
  for (const auto& atom : expected_atoms) {
    expect(atom);
  }
  std::string ns = next();
  expectChar('.');
  std::string classname = next();
  std::string customClassName = "__torch__.torch.classes.";
  customClassName.reserve(
      customClassName.size() + ns.size() + 1 + classname.size());
  customClassName.append(ns);
  customClassName.push_back('.');
  customClassName.append(classname);
  return torch::getCustomClass(customClassName);
}

void TypeParser::expect(const char* s) {
  std::string_view token = cur();
  TORCH_CHECK(
      token == s,
      "Error when parsing type ",
      pythonStr_,
      ": Expect ",
      s,
      ", but get ",
      token);
  advance();
}

// c10::string_view::operator== calls memcmp to compare against the target
// string; we can do better if we specialize for a single character.
void TypeParser::expectChar(char c) {
  std::string_view token = cur();
  TORCH_CHECK(
      token.size() == 1 && token[0] == c,
      "Error when parsing type ",
      pythonStr_,
      ": Expect ",
      c,
      ", but get ",
      token);
  advance();
}

void TypeParser::lex() {
  // skip white spaces
  while (start_ < pythonStr_.size() && pythonStr_[start_] == ' ')
    ++start_;
  if (start_ < pythonStr_.size()) {
    if (isSpecialChar(pythonStr_[start_])) {
      next_token_ = std::string_view(pythonStr_.data() + start_++, 1);
    } else { // A word
      size_t end = start_;
      for (; end < pythonStr_.size() && !isSpecialChar(pythonStr_[end]) &&
           pythonStr_[end] != ' ';
           ++end)
        ;
      next_token_ = std::string_view(pythonStr_.data() + start_, end - start_);
      start_ = end;
    }
  }
}

std::string_view TypeParser::nextView() {
  TORCH_CHECK(
      !next_token_.empty(),
      "Empty token queue in mobile type parser.",
      "Check the format of the type string and make sure it's correct.");
  std::string_view token = cur();
  advance();
  return token;
}

std::string TypeParser::next() {
  auto token = nextView();
  return std::string(token.begin(), token.end());
}

void TypeParser::advance() {
  next_token_ = "";
  lex();
}

[[nodiscard]] std::string_view TypeParser::cur() const {
  return next_token_;
}

TORCH_API at::TypePtr parseType(const std::string& pythonStr) {
  at::TypeParser parser(pythonStr);
  return parser.parse();
}

TORCH_API std::vector<at::TypePtr> parseType(
    std::vector<std::string>& pythonStrs) {
  at::TypeParser parser(pythonStrs);
  return parser.parseList();
}

} // namespace c10
