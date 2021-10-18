#include <ATen/core/jit_type.h>
#include <c10/util/string_view.h>
#include <torch/csrc/jit/frontend/parser_constants.h>
#include <torch/csrc/jit/mobile/runtime_compatibility.h>
#include <torch/csrc/jit/mobile/type_parser.h>
#include <torch/custom_class.h>
#include <queue>

namespace torch {
namespace jit {
const std::unordered_map<std::string, c10::TypePtr>& string_to_type_lut();
}
} // namespace torch

using torch::jit::string_to_type_lut;
using torch::jit::valid_single_char_tokens;

namespace c10 {
namespace {

// Torchbind custom class always starts with the follow prefix, so use it as an
// identifier for torchbind custom class type
static constexpr const char* kTypeTorchbindCustomClass =
    "__torch__.torch.classes";

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

// The list of non-simple types supported by currrent parser.
std::unordered_set<std::string> TypeParser::getNonSimpleType() {
  static std::unordered_set<std::string> nonSimpleTypes{
      "List", "Union", "Optional", "Future", "Dict", "Tuple"};
  return nonSimpleTypes;
}

// The list of custom types supported by currrent parser.
std::unordered_set<std::string> TypeParser::getCustomType() {
  static std::unordered_set<std::string> customeTypes{
      kTypeTorchbindCustomClass};
  return customeTypes;
}

// Given a PyThon str, get all contained types. It's usually used for
// compatibility check between model and runtime. For example:
// PyThon string: "Dict[int, Tuple[Tensor, Tensor, Tensor]]"
// contained type is: [Dict, int, Tuple, Tensor]
std::unordered_set<std::string> TypeParser::getContainedTypes() {
  return contained_types_;
}

TypePtr TypeParser::parseNonSimple(const std::string& token) {
  if (token == "List") {
    return CreateSingleElementType<ListType>();
  } else if (token == "Optional") {
    return CreateSingleElementType<OptionalType>();
  } else if (token == "Future") {
    return CreateSingleElementType<FutureType>();
  } else if (token == "Dict") {
    expectChar('[');
    auto key = parse();
    expectChar(',');
    auto val = parse();
    expectChar(']');
    return DictType::create(std::move(key), std::move(val));
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
    return TupleType::create(types);
  }
  return nullptr;
}

TypePtr TypeParser::parse() {
  std::string token = next();
  auto simpleTypeIt = string_to_type_lut().find(token);
  if (simpleTypeIt != string_to_type_lut().end()) {
    if (cur() != "]" && cur() != "," && cur() != "") {
      TORCH_CHECK(
          false, "Simple type ", token, " is followed by ", "invalid chars.");
    }
    contained_types_.insert(token);
    return simpleTypeIt->second;
  } else if (getNonSimpleType().find(token) != getNonSimpleType().end()) {
    contained_types_.insert(token);
    return parseNonSimple(token);
  } else if (token == "__torch__") {
    return parseTorchbindClassType();
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

TypePtr TypeParser::parseTorchbindClassType() {
  static constexpr std::array<const char*, 5> expected_atoms = {
      ".", "torch", ".", "classes", "."};
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
  c10::string_view token = cur();
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
  c10::string_view token = cur();
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

template <class T>
TypePtr TypeParser::CreateSingleElementType() {
  expectChar('[');
  auto result = T::create(parse());
  expectChar(']');
  return result;
}

void TypeParser::lex() {
  // skip white spaces
  while (start_ < pythonStr_.size() && pythonStr_[start_] == ' ')
    ++start_;
  if (start_ < pythonStr_.size()) {
    if (isSpecialChar(pythonStr_[start_])) {
      next_token_ = c10::string_view(pythonStr_.data() + start_++, 1);
    } else { // A word
      size_t end = start_;
      for (; end < pythonStr_.size() && !isSpecialChar(pythonStr_[end]) &&
           pythonStr_[end] != ' ';
           ++end)
        ;
      next_token_ = c10::string_view(pythonStr_.data() + start_, end - start_);
      start_ = end;
    }
  }
}

std::string TypeParser::next() {
  TORCH_CHECK(
      !next_token_.empty(),
      "Empty token queue in mobile type parser.",
      "Check the format of the type string and make sure it's correct.");
  c10::string_view token = cur();
  std::string ret(token.begin(), token.end());
  advance();
  return ret;
}

void TypeParser::advance() {
  next_token_ = "";
  lex();
}

C10_NODISCARD c10::string_view TypeParser::cur() const {
  return next_token_;
}

TORCH_API TypePtr parseType(const std::string& pythonStr) {
  TypeParser parser(pythonStr);
  return parser.parse();
}

} // namespace c10
