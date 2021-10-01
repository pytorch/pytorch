#include <torch/csrc/jit/mobile/type_parser.h>

#include <ATen/core/jit_type.h>
#include <torch/csrc/jit/frontend/parser_constants.h>
#include <torch/csrc/jit/mobile/runtime_compatibility.h>
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
    expect("[");
    auto key = parse();
    expect(",");
    auto val = parse();
    expect("]");
    return DictType::create(key, val);
  } else if (token == "Tuple") {
    std::vector<TypePtr> types;
    expect("[");
    while (cur() != "]") {
      types.emplace_back(parse());
      if (cur() != "]") {
        expect(",");
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
        false, "Simple type ", token, " is followed by ", "invalid chars.");
  }
  return simpleTypeIt->second;
}

TypePtr TypeParser::parseTorchbindClassType() {
  std::vector<std::string> expected_atoms{".", "torch", ".", "classes", "."};
  for (const auto& atom : expected_atoms) {
    expect(atom);
  }

  std::string ns = next();
  expect(".");
  std::string classname = next();
  contained_types_.insert(kTypeTorchbindCustomClass);
  return torch::getCustomClass(std::string(kTypeTorchbindCustomClass)
                                   .append("." + ns + "." + classname));
}

void TypeParser::expect(const std::string& s) {
  auto token = next();
  TORCH_CHECK(
      token == s,
      "Error when parsing type ",
      pythonStr_,
      "Expect ",
      s,
      ", but get ",
      token);
}

template <class T>
TypePtr TypeParser::CreateSingleElementType() {
  expect("[");
  auto result = T::create(parse());
  expect("]");
  return result;
}

void TypeParser::lex() {
  // skip white spaces
  while (start_ < pythonStr_.size() && pythonStr_[start_] == ' ')
    ++start_;
  if (start_ < pythonStr_.size()) {
    if (isSpecialChar(pythonStr_[start_])) {
      next_token_ = pythonStr_.substr(start_++, 1);
    } else { // A word
      size_t end = start_;
      for (; end < pythonStr_.size() && !isSpecialChar(pythonStr_[end]) &&
           pythonStr_[end] != ' ';
           ++end)
        ;
      next_token_ = pythonStr_.substr(start_, end - start_);
      start_ = end;
    }
  }
}

std::string TypeParser::next() {
  TORCH_CHECK(
      !next_token_.empty(),
      "Empty token queue in mobile type parser.",
      "Check the format of the type string and make sure it's correct.");
  std::string token = next_token_;
  next_token_ = "";
  lex();
  return token;
}

std::string& TypeParser::cur() {
  return next_token_;
}

TORCH_API TypePtr parseType(const std::string& pythonStr) {
  TypeParser parser(pythonStr);
  return parser.parse();
}

} // namespace c10
