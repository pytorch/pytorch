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
bool isSpecialChar(char a) {
  for (const char* c = valid_single_char_tokens; *c; c++) {
    if (a == *c)
      return true;
  }
  return false;
}

class TypeParser {
 public:
  explicit TypeParser(std::string pythonStr)
      : pythonStr_(std::move(pythonStr)), start_(0) {
    lex();
  }

  static std::unordered_set<std::string> getNonSimpleType() {
    static std::unordered_set<std::string> nonSimpleType{
        "List", "Optional", "Dict", "Tuple", "__torch__"};
    return nonSimpleType;
  }

  std::unordered_set<std::string> getContainedTypes() {
    return contained_types_;
  }

  TypePtr parseNonSimple(const std::string& token) {
    if (token == "List") {
      return CreateSingleElementType<ListType>();
    } else if (token == "Union") {
      std::vector<TypePtr> types;
      expect("[");
      while (cur() != "]") {
        types.emplace_back(parse());
        if (cur() != "]") {
          expect(",");
        }
      }
      expect("]");
      return UnionType::create(types);
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
    } else if (token == "__torch__") {
      return parseClassType();
    }
    return nullptr;
  }

  TypePtr parse(bool hard_fail = true) {
    std::string token = next();
    contained_types_.insert(token);
    auto simpleTypeIt = string_to_type_lut().find(token);
    if (simpleTypeIt != string_to_type_lut().end()) {
      if (hard_fail && cur() != "]" && cur() != "," && cur() != "") {
        TORCH_CHECK(
            false, "Simple type ", token, " is followed by ", "invalid chars.");
      }
      return simpleTypeIt->second;
    } else if (getNonSimpleType().find(token) != getNonSimpleType().end()) {
      return parseNonSimple(token);
    } else {
      if (hard_fail) {
        TORCH_CHECK(
            false,
            "Type ",
            token,
            " is not supported in the parser, ",
            "or the token is in wrong format.");
      }
    }
    return nullptr;
  }

 private:
  TypePtr parseClassType() {
    std::vector<std::string> expected_atoms{".", "torch", ".", "classes", "."};
    for (const auto& atom : expected_atoms) {
      expect(atom);
    }

    std::string ns = next();
    expect(".");
    std::string classname = next();

    return torch::getCustomClass(
        std::string("__torch__.torch.classes." + ns + "." + classname));
  }

  void expect(const std::string& s) {
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
  TypePtr CreateSingleElementType() {
    expect("[");
    auto result = T::create(parse());
    expect("]");
    return result;
  }

  void lex() {
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

  std::string next() {
    TORCH_CHECK(
        !next_token_.empty(),
        "Empty token queue in mobile type parser.",
        "Check the format of the type string and make sure it's correct.");
    std::string token = next_token_;
    next_token_ = "";
    lex();
    return token;
  }

  std::string& cur() {
    return next_token_;
  }

  std::string pythonStr_;
  size_t start_;
  std::string next_token_;
  std::unordered_set<std::string> contained_types_;
};
} // namespace

TORCH_API TypePtr parseType(const std::string& pythonStr) {
  TypeParser parser(pythonStr);
  return parser.parse();
}

TORCH_API std::unordered_set<std::string> getContainedTypes(
    const std::string& pythonStr) {
  TypeParser parser(pythonStr);
  parser.parse(false);
  return parser.getContainedTypes();
}

TORCH_API torch::jit::SupportedType getSupportedType() {
  std::unordered_set<std::string> primitive_types;
  for (const auto& it : string_to_type_lut()) {
    primitive_types.insert(it.first);
  }
  primitive_types.insert(
      TypeParser::getNonSimpleType().begin(),
      TypeParser::getNonSimpleType().end());

  return torch::jit::SupportedType{primitive_types, {}};
}

} // namespace c10
