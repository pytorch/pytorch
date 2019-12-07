#include <ATen/core/jit_type.h>
#include <queue>

namespace torch {
namespace jit {
namespace script {
const std::unordered_map<std::string, c10::TypePtr> &string_to_type_lut();
}
}
}

using torch::jit::script::string_to_type_lut;

namespace c10 {
namespace {
bool isSpecialChar(char a) {
  return a == '[' || a == ',' || a == ']';
}

class TypeParser {
 public:
  explicit TypeParser(std::string pythonStr)
    : pythonStr_(std::move(pythonStr)), start_(0) {
    lex();
  }

  TypePtr parse() {
    std::string token = next();
    auto simpleTypeIt = string_to_type_lut().find(token);
    if (simpleTypeIt != string_to_type_lut().end()) {
      return simpleTypeIt->second;
    } else if (token == "List") {
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
    } else {
      TORCH_CHECK(false, "Type ", token, " is not supported in the parser, ",
          "or the token is in wrong format.");
    }
    return nullptr;
  }

 private:
  void expect(const std::string& s) {
    auto token = next();
    TORCH_CHECK(token == s, "Error when parsing type ", pythonStr_,
        "Expect ", s, ", but get ", token);
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
    while (start_ < pythonStr_.size() && pythonStr_[start_] == ' ') ++start_;
    if (start_ < pythonStr_.size()) {
      if (isSpecialChar(pythonStr_[start_])) {
        next_tokens_.emplace(pythonStr_.substr(start_++, 1));
      } else { // A word
        size_t end = start_;
        for (; end < pythonStr_.size() && !isSpecialChar(pythonStr_[end]) &&
            pythonStr_[end] != ' '; ++end);
        next_tokens_.emplace(pythonStr_.substr(start_, end - start_));
        start_ = end;
      }
    }
  }

  std::string next() {
    TORCH_CHECK(!next_tokens_.empty(), "Empty token queue in mobile type parser.",
        "Check the format of the type string and make sure it's correct.");
    std::string token = next_tokens_.front();
    next_tokens_.pop();
    if (next_tokens_.empty()) {
      lex();
    }
    return token;
  }

  std::string& cur() {
    TORCH_CHECK(!next_tokens_.empty(), "Empty token queue in mobile type parser.",
        "Check the format of the type string and make sure it's correct.");
    return next_tokens_.front();
  }

  std::string pythonStr_;
  size_t start_;
  std::queue<std::string> next_tokens_;
};
} // namespace

TORCH_API TypePtr parseType(const std::string& pythonStr) {
  TypeParser paser(pythonStr);
  return paser.parse();
}
} // namespace c10
