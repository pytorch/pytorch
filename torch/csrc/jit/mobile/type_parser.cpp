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

static constexpr const char* kTypingNamedTuple = "NamedTuple";

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

  TypePtr parseNonSimple(const std::string& token) {
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
    } else if (token == "__torch__") {
      return parseClassType();
    }
    return nullptr;
  }

  TypePtr parse() {
    std::string token = next();
    auto simpleTypeIt = string_to_type_lut().find(token);
    if (simpleTypeIt != string_to_type_lut().end()) {
      if (cur() != "]" && cur() != "," && cur() != "") {
        TORCH_CHECK(
            false, "Simple type ", token, " is followed by ", "invalid chars.");
      }
      return simpleTypeIt->second;
    } else if (getNonSimpleType().find(token) != getNonSimpleType().end()) {
      return parseNonSimple(token);
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
};

/*
CustomTypeParser manages a list of backport from n to n-1 function, and provides
function to check if a specific function exists.
*/
class CustomTypeParser final {
 public:
  CustomTypeParser(CustomTypeParser const&) = delete;
  CustomTypeParser& operator=(CustomTypeParser const&) = delete;
  CustomTypeParser() {
    registerParserFunction(kTypingNamedTuple, parseCustomTypeNamedTuple);
  }

  TypePtr parse(IValue custom_type) {
    // Parse custom type, currently only support NamedTuple
    if (custom_type.isTuple()) {
      auto tuple_difinition = custom_type.toTuple()->elements();
      const std::string type_name = getType(custom_type);
      if (hasParserFunction(type_name)) {
        return parserFunctions()[type_name](custom_type);
      } else {
        TORCH_CHECK(
            false,
            "The custom type: ",
            type_name,
            " is currently not supported.")
      }
    }
    TORCH_CHECK(
        false,
        "For custom type, the definition needs to tuple, however receives : ",
        custom_type.tagKind())
    return nullptr;
  }

  [[nodiscard]] std::unordered_set<std::string> getSupportedTypes() {
    std::unordered_set<std::string> supported_type_list;
    for (const auto& type_name : parserFunctions()) {
      supported_type_list.insert(type_name.first);
    }
    return supported_type_list;
  }

 private:
  [[nodiscard]] bool hasParserFunction(const std::string& custom_type) const {
    return parserFunctions().count(custom_type);
  }

  std::string getType(IValue custom_type) {
    auto difinition = custom_type.toTuple()->elements()[1];
    return difinition.toTuple()->elements()[0].toString()->string();
  }

  std::unordered_map<std::string, std::function<TypePtr(IValue&)>>&
  parserFunctions() const {
    static std::unordered_map<std::string, std::function<TypePtr(IValue&)>>
        custom_type_parser_functions;
    return custom_type_parser_functions;
  }

  // Registry of backport functions.
  void registerParserFunction(
      const std::string& custom_type,
      const std::function<TypePtr(IValue&)>& custom_type_parser_function) {
    parserFunctions()[custom_type] = custom_type_parser_function;
  }

  static TypePtr parseCustomTypeNamedTuple(IValue named_tuple_type_tuple) {
    //  Example NamedTuple type structure:
    //  ('__torch__.A.B.CType',
    //   ('NamedTuple',
    //     ('id_list_features', 'Dict[int, Tensor]'),
    //     ('label', 'Tuple[Tensor, Tensor]'),
    //     ('weight', 'Tuple[Tensor, Tensor]'),
    //     ('id_score_list_features', 'Dict[int, Tensor]')))),
    named_tuple_type_tuple.dump();
    std::vector<IValue> named_tuple_type =
        named_tuple_type_tuple.toTuple()->elements();

    TORCH_CHECK(
        named_tuple_type.size() == 2, "Invalid NamedTuple type definition.")

    const std::string custom_type_name =
        named_tuple_type[0].toString()->string();

    // get namedtuple definition
    std::vector<IValue> name_type_pairs =
        named_tuple_type[1].toTuple()->elements();

    TORCH_CHECK(
        name_type_pairs.size() == 2, "Invalid NamedTuple type definition.")

    at::QualifiedName qualified_name = at::QualifiedName(custom_type_name);
    std::vector<std::string> field_names;
    std::vector<TypePtr> field_types;

    // Find all type names and it's corresponding type
    for (auto const& name_type_pair :
         name_type_pairs[1].toTuple()->elements()) {
      std::vector<IValue> name_type_vector =
          name_type_pair.toTuple()->elements();
      TORCH_CHECK(
          name_type_vector.size() == 2, "Invalid NamedTuple type definition.")
      std::string field_name = name_type_vector[0].toString()->string();
      TypePtr field_type =
          c10::parseType(name_type_vector[1].toString()->string());
      field_names.emplace_back(field_name);
      field_types.emplace_back(field_type);
    }
    // Create the NamedTuple type after reading from the tuple, and add it
    // to function
    auto tt = TupleType::createNamed(qualified_name, field_names, field_types);
    return tt;
  }
};

} // namespace

TORCH_API TypePtr parseType(const std::string& pythonStr) {
  TypeParser paser(pythonStr);
  return paser.parse();
}

TORCH_API TypePtr parseCustomType(IValue custom_type) {
  CustomTypeParser custom_type_parser;
  return custom_type_parser.parse(custom_type);
}

TORCH_API torch::jit::SupportedType getSupportedType() {
  CustomTypeParser custom_type_parser;
  std::unordered_set<std::string> primitive_types;

  for (const auto& it : string_to_type_lut()) {
    primitive_types.insert(it.first);
  }
  primitive_types.insert(
      TypeParser::getNonSimpleType().begin(),
      TypeParser::getNonSimpleType().end());

  return torch::jit::SupportedType{
      primitive_types, custom_type_parser.getSupportedTypes()};
}

} // namespace c10
