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
static constexpr const char* kTypeNamedTuple = "NamedTuple";

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

  explicit TypeParser(std::vector<std::string>& pythonStrs)
      : pythonStrs_(pythonStrs) {}

  // For the Python string list parsing, the order of the Python string matters.
  // In bytecode, the order of the type list correspondings to the order of
  // instruction. In nested type, the lowest level type will be at the beginning
  // of the type list. It is possible to parse it without worrying about
  // ordering, but it also introduces 1) extra cost to process nested type in
  // the correct order 2) lost the benifit that the instruction order is likely
  // problematic if type list parsing fails.
  std::vector<TypePtr> parseList() {
    std::vector<TypePtr> typePtrs;
    static const c10::QualifiedName classPrefix = "__torch__.torch.classes";
    for (const auto& pythonStr : pythonStrs_) {
      c10::QualifiedName qn(pythonStr);
      c10::TypePtr type_ptr;
      if (classPrefix.isPrefixOf(qn)) {
        type_ptr = torch::getCustomClass(qn.qualifiedName());
        TORCH_CHECK(
            type_ptr,
            "The implementation of class ",
            qn.qualifiedName(),
            " cannot be found.");
      } else {
        pythonStr_ = pythonStr;
        start_ = 0;
        lex();
        type_ptr = parse();
      }
      typePtrs.emplace_back(type_ptr);
      str_type_ptr_map_[type_ptr->repr_str()] = type_ptr;
    }
    return typePtrs;
  }

  // The list of non-simple types supported by currrent parser.
  static std::unordered_set<std::string> getNonSimpleType() {
    static std::unordered_set<std::string> nonSimpleTypes{
        "List", "Union", "Optional", "Future", "Dict", "Tuple"};
    return nonSimpleTypes;
  }

  // The list of custom types supported by currrent parser.
  static std::unordered_set<std::string> getCustomType() {
    static std::unordered_set<std::string> customeTypes{
        kTypeTorchbindCustomClass, kTypeNamedTuple};
    return customeTypes;
  }

  // Given a PyThon str, get all contained types. It's usually used for
  // compatibility check between model and runtime. For example:
  // PyThon string: "Dict[int, Tuple[Tensor, Tensor, Tensor]]"
  // contained type is: [Dict, int, Tuple, Tensor]
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
      contained_types_.insert(token);
      return simpleTypeIt->second;
    } else if (getNonSimpleType().find(token) != getNonSimpleType().end()) {
      contained_types_.insert(token);
      return parseNonSimple(token);
    } else if (token == "__torch__") {
      expect(".");
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

 private:
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
  TypePtr parseNamedTuple(const std::string& qualified_name) {
    std::vector<std::string> field_names;
    std::vector<TypePtr> field_types;
    std::string ns;
    expect(",");
    expect("[");
    while (cur() != "]") {
      expect("[");
      std::string field_name = next();
      expect(",");
      TypePtr field_type = parse();
      field_names.emplace_back(field_name);
      field_types.emplace_back(field_type);
      std::cout << cur() << std::endl;
      expect("]");
      if (cur() == ",") {
        next();
      }
    }
    return TupleType::createNamed(qualified_name, field_names, field_types);
  }

  // Custom type will be following structure:
  // "qualified_named[
  //   custom_type, [
  //       [filed_name_1, field_type_1],
  //       [filed_name_2, field_type_2]
  //   ]
  // ]"
  TypePtr parseCustomType() {
    std::string qualified_name = "__torch__." + cur();
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
            false,
            "Custom Type ",
            type_name,
            " is not supported in the parser.");
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

  TypePtr parseTorchbindClassType() {
    std::vector<std::string> expected_atoms{"torch", ".", "classes", "."};
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

  std::vector<std::string> pythonStrs_;
  std::unordered_map<std::string, c10::TypePtr> str_type_ptr_map_;

  // Store all contained types when parsing a string
  std::unordered_set<std::string> contained_types_;
};
} // namespace

TORCH_API TypePtr parseType(const std::string& pythonStr) {
  TypeParser parser(pythonStr);
  return parser.parse();
}

TORCH_API std::vector<TypePtr> parseType(std::vector<std::string>& pythonStrs) {
  TypeParser parser(pythonStrs);
  return parser.parseList();
}

// Get all contained type given a string
TORCH_API std::unordered_set<std::string> getContainedTypes(
    const std::string& pythonStr) {
  TypeParser parser(pythonStr);
  parser.parse();
  return parser.getContainedTypes();
}

TORCH_API std::unordered_set<std::string> getContainedTypes(
    std::vector<std::string>& pythonStrs) {
  TypeParser parser(pythonStrs);
  parser.parseList();
  return parser.getContainedTypes();
}

// Get all supported type given a runtime
TORCH_API std::unordered_set<std::string> getSupportedType() {
  std::unordered_set<std::string> supported_types;
  for (const auto& it : string_to_type_lut()) {
    supported_types.insert(it.first);
  }
  supported_types.insert(
      TypeParser::getNonSimpleType().begin(),
      TypeParser::getNonSimpleType().end());
  supported_types.insert(
      TypeParser::getCustomType().begin(), TypeParser::getCustomType().end());

  return supported_types;
}

} // namespace c10
