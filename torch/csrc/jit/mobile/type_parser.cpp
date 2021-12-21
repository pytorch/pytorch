#include <torch/csrc/jit/mobile/type_parser.h>

#include <ATen/core/jit_type.h>
#include <c10/util/string_view.h>
#include <torch/csrc/jit/frontend/parser_constants.h>
#include <torch/csrc/jit/mobile/runtime_compatibility.h>
#include <torch/custom_class.h>
#include <queue>

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
      type_ptr = parse<c10::DynamicType>();
    }
    typePtrs[i] = type_ptr;
    str_type_ptr_map_[type_ptr->repr_str()] = type_ptr;
  }
  return typePtrs;
}

// The list of non-simple types supported by currrent parser.
const std::unordered_set<std::string>& TypeParser::getNonSimpleType() {
  static std::unordered_set<std::string> nonSimpleTypes{
      "List", "Optional", "Future", "Dict", "Tuple"};
  return nonSimpleTypes;
}

// The list of custom types supported by currrent parser.
const std::unordered_set<std::string>& TypeParser::getCustomType() {
  static std::unordered_set<std::string> customeTypes{
      kTypeTorchbindCustomClass, kTypeNamedTuple};
  return customeTypes;
}

// Given a PyThon str, get all contained types. It's usually used for
// compatibility check between model and runtime. For example:
// PyThon string: "Dict[int, Tuple[Tensor, Tensor, Tensor]]"
// contained type is: [Dict, int, Tuple, Tensor]
const std::unordered_set<std::string>& TypeParser::getContainedTypes() {
  return contained_types_;
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

c10::string_view TypeParser::nextView() {
  TORCH_CHECK(
      !next_token_.empty(),
      "Empty token queue in mobile type parser.",
      "Check the format of the type string and make sure it's correct.");
  c10::string_view token = cur();
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

C10_NODISCARD c10::string_view TypeParser::cur() const {
  return next_token_;
}

TORCH_API std::vector<at::TypePtr> parseType(
    std::vector<std::string>& pythonStrs) {
  at::TypeParser parser(pythonStrs);
  return parser.parseList();
}

const std::unordered_map<std::string, c10::TypePtr>& TypeParser::TypeFactory<
    c10::DynamicType>::baseTypes() {
  static const std::unordered_map<std::string, TypePtr> map = {
#define MAP_ITEM(NAME, TYPE) \
  {#NAME, DynamicTypeTrait<TYPE##Type>::getBaseType()},
      FORALL_JIT_BASE_TYPES(MAP_ITEM)
#undef MAP_ITEM
  };
  return map;
}

const std::unordered_map<std::string, c10::TypePtr>& TypeParser::TypeFactory<
    c10::Type>::baseTypes() {
  static const std::unordered_map<std::string, TypePtr> map = {
#define MAP_ITEM(NAME, TYPE) {#NAME, TYPE##Type::get()},
      FORALL_JIT_BASE_TYPES(MAP_ITEM)
#undef MAP_ITEM
  };
  return map;
}

} // namespace c10
