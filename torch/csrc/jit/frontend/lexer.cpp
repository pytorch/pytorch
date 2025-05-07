#include <torch/csrc/jit/frontend/lexer.h>

#include <c10/util/Exception.h>

#include <cstring>
#include <string>
#include <unordered_map>

namespace torch::jit {

static const std::unordered_map<int, int> binary_prec = {
    {TK_IF, 1},
    {TK_FOR, 1},
    {TK_AND, 2},
    {TK_OR, 2},
    // reserve a level for unary not
    {TK_IN, 4},
    {TK_NOTIN, 4},
    {'<', 4},
    {'>', 4},
    {TK_IS, 4},
    {TK_ISNOT, 4},
    {TK_EQ, 4},
    {TK_LE, 4},
    {TK_GE, 4},
    {TK_NE, 4},
    {'|', 5},
    {'^', 6},
    {'&', 7},
    {TK_LSHIFT, 8},
    {TK_RSHIFT, 8},
    {'+', 9},
    {'-', 9},
    {'*', 10},
    {'/', 10},
    {TK_FLOOR_DIV, 10},
    {'%', 10},
    {'@', 10},
    {TK_POW, 11},
};

static const std::unordered_map<int, int> unary_prec = {
    {TK_NOT, 3},
    {'~', 3},
    {'-', 10},
    {'*', 10},
};

bool SharedParserData::isUnary(int kind, int* prec) {
  auto it = unary_prec.find(kind);
  if (it != unary_prec.end()) {
    *prec = it->second;
    return true;
  }
  return false;
}
bool SharedParserData::isBinary(int kind, int* prec) {
  auto it = binary_prec.find(kind);
  if (it != binary_prec.end()) {
    *prec = it->second;
    return true;
  }
  return false;
}

C10_EXPORT int stringToKind(const std::string& str) {
  static std::unordered_map<std::string, int> str_to_kind = []() {
    std::unordered_map<std::string, int> ret_str_to_kind;
    ret_str_to_kind.reserve(std::strlen(valid_single_char_tokens));
    for (const char* tok = valid_single_char_tokens; *tok; tok++) {
      ret_str_to_kind[std::string(1, *tok)] = static_cast<unsigned char>(*tok);
    }
#define DEFINE_CASE(tok, _, str) \
  if (std::string(str) != "")    \
    ret_str_to_kind[str] = tok;
    TC_FORALL_TOKEN_KINDS(DEFINE_CASE)
#undef DEFINE_CASE
    return ret_str_to_kind;
  }();
  try {
    return str_to_kind.at(str);
  } catch (std::out_of_range&) {
    throw std::out_of_range("unknown token in stringToKind");
  }
}

C10_EXPORT std::string kindToString(int kind) {
  if (kind < 256)
    return std::string(1, static_cast<char>(kind));
  switch (kind) {
#define DEFINE_CASE(tok, str, _) \
  case tok:                      \
    return str;
    TC_FORALL_TOKEN_KINDS(DEFINE_CASE)
#undef DEFINE_CASE
    default:
      throw std::runtime_error("Unknown kind: " + std::to_string(kind));
  }
}

C10_EXPORT SharedParserData& sharedParserData() {
  static SharedParserData data; // safely handles multi-threaded init
  return data;
}

} // namespace torch::jit
