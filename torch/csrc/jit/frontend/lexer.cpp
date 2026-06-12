#include <torch/csrc/jit/frontend/lexer.h>

#include <cstring>
#include <string>
#include <unordered_map>

namespace torch::jit {

namespace {

constexpr int getBinaryPrec(int kind) {
  switch (kind) {
    case TK_IF:
    case TK_FOR:
      return 1;
    case TK_AND:
    case TK_OR:
      return 2;
    // reserve a level for unary not
    case TK_IN:
    case TK_NOTIN:
    case '<':
    case '>':
    case TK_IS:
    case TK_ISNOT:
    case TK_EQ:
    case TK_LE:
    case TK_GE:
    case TK_NE:
      return 4;
    case '|':
      return 5;
    case '^':
      return 6;
    case '&':
      return 7;
    case TK_LSHIFT:
    case TK_RSHIFT:
      return 8;
    case '+':
    case '-':
      return 9;
    case '*':
    case '/':
    case TK_FLOOR_DIV:
    case '%':
    case '@':
      return 10;
    case TK_POW:
      return 11;
    default:
      return -1;
  }
}

constexpr int getUnaryPrec(int kind) {
  switch (kind) {
    case TK_NOT:
    case '~':
      return 3;
    case '-':
    case '*':
      return 10;
    default:
      return -1;
  }
}

} // namespace

bool SharedParserData::isUnary(int kind, int* prec) {
  int p = getUnaryPrec(kind);
  if (p >= 0) {
    if (prec) {
      *prec = p;
    }
    return true;
  }
  return false;
}

bool SharedParserData::isBinary(int kind, int* prec) {
  int p = getBinaryPrec(kind);
  if (p >= 0) {
    if (prec) {
      *prec = p;
    }
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
  if (!std::string(str).empty()) \
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
  if (kind < 256) {
    return std::string(1, static_cast<char>(kind));
  }
  switch (kind) {
#define DEFINE_CASE(tok, str, _) \
  case tok:                      \
    return str;
    TC_FORALL_TOKEN_KINDS(DEFINE_CASE)
#undef DEFINE_CASE
    default:
      TORCH_CHECK(false, "Unknown kind: ", kind);
  }
}

C10_EXPORT SharedParserData& sharedParserData() {
  static SharedParserData data; // safely handles multi-threaded init
  return data;
}

} // namespace torch::jit
