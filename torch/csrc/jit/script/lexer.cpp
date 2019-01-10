#include "torch/csrc/jit/script/lexer.h"
#include <string>
#include <unordered_map>
#include <mutex>

namespace torch {
namespace jit {
namespace script {

int stringToKind(std::string str) {
  static std::once_flag init_flag;
  static std::unordered_map<std::string, int> str_to_kind;
  std::call_once(init_flag, []() {
    for (char tok : std::string(valid_single_char_tokens))
      str_to_kind[std::string(1, tok)] = tok;
#define DEFINE_CASE(tok, _, str) \
    if (std::string(str) != "") str_to_kind[str] = tok;
    TC_FORALL_TOKEN_KINDS(DEFINE_CASE)
#undef DEFINE_CASE
  });
  try {
    return str_to_kind.at(str);
  } catch (std::out_of_range& err) {
    throw std::out_of_range("unknown token in stringToKind");
  }
}

std::string kindToString(int kind) {
  if (kind < 256)
    return std::string(1, kind);
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

SharedParserData& sharedParserData() {
  static SharedParserData data; // safely handles multi-threaded init
  return data;
}
} // namespace script
} // namespace jit
} // namespace torch
