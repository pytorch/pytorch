#include "caffe2/contrib/script/lexer.h"
#include "caffe2/core/common.h"

namespace caffe2 {
namespace script {

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
      throw std::runtime_error("unknown kind: " + caffe2::to_string(kind));
  }
}

SharedParserData& sharedParserData() {
  static SharedParserData data; // safely handles multi-threaded init
  return data;
}
} // namespace script
} // namespace caffe2
