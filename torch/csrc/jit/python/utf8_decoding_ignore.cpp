#include <torch/csrc/jit/python/utf8_decoding_ignore.h>

namespace torch::jit {

namespace {
thread_local bool kIgnore = false;
}

void setUTF8DecodingIgnore(bool o) {
  kIgnore = o;
}
bool getUTF8DecodingIgnore() {
  return kIgnore;
}

} // namespace torch::jit
