#include <vector>
#include <stdint.h>
#include <string>
#include <unordered_map>
#include <mutex>
#include "torch/csrc/assertions.h"
#include "torch/csrc/jit/interned_strings.h"

namespace torch { namespace jit {

struct InternedStrings {
  InternedStrings()
  : next_sym(kLastSymbol) {
    #define REGISTER_SYMBOL(s) \
      string_to_sym_[#s] = k##s; \
      sym_to_string_[k##s] = #s;
    FORALL_BUILTIN_SYMBOLS(REGISTER_SYMBOL)
    #undef REGISTER_SYMBOL
  }
  uint32_t symbol(const std::string & s) {
    std::lock_guard<std::mutex> guard(mutex_);
    auto it = string_to_sym_.find(s);
    if(it != string_to_sym_.end())
      return it->second;
    uint32_t k = next_sym++;
    string_to_sym_[s] = k;
    sym_to_string_[k] = s;
    return k;
  }
  const char * string(Symbol sym) {
    // Builtin Symbols are also in the maps, but
    // we can bypass the need to acquire a lock
    // to read the map for Builtins because we already
    // know their string value
    switch(sym) {
      #define DEFINE_CASE(s) \
        case k##s: return #s;
      FORALL_BUILTIN_SYMBOLS(DEFINE_CASE)
      #undef DEFINE_CASE
        default:
          return customString(sym);
    }
  }
private:
  const char * customString(Symbol sym) {
    std::lock_guard<std::mutex> guard(mutex_);
    auto it = sym_to_string_.find(sym);
    JIT_ASSERT(it != sym_to_string_.end());
    return it->second.c_str();
  }
  std::unordered_map<std::string, uint32_t> string_to_sym_;
  std::unordered_map<uint32_t, std::string> sym_to_string_;
  uint32_t next_sym;
  std::mutex mutex_;
};

static InternedStrings & globalStrings() {
  static InternedStrings s;
  return s;
}

const char * Symbol::toString() const {
  return globalStrings().string(*this);
}

Symbol::Symbol(const std::string & s)
: value(globalStrings().symbol(s)) {
}

}}
