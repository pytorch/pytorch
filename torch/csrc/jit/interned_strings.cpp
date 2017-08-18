#include <vector>
#include <stdint.h>
#include <string>
#include <unordered_map>
#include <mutex>
#include "torch/csrc/jit/interned_strings.h"
#include "torch/csrc/jit/assert.h"

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
  Symbol symbol(const std::string & s) {
    std::lock_guard<std::mutex> guard(mutex_);
    auto it = string_to_sym_.find(s);
    if(it != string_to_sym_.end())
      return it->second;
    Symbol k = next_sym++;
    string_to_sym_[s] = k;
    sym_to_string_[k] = s;
    return k;
  }
  const char * string(Symbol sym) {
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
  std::unordered_map<std::string, Symbol> string_to_sym_;
  std::unordered_map<Symbol, std::string> sym_to_string_;
  Symbol next_sym;
  std::mutex mutex_;
};

static InternedStrings & globalStrings() {
  static InternedStrings s;
  return s;
}

const char * symbolToString(Symbol s) {
  return globalStrings().string(s);
}
Symbol stringToSymbol(const std::string & s) {
  return globalStrings().symbol(s);
}

}}
