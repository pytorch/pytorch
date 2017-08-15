#pragma once
#include <vector>
#include <stdint.h>
#include <string>
#include <unordered_map>
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
  Symbol symbol(const std::string & s) {
    auto it = string_to_sym_.find(s);
    if(it != string_to_sym_.end())
      return it->second;
    Symbol k = next_sym++;
    string_to_sym_[s] = k;
    sym_to_string_[k] = s;
    return k;
  }
  const std::string & string(Symbol sym) {
    auto it = sym_to_string_.find(sym);
    JIT_ASSERT(it != sym_to_string_.end());
    return it->second;
  }
private:
  std::unordered_map<std::string, Symbol> string_to_sym_;
  std::unordered_map<Symbol, std::string> sym_to_string_;
  Symbol next_sym;
};

static InternedStrings & globalStrings() {
  static InternedStrings s;
  return s;
}

const std::string & symbolToString(Symbol s) {
  return globalStrings().string(s);
}
Symbol stringToSymbol(const std::string & s) {
  return globalStrings().symbol(s);
}

}}
