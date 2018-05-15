#include <vector>
#include <stdint.h>
#include <string>
#include <unordered_map>
#include <mutex>
#include <sstream>
#include "ATen/optional.h"
#include "torch/csrc/assertions.h"
#include "torch/csrc/jit/interned_strings.h"

namespace torch { namespace jit {

// TODO: programatically generate this
inline std::string valid_namespaces_str() {
  return "'onnx', 'prim', 'aten', 'attr', 'scope'";
}

// Alas, this is only constexpr in C++14
const unique_t unique_start =
  std::max({static_cast<unique_t>(aten::_keys::num_symbols),
            static_cast<unique_t>(prim::_keys::num_symbols),
            static_cast<unique_t>(onnx::_keys::num_symbols),
            static_cast<unique_t>(attr::_keys::num_symbols)});

SymbolNamespace parseNamespace(const std::string & s) {
  size_t colon_pos = s.find(':');
  if (colon_pos == std::string::npos) {
    std::ostringstream ss;
    ss << "Symbol: illegal unqualified string '" << s << "'; "
       << "all symbols must be qualified, e.g., 'ns::" << s << "'. "
       << "Valid namespaces are: " << valid_namespaces_str();
    throw std::runtime_error(ss.str());
  }
  if (colon_pos == 0) {
    std::ostringstream ss;
    ss << "Symbol: illegal leading colon in '" << s << "'; "
       << "all symbols must have a non-empty namespace. "
       << "Valid namespaces are: " << valid_namespaces_str();
    throw std::runtime_error(ss.str());
  }
  // attr::x
  //     ^___ colon_pos
  //        ^___ s.size()
  if (colon_pos + 2 > s.size()) {
    std::ostringstream ss;
    ss << "Symbol: underlong string '" << s << "'; "
       << "namespace must be followed by double colon and a "
       << "non-empty string.";
    throw std::runtime_error(ss.str());
  }
  if (s[colon_pos + 1] != ':') {
    std::ostringstream ss;
    ss << "Symbol: invalid use of colons in '" << s << "'; "
       << "namespace must be followed by double colon, not a"
       << "single colon.";
    throw std::runtime_error(ss.str());
  }
  auto ns = s.substr(0, colon_pos);
  if (ns == "aten") {
    return SymbolNamespace::aten;
  } else if (ns == "prim") {
    return SymbolNamespace::prim;
  } else if (ns == "onnx") {
    return SymbolNamespace::onnx;
  } else if (ns == "attr") {
    return SymbolNamespace::attr;
  } else if (ns == "scope") {
    return SymbolNamespace::scope;
  } else {
    std::ostringstream ss;
    ss << "Symbol: invalid namespace in '" << s << "'. "
       << "Valid namespaces are: " << valid_namespaces_str();
    throw std::runtime_error(ss.str());
  }
}

struct InternedStrings {
  InternedStrings()
  : next_uniq(unique_start) {
    #define REGISTER_SYMBOL(ns, s) \
      string_to_sym_[#ns "::" #s] = ns::s; \
      sym_to_string_[ns::s] = #ns "::" #s;
    FORALL_BUILTIN_SYMBOLS(REGISTER_SYMBOL)
    #undef REGISTER_SYMBOL
  }
  Symbol symbol(const std::string & s, at::optional<SymbolNamespace> known_ns = at::nullopt) {
    JIT_ASSERT(!known_ns || parseNamespace(s) == known_ns);
    std::lock_guard<std::mutex> guard(mutex_);
    auto it = string_to_sym_.find(s);
    if(it != string_to_sym_.end())
      return it->second;
    unique_t uniq = next_uniq++;
    // TODO: Doing the parsing while we hold the lock is a bit naughty
    SymbolNamespace ns = known_ns ? *known_ns : parseNamespace(s);
    Symbol sym(ns, uniq);
    string_to_sym_[s] = sym;
    sym_to_string_[sym] = s;
    return sym;
  }
  const char * string(Symbol sym) {
    // Builtin Symbols are also in the maps, but
    // we can bypass the need to acquire a lock
    // to read the map for Builtins because we already
    // know their string value
    switch(sym) {
      #define DEFINE_CASE(ns, s) \
        case ns::s: return #ns "::" #s;
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
  unique_t next_uniq;
  std::mutex mutex_;
};

static InternedStrings & globalStrings() {
  static InternedStrings s;
  return s;
}

Symbol Symbol::fromQualString(const std::string & s) {
  return globalStrings().symbol(s);
}

const char* symbolNamespaceString(SymbolNamespace ns) {
  switch (ns) {
    case SymbolNamespace::onnx: return "onnx";
    case SymbolNamespace::prim: return "prim";
    case SymbolNamespace::aten: return "aten";
    case SymbolNamespace::attr: return "attr";
    case SymbolNamespace::scope: return "scope";
    // NB: throwing an exception here causes gcc -O3 to produce far worse code
    default: return "";
  }
}

// NB: I really wanted to define this as std::strlen(symbolNamespaceString(ns)),
// but gcc -O3 doesn't seem to be smart enough to push the std::strlen into
// the switch statement.
size_t symbolNamespaceLength(SymbolNamespace ns) {
  switch (ns) {
    case SymbolNamespace::onnx: return 4;
    case SymbolNamespace::prim: return 4;
    case SymbolNamespace::aten: return 4;
    case SymbolNamespace::attr: return 4;
    case SymbolNamespace::scope: return 5;
    default: return 0;
  }
}

const char * Symbol::toUnqualString() const {
  return globalStrings().string(*this) + symbolNamespaceLength(ns()) + 2 /* double colon */;
}

const char * Symbol::toQualString() const {
  return globalStrings().string(*this);
}

const char * Symbol::toDisplayString() const {
  // TODO: Make this actually return something that's "user friendly".
  // The trouble is that, for this to be usable in printf-style assert
  // statements, this has to return a const char* (whose lifetime is
  // global), so we can't actually assemble a string on the fly.
  return globalStrings().string(*this);
}

std::string Symbol::domainString() const {
  return domain_prefix + symbolNamespaceString(ns());
}

Symbol Symbol::fromDomainAndUnqualString(const std::string & d, const std::string & s) {
  std::string qualString = d.substr(domain_prefix.size()) + "::" + s;
  return fromQualString(qualString);
}

std::string qualifyString(SymbolNamespace ns, const std::string & s) {
  std::string qual_s;
  qual_s.reserve(s.size() + 2 /* double colon */ + symbolNamespaceLength(ns));
  qual_s.append(symbolNamespaceString(ns));
  qual_s.append("::");
  qual_s.append(s);
  return qual_s;
}

Symbol::Symbol(SymbolNamespace ns, const std::string & s)
: value(globalStrings().symbol(qualifyString(ns, s), ns)) {
}

}}
