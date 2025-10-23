// aten_interned_strings.h includes the names of all operators
#undef TORCH_ASSERT_ONLY_METHOD_OPERATORS

#include <ATen/core/interned_strings.h>
#include <cstdint>
#include <cstring>
#include <mutex>
#include <sstream>
#include <string>
#include <c10/util/Exception.h>
#include <ATen/core/interned_strings_class.h>

namespace c10 {

const std::string& domain_prefix() {
  static const std::string _domain_prefix = "org.pytorch.";
  return _domain_prefix;
}

Symbol InternedStrings::symbol(const std::string& s) {
  std::lock_guard<std::mutex> guard(mutex_);
  return _symbol(s);
}

std::pair<const char*, const char*> InternedStrings::string(Symbol sym) {
  // Builtin Symbols are also in the maps, but
  // we can bypass the need to acquire a lock
  // to read the map for Builtins because we already
  // know their string value
#if defined C10_MOBILE
  return customString(sym);
#else
  switch (sym) {
#define DEFINE_CASE(ns, s) \
  case static_cast<unique_t>(ns::s): \
    return {#ns "::" #s, #s};
    FORALL_NS_SYMBOLS(DEFINE_CASE)
#undef DEFINE_CASE
    default:
      return customString(sym);
  }
#endif
}

Symbol InternedStrings::ns(Symbol sym) {
#if defined C10_MOBILE
  std::lock_guard<std::mutex> guard(mutex_);
  return sym_to_info_.at(sym).ns;
#else
  switch (sym) {
#define DEFINE_CASE(ns, s) \
  case static_cast<unique_t>(ns::s): \
    return namespaces::ns;
    // NOLINTNEXTLINE(bugprone-branch-clone)
    FORALL_NS_SYMBOLS(DEFINE_CASE)
#undef DEFINE_CASE
    default: {
      std::lock_guard<std::mutex> guard(mutex_);
      return sym_to_info_.at(sym).ns;
    }
  }
#endif
}

Symbol InternedStrings::_symbol(const std::string& s) {
  auto it = string_to_sym_.find(s);
  if (it != string_to_sym_.end())
    return it->second;

  auto pos = s.find("::");
  TORCH_CHECK(pos != std::string::npos, "all symbols must have a namespace, <namespace>::<string>, but found: ", s);
  Symbol ns = _symbol("namespaces::" + s.substr(0, pos));

  Symbol sym(sym_to_info_.size());
  string_to_sym_[s] = sym;
  sym_to_info_.push_back({ns, s, s.substr(pos + strlen("::"))});
  return sym;
}

std::pair<const char*, const char*> InternedStrings::customString(Symbol sym) {
  std::lock_guard<std::mutex> guard(mutex_);
  SymbolInfo& s = sym_to_info_.at(sym);
  return {s.qual_name.c_str(), s.unqual_name.c_str()};
}

static InternedStrings & globalStrings() {
  static InternedStrings s;
  return s;
}

Symbol Symbol::fromQualString(const std::string & s) {
  return globalStrings().symbol(s);
}

const char * Symbol::toUnqualString() const {
  return globalStrings().string(*this).second;
}

const char * Symbol::toQualString() const {
  return globalStrings().string(*this).first;
}

const char * Symbol::toDisplayString() const {
  // TODO: Make this actually return something that's "user friendly".
  // The trouble is that, for this to be usable in printf-style assert
  // statements, this has to return a const char* (whose lifetime is
  // global), so we can't actually assemble a string on the fly.
  return toQualString();
}

Symbol Symbol::ns() const {
  return globalStrings().ns(*this);
}

std::string Symbol::domainString() const {
  return domain_prefix() + ns().toUnqualString();
}

Symbol Symbol::fromDomainAndUnqualString(const std::string & d, const std::string & s) {
  TORCH_CHECK(d.compare(0, domain_prefix().size(), domain_prefix()) == 0, "Symbol: domain string is expected to be prefixed with '", domain_prefix(), "', e.g. 'org.pytorch.aten'");
  std::string qualString = d.substr(domain_prefix().size()) + "::" + s;
  return fromQualString(qualString);
}

bool Symbol::is_attr() const { return ns() == namespaces::attr; }
bool Symbol::is_aten() const { return ns() == namespaces::aten; }
bool Symbol::is_cuda() const { return ns() == namespaces::cuda; }
bool Symbol::is_prim() const { return ns() == namespaces::prim; }
bool Symbol::is_prims() const { return ns() == namespaces::prims; }
bool Symbol::is_nvprims() const { return ns() == namespaces::nvprims; }
bool Symbol::is_onnx() const { return ns() == namespaces::onnx; }
bool Symbol::is_user() const { return ns() == namespaces::user; }
bool Symbol::is_caffe2() const { return ns() == namespaces::_caffe2; }
bool Symbol::is_dimname() const { return ns() == namespaces::dimname; }

} // namespace c10
