#include <ATen/core/interned_strings_class.h>

namespace c10 {

namespace {

struct Entry {
  const char* const qual_name;
  const char* const unqual_name;
  const Symbol sym;
  const Symbol ns_sym;
};

constexpr Entry entries[] = {
#define SYMBOL_ENTRY(n, s) {#n "::" #s, #s, n::s, namespaces::n},

    FORALL_NS_SYMBOLS(SYMBOL_ENTRY)
#undef SYMBOL_ENTRY
};

} // namespace

InternedStrings::InternedStrings()
    : sym_to_info_(static_cast<size_t>(_keys::num_symbols)) {
  // Instead of a loop, this could be done by expanding the
  // assignments directly into FORALL_NS_SYMBOLS, but it would create
  // a huge function (thanks to all the std::string constructors and
  // operator[]s) which would take several minutes to optimize. A
  // static C array of constexpr-constructible structs takes instead
  // no time to compile.
  for (const auto& entry : entries) {
    string_to_sym_[entry.qual_name] = entry.sym;
    sym_to_info_[entry.sym] = {
        entry.ns_sym, entry.qual_name, entry.unqual_name};
  }
}

} // namespace c10
