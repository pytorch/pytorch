// aten_interned_strings.h includes the names of all operators
#undef TORCH_ASSERT_ONLY_METHOD_OPERATORS

#include <ATen/core/interned_strings.h>
#include <ATen/core/interned_strings_class.h>

#include <cstring>

namespace c10 {

namespace {

// NOLINTBEGIN(cppcoreguidelines-avoid-const-or-ref-data-members)
struct Entry {
  const char* const namespace_;
  const char* const unqual_name;
  const Symbol sym;
  const Symbol ns_sym;
};
// NOLINTEND(cppcoreguidelines-avoid-const-or-ref-data-members)

std::string qual_name_for_entry(const Entry& entry) {
  const char* const sep = "::";
  const auto namespace_len = strlen(entry.namespace_);
  const auto sep_len = strlen(sep);
  const auto unqual_name_len = strlen(entry.unqual_name);
  std::string s;
  s.reserve(namespace_len + sep_len + unqual_name_len);
  s.append(entry.namespace_, namespace_len);
  s.append(sep, sep_len);
  s.append(entry.unqual_name, unqual_name_len);
  return s;
}

// NOTE: we could save even more space by packing the string data as follows:
// constexpr char namespaces[] = "namespaces\0prim\0aten\0...";
// constexpr char unqual_names[] = "prim\0aten\0cuda\0...";
// and then storing two uint16_t (or uint32_t if needed) offsets into
// the raw string tables in Entry instead of 8-byte pointers.
// I haven't implemented that because it's not clear to me how to
// dedupe the namespaces array at compile-time, particularly in C++14,
// but it would be straightforward if we switched to codegen.
// NOLINTNEXTLINE(modernize-avoid-c-arrays,cppcoreguidelines-avoid-c-arrays)
constexpr Entry entries[] = {
#define SYMBOL_ENTRY(n, s) {#n, #s, n::s, namespaces::n},

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
    auto qual_name = qual_name_for_entry(entry);
    string_to_sym_[qual_name] = entry.sym;
    sym_to_info_[entry.sym] = {
        entry.ns_sym, std::move(qual_name), entry.unqual_name};
  }
}

} // namespace c10
