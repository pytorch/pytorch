#include <torch/csrc/jit/frontend/versioned_symbols.h>

#include <torch/csrc/api/include/torch/jit.h>

#include <unordered_map>

namespace torch {
namespace jit {

// Note [Versioned Symbols]
// When the schema or behavior of a symbol changes, serialized Torchscript
// programs using that symbol are likely to break. To prevent those breaks,
// the symbol's historic behavior can be implemented as a Torchscript builtin
// and when an older Torchscript program is loaded the program's uses of the
// symbol can be replaced with the builtin.
//
// For example, a function _test_serialization_subcmul(a, b, alpha) might have
// been improperly implemented as (b - alpha * a).
// Some users may have written and serialized programs using that function,
// however, and fixing it to perform (a - alpha * b) would break their programs.
// Using the "Versioned Symbol" pattern lets you replace
// _test_serialization_subcmul in older programs with a builtin
// _test_serialization_subcmul<version_range> that implements the historic
// behavior. That way old programs preserve their semantics while new programs
// can take advantage of the fix.
//
// To do this:
//
// 1) Identify the file version range where the symbol should be replaced,
//    e.g. versions 0 to 2, inclusive.
// 2) Create one or more builtins implementing the symbol's historic behavior.
//    These should be named <function>_<start_version>_<end_version> and
//    go into the "upgraders" namespace.
//    For example, the test-only aten::_test_serialization_subcmul has a builtin
//    for its "historic" behavior called
//    upgraders::_test_serialization_subcmul_0_2.
// 3) Add a mapping from the symbol to the corresponding SymbolRange
//    in the symbol_range_map (below).
//
// To test your versioning:
//
// 1) Serialize a module demonstrating the historic behavior.
// 2) Save it to test/jit/fixtures.
// 3) Implement your new behavior and bump the version counter.
// 4) Write the builtins and extend the symbol_range_map per the above
//    instructions.
// 5) Create a test in jit/test_save_load.py that loads the old module
//    and verifies it exhibits the historic behavior, then saves and
//    loads the same module and verifies it exhibits the current behavior.
//    See test_versioned_symbols for an example.

// Helper to hold the version range (inclusive on both ends) and the symbol
// to map to for that range.
struct SymbolRange {
  SymbolRange(
      const uint64_t _start_version,
      const uint64_t _end_version,
      const Symbol _sym)
      : start_version_{_start_version},
        end_version_{_end_version},
        sym_{_sym} {}
  const uint64_t start_version_;
  const uint64_t end_version_;
  const Symbol sym_;
};

// NOLINTNEXTLINE(cppcoreguidelines-avoid-non-const-global-variables)
static std::unordered_map<Symbol, SymbolRange> symbol_range_map({
    {Symbol::fromQualString("aten::_test_serialization_subcmul"),
     {0,
      2,
      Symbol::fromQualString("upgraders::_test_serialization_subcmul_0_2")}},
    {Symbol::fromQualString("aten::div"),
     {0, 3, Symbol::fromQualString("upgraders::div_0_3")}},
    {Symbol::fromQualString("aten::div_"),
     {0, 3, Symbol::fromQualString("upgraders::div__0_3")}},
    {Symbol::fromQualString("aten::full"),
     {0, 4, Symbol::fromQualString("upgraders::full_0_4")}},
});

// NOLINTNEXTLINE(cppcoreguidelines-avoid-non-const-global-variables)
static std::unordered_map<NodeKind, uint64_t> kind_min_version_map({
    {aten::div, 4},
    {aten::div_, 4},
    {aten::full, 5}, // NOLINT(cppcoreguidelines-avoid-magic-numbers)
});

Symbol get_symbol_for_version(const Symbol name, const uint64_t version) {
  auto it = symbol_range_map.find(name);
  if (it == symbol_range_map.end()) {
    return name;
  }

  auto& entry = it->second;
  if (entry.start_version_ <= version && entry.end_version_ >= version) {
    return entry.sym_;
  }

  return name;
}

uint64_t get_min_version_for_kind(const NodeKind& kind) {
  auto it = kind_min_version_map.find(kind);
  if (it == kind_min_version_map.end()) {
    return 0;
  }

  return it->second;
}

} // namespace jit
} // namespace torch
