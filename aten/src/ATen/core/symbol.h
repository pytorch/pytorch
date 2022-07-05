#pragma once
#include <c10/macros/Export.h>
#include <cstdint>
#include <functional>  // For std::hash
#include <string>


namespace c10 {

// 'prim' symbols are synthetic operators that occur only in the IR
// and don't have corresponding implementations in ATen.

// 'onnx' symbols correspond to ONNX operators.  Their semantics
// are defined in https://github.com/onnx/onnx/blob/master/docs/Operators.md
// The particular version we are targeting is specified by '_onnx_opset_version'
// in torch.onnx.symbolic_helper
//
// In general, most ONNX operators won't get an entry here, because they
// are handled from the Python end.  However, you may occasionally need
// to intern an ONNX symbol here so that you can conveniently write an
// optimization on ONNX operations.

// 'attr' symbols are attribute keys.  They are shared between both ONNX and ATen
// operators (you disambiguate their meaning by looking at the operator itself).
// In general, you only need to define attribute keys that are used by
// onnx or prim; ATen attributes are automatically generated in FORALL_ATTR_BASE_SYMBOLS.

// Note [Symbol allocation]
// ~~~~~~~~~~~~~~~~~~~~~~~~
//
//  1. Symbol namespace is split up into namespaces.
//
//  2. The intended access pattern for built-in symbols is onnx::MatMul
//  in the c10 namespace (this is a Symbol).
//

// Built-in constant definition strategy:
// - Enum is the most convenient way to generate a contiguous sequence
//   of numbers for an identifier.
// - However, an enum gives you a fresh type.  We want onnx::MatMul to
//   be type Symbol, not some random enum type!
// - Therefore, after using enums to generate the sequence of integers,
//   we then declare constexpr Symbols to get everything the actual Symbol
//   type we want.  Symbols must be constexpr to be valid to be "case"ed on.

using unique_t = uint32_t;

const std::string& domain_prefix();

// A Symbol is like an interned string, but with a little extra
// structure; it is namespaced via SymbolNamespace and the resulting
// intern pointers support efficient namespace testing.
struct TORCH_API Symbol {
  explicit constexpr Symbol() : value(0) {};
  explicit constexpr Symbol(unique_t uniq)
  : value(uniq) {}

  // Get a Symbol for a qualified string like "attr::bar"
  static Symbol fromQualString(const std::string & s);

  // Get a Symbol from a domain and an unqualified string like "org.pytorch.attr" and "bar"
  static Symbol fromDomainAndUnqualString(const std::string & d, const std::string & s);

  // Constructors for our various namespaced strings.  This will construct
  // the appropriate namespaced string, e.g., "attr::foo" for the
  // argument "foo", and then attempt to intern it.  DO NOT USE THIS
  // with a string literal; attr::foo should be available in that case
  // (and if it's not, you should add it to the built-ins list above.)
  static Symbol attr(const std::string & s);
  static Symbol aten(const std::string & s);
  static Symbol cuda(const std::string & s);
  static Symbol onnx(const std::string & s);
  static Symbol prim(const std::string & s);
  static Symbol user(const std::string & s);
  static Symbol caffe2(const std::string & s);
  static Symbol dimname(const std::string & s);
  // TODO: eliminate me
  static Symbol scope(const std::string & s);

  bool is_attr() const;
  bool is_aten() const;
  bool is_cuda() const;
  bool is_prim() const;
  bool is_onnx() const;
  bool is_user() const;
  bool is_caffe2() const;
  bool is_dimname() const;

  // So we can switch on this
  constexpr operator unique_t() const {
    return value;
  }

  Symbol ns() const;

  // Give a string corresponding to the unqualified version of this name, e.g.,
  // "mm". Use this in a context where the intended namespace of the string is
  // obvious; this is a *lossy* conversion.
  const char * toUnqualString() const;

  // Give a string corresponding to the qualified version of this name,
  // e.g., "aten::mm".  This string format is made available to Python bindings
  // (so we know how to parse it.)
  const char * toQualString() const;

  // This describes a symbol in a case where humans read it.  At the moment it's
  // the same as toQualString.  This has to be a const char* returned because
  // a lot of printf style macros use it.
  const char * toDisplayString() const;

  // Give a string corresponding to the domain name for the symbol,
  // e.g., "org.pytorch.aten".
  std::string domainString() const;

private:

  explicit Symbol(Symbol ns, const std::string & s);
  unique_t value;
};

static inline bool operator==(Symbol lhs, Symbol rhs) {
  return static_cast<unique_t>(lhs) == static_cast<unique_t>(rhs);
}

inline Symbol Symbol::attr(const std::string & s) { return Symbol::fromQualString("attr::" + s); }
inline Symbol Symbol::aten(const std::string & s)  { return Symbol::fromQualString("aten::" + s); }
inline Symbol Symbol::cuda(const std::string & s)  { return Symbol::fromQualString("cuda::" + s); }
inline Symbol Symbol::onnx(const std::string & s)  { return Symbol::fromQualString("onnx::" + s); }
inline Symbol Symbol::prim(const std::string & s)  { return Symbol::fromQualString("prim::" + s); }
inline Symbol Symbol::scope(const std::string & s) { return Symbol::fromQualString("scope::" + s); }
inline Symbol Symbol::user(const std::string & s) { return Symbol::fromQualString("user::" + s); }
inline Symbol Symbol::caffe2(const std::string & s) { return Symbol::fromQualString("_caffe2::" + s); }
inline Symbol Symbol::dimname(const std::string & s) { return Symbol::fromQualString("dimname::" + s); }

} // namespace c10

// make symbol behave like an integer in hash tables
namespace std {
template <>
struct hash<c10::Symbol> {
  size_t operator()(c10::Symbol s) const {
    return std::hash<uint32_t>()(static_cast<uint32_t>(s));
  }
};
}
