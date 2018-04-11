#pragma once
#include <vector>
#include <stdint.h>
#include <string>
#include <unordered_map>
#include <algorithm>

#include "torch/csrc/jit/generated/aten_interned_strings.h"

namespace torch { namespace jit {

// Every symbol is classified in a namespace, specifying what kind of symbol it
// is.  Unsigned char to ensure widening to unique_t (also an unsigned type)
enum class SymbolNamespace : unsigned char {
  onnx  = 'o',
  prim  = 'p',
  aten  = 't',
  // NB: ONNX and ATen attributes all live in a unified namespace, as
  // their interpretation depends on the operator name (which is namespaced)
  attr  = 'a',
  // TODO: eliminate me
  scope = 's'
};

// Primitive symbols are synthetic operators that occur only in the IR
// and don't have corresponding implementations in ATen.
//
// TODO: We need documentation for all of these symbols.
//
// TODO: Consider moving the synthetic onnx operators to their own
// namespace.

#define FORALL_PRIM_SYMBOLS(_) \
_(prim, Assign) \
_(prim, Constant) \
_(prim, CppOp) \
_(prim, Drop) \
_(prim, Eval) \
_(prim, Expand) /* onnx */ \
_(prim, FusionGroup) \
_(prim, GraphExecutor) \
_(prim, If) \
_(prim, Jump) /* debug */ \
_(prim, JumpNZ) /* debug */ \
_(prim, JumpZ) /* debug */ \
_(prim, Load) \
_(prim, Loop) \
_(prim, Param) \
_(prim, PackPadded) /* onnx */ \
_(prim, PadPacked) /* onnx */ \
_(prim, Placeholder) /* debug */ \
_(prim, Print) \
_(prim, PythonOp) \
_(prim, ReplaceIfUndef) \
_(prim, Reverse) \
_(prim, Return) \
_(prim, Store) \
_(prim, Undefined) \
_(prim, Starred) \
_(prim, TupleConstruct) \
_(prim, TupleUnpack)
/* end */

// Workaround for some not-yet-defined ATen symbols, see
//  - __not__: https://github.com/pytorch/pytorch/issues/5495
//  - ones, zeros: https://github.com/pytorch/pytorch/issues/5496

#define FORALL_ATEN_EXTRA_SYMBOLS(_) \
_(aten, __not__) \
/* end */

#define FORALL_ATEN_SYMBOLS(_) \
FORALL_ATEN_BASE_SYMBOLS(_) \
FORALL_ATEN_EXTRA_SYMBOLS(_)

// These symbols correspond to ONNX operators.  Their semantics
// are defined in https://github.com/onnx/onnx/blob/master/docs/Operators.md
// The particular version we are targeting is specified by '_onnx_opset_version'
// in torch.onnx.symbolic
//
// In general, most ONNX operators won't get an entry here, because they
// are handled from the Python end.  However, you may occasionally need
// to intern an ONNX symbol here so that you can conveniently write an
// optimization on ONNX operations.

#define FORALL_ONNX_SYMBOLS(_) \
_(onnx, Add) \
_(onnx, Concat) \
_(onnx, Constant) \
_(onnx, ConstantFill) \
_(onnx, Div) \
_(onnx, GRU) \
_(onnx, Gather) \
_(onnx, Gemm) \
_(onnx, LSTM) \
_(onnx, Mul) \
_(onnx, Pow) \
_(onnx, RNN) \
_(onnx, Shape) \
_(onnx, Size) \
_(onnx, Slice) \
_(onnx, Squeeze) \
_(onnx, Sub) \
_(onnx, Transpose) \
_(onnx, Unsqueeze) \
/* end */

// These symbols are attribute keys.  They are shared between both ONNX and ATen
// operators (you disambiguate their meaning by looking at the operator itself).
// In general, you only need to define attribute keys that are used by
// onnx or prim; ATen attributes are automatically generated in FORALL_ATTR_BASE_SYMBOLS.

#define FORALL_ATTR_EXTRA_SYMBOLS(_) \
_(attr, Subgraph) \
_(attr, axes) \
_(attr, axis) \
_(attr, broadcast) \
_(attr, device) \
_(attr, direction) \
_(attr, ends) \
_(attr, inplace) \
_(attr, input_as_shape) \
_(attr, is_zero) \
_(attr, perm) \
_(attr, sizes) \
_(attr, starts) \
_(attr, transA) \
_(attr, transB) \
/* end */

#define FORALL_ATTR_SYMBOLS(_) \
FORALL_ATTR_BASE_SYMBOLS(_) \
FORALL_ATTR_EXTRA_SYMBOLS(_)

#define FORALL_BUILTIN_SYMBOLS(_) \
  FORALL_ONNX_SYMBOLS(_) \
  FORALL_ATEN_SYMBOLS(_) \
  FORALL_ATTR_SYMBOLS(_) \
  FORALL_PRIM_SYMBOLS(_) \
  /* end */

// Note [Symbol allocation]
// ~~~~~~~~~~~~~~~~~~~~~~~~
//
//  1. Symbol namespace is split up into namespaces.  The hex structure
//  of our symbols is TTUUUUUU, where TT is the tag byte and U are the unique
//  bytes.
//
//  2. We only maintain a single counter for the unique bytes, which means that
//  we take 256 more space than we would have if we maintained multiple
//  counters.
//
//  3. The first unique_start symbols are reserved for "built-in" symbols.
//  These symbols are allocated at compile time and get put into the intern
//  table at process startup time.  Since it's pretty easy to maintain a
//  distinct counter for every built-in namespace, we let the unique bytes of
//  built-in symbols to overlap (this is why unique_start is a max)
//
//  4. The intended access pattern for built-in symbols is onnx::MatMul
//  in the torch::jit namespace (this is a Symbol).
//
// The code here is not very economical but it gets the job done.


// Built-in constant definition strategy:
// - Enum is the most convenient way to generate a contiguous sequence
//   of numbers for an identifier.
// - However, an enum gives you a fresh type.  We want onnx::MatMul to
//   be type Symbol, not some random enum type!
// - Therefore, after using enums to generate the sequence of integers,
//   we then declare constexpr Symbols to get everything the actual Symbol
//   type we want.  Symbols must be constexpr to be valid to be "case"ed on.

typedef uint32_t unique_t;

constexpr size_t unique_tag_bits = 8;
constexpr size_t unique_bits = sizeof(unique_t) * 8 - unique_tag_bits;
constexpr unique_t unique_mask = (1ULL << unique_bits) - 1;

// A Symbol is like an interned string, but with a little extra
// structure; it is namespaced via SymbolNamespace and the resulting
// intern pointers support efficient namespace testing.
struct Symbol {
  explicit constexpr Symbol() : value(0) {};
  explicit constexpr Symbol(SymbolNamespace ns, uint32_t uniq)
  : value((static_cast<uint32_t>(ns) << unique_bits) | (uniq & unique_mask)) {};

  // Get a Symbol for a qualified string like "attr::bar"
  static Symbol fromQualString(const std::string & s);

  // Constructors for our various namespaced strings.  This will construct
  // the appropriate namespaced string, e.g., "attr::foo" for the
  // argument "foo", and then attempt to intern it.  DO NOT USE THIS
  // with a string literal; attr::foo should be available in that case
  // (and if it's not, you should add it to the built-ins list above.)
  static Symbol attr(const std::string & s) { return Symbol(SymbolNamespace::attr, s); };
  static Symbol aten(const std::string & s) { return Symbol(SymbolNamespace::aten, s); };
  static Symbol onnx(const std::string & s) { return Symbol(SymbolNamespace::onnx, s); };
  static Symbol prim(const std::string & s) { return Symbol(SymbolNamespace::prim, s); };
  // TODO: eliminate me
  static Symbol scope(const std::string & s) { return Symbol(SymbolNamespace::scope, s); };

  constexpr bool is_attr() const { return ns() == SymbolNamespace::attr; };
  constexpr bool is_aten() const { return ns() == SymbolNamespace::aten; };
  constexpr bool is_prim() const { return ns() == SymbolNamespace::prim; };
  constexpr bool is_onnx() const { return ns() == SymbolNamespace::onnx; };

  // So we can switch on this
  constexpr operator unique_t() const {
    return value;
  }

  constexpr SymbolNamespace ns() const {
    return static_cast<SymbolNamespace>(value >> unique_bits);
  }

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

private:
  explicit Symbol(SymbolNamespace ns, const std::string & s);
  unique_t value;
};

static inline bool operator==(Symbol lhs, Symbol rhs) {
  return static_cast<unique_t>(lhs) == static_cast<unique_t>(rhs);
}

#define DEFINE_KEY(ns, s) s,
#define DEFINE_SYMBOL(ns, s) constexpr Symbol s(SymbolNamespace::ns, static_cast<unique_t>(_keys::s));
#define DEFINE_BUILTINS(ns, forall_symbols) \
  namespace ns { \
    enum class _keys : unique_t { \
      forall_symbols(DEFINE_KEY) \
      num_symbols \
    }; \
    forall_symbols(DEFINE_SYMBOL) \
  }

DEFINE_BUILTINS(onnx, FORALL_ONNX_SYMBOLS)
DEFINE_BUILTINS(aten, FORALL_ATEN_SYMBOLS)
DEFINE_BUILTINS(attr, FORALL_ATTR_SYMBOLS)
DEFINE_BUILTINS(prim, FORALL_PRIM_SYMBOLS)

#undef DEFINE_KEY
#undef DEFINE_SYMBOL

}} // namespace torch::jit

// make symbol behave like an integer in hash tables
namespace std {
  template<>
  struct hash<torch::jit::Symbol> {
    std::size_t operator()(torch::jit::Symbol s) const {
      return std::hash<uint32_t>()(static_cast<uint32_t>(s));
    }
  };
}
