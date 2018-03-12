#pragma once
#include <vector>
#include <stdint.h>
#include <string>
#include <unordered_map>

#include "torch/csrc/jit/generated/aten_interned_strings.h"

namespace torch { namespace jit {

// JIT symbols are synthetic operators that occur only in the JIT IR
// and don't have corresponding implementations in ATen.
//
// TODO: We need documentation for all of these symbols.

#define FORALL_JIT_SYMBOLS(_) \
_(Assign) \
_(Constant) \
_(CppOp) \
_(Drop) \
_(Eval) \
_(Expand) \
_(FusionGroup) \
_(GraphExecutor) \
_(If) \
_(Jump) \
_(JumpNZ) \
_(JumpZ) \
_(Load) \
_(Loop) \
_(Param) \
_(Placeholder) \
_(Print) \
_(PythonOp) \
_(ReplaceIfUndef) \
_(Reverse) \
_(Return) \
_(Store) \
_(Undefined) \
_(__JIT_END)

// Workaround for some not-yet-defined ATen symbols, see
//  - __not__: https://github.com/pytorch/pytorch/issues/5495
//  - ones, zeros: https://github.com/pytorch/pytorch/issues/5496

#define FORALL_ATEN_EXTRA_SYMBOLS(_) \
_(__not__) \
_(ones) \
_(zeros) \
_(__ATEN_EXTRA_END)

// These symbols correspond to ONNX operators.  Their semantics
// are defined in https://github.com/onnx/onnx/blob/master/docs/Operators.md
// The particular version we are targeting is specified by '_onnx_opset_version'
// in torch.onnx.symbolic

#define FORALL_ONNX_SYMBOLS(_) \
_(Add) \
/* _(Constant) conflicts with JIT */ \
_(Div) \
_(GRU) \
_(Gemm) \
_(LSTM) \
_(Mul) \
_(PackPadded) \
_(PadPacked) \
_(Pow) \
_(RNN) \
_(Slice) /* used by test only */ \
_(Sub) \
_(Transpose) \
_(__ONNX_END)

// These symbols are attribute keys.  They are shared between both ONNX and ATen
// operators (you disambiguate their meaning by looking at the operator itself)

#define FORALL_ATTR_SYMBOLS(_) \
_(Subgraph) \
_(alpha) \
_(axis) \
_(broadcast) \
_(device) \
/* _(dim) conflicts with ATen */ \
_(end) \
_(exponent) \
_(inplace) \
_(is_zero) \
_(keepdim) \
_(length) \
_(other) \
_(perm) \
/* _(size) conflicts with ATen */ \
/* _(sizes) conflicts with ATen */ \
_(start) \
_(step) \
_(transA) \
_(transB) \
_(value) \
_(__ATTR_END)

#define FORALL_BUILTIN_SYMBOLS(_) \
FORALL_JIT_SYMBOLS(_) \
FORALL_ATEN_SYMBOLS(_) \
FORALL_ATEN_EXTRA_SYMBOLS(_) \
FORALL_ONNX_SYMBOLS(_) \
FORALL_ATTR_SYMBOLS(_) \
_(__BUILTIN_END)

  enum BuiltinSymbol {
    #define DEFINE_SYMBOL(s) \
      k##s,
    FORALL_BUILTIN_SYMBOLS(DEFINE_SYMBOL)
    #undef DEFINE_SYMBOL
    kLastSymbol, //where we start counting for new symbols
  };


struct Symbol {
  Symbol() {}
  /*implicit*/ Symbol(BuiltinSymbol value)
  : value(value) {}
  explicit Symbol(uint32_t value)
  : value(value) {}

  // Parse a string in the toQualString format to Symbol.  This is a proper
  // parser, so avoid using it in a hotpath.
  static Symbol parseQualString(const std::string & s);

  // Constructors for our various namespaced strings.  ATM these don't do
  // anything but they will soon.
  static Symbol attr(const std::string & s) { return Symbol(s); };
  static Symbol aten(const std::string & s) { return Symbol(s); };
  static Symbol onnx(const std::string & s) { return Symbol(s); };
  static Symbol jit(const std::string & s) { return Symbol(s); };
  // TODO: eliminate me
  static Symbol scope(const std::string & s) { return Symbol(s); };

  operator uint32_t() const {
    return value;
  }

  // Give a string corresponding to the unqualified version of this name, e.g.,
  // "mm". Use this in a context where the intended namespace of the string is
  // obvious; this is a *lossy* conversion.
  const char * toUnqualString() const;

  // Give a string corresponding to the qualified version of this name,
  // e.g., "aten::mm".  This string format is made available to Python bindings
  // (so we know how to parse it.)  THIS FUNCTION DOES AN ALLOCATION, DON'T USE
  // IT IN A TIGHT LOOP.
  std::string toQualString() const;

  // Give an unambiguous, machine readable description of the name.  This
  // is strictly an implementation, but it will look something like "tmm"
  const char * toRawString() const;

  // This describes a symbol in human readable form.  DO NOT use
  // this for uses-cases that are intended to be machine-readable.
  // This will look something like "ATen operator 'mm'"
  const char * toDisplayString() const;

private:
  explicit Symbol(const std::string & s);
  uint32_t value;
};

static inline bool operator==(Symbol lhs, Symbol rhs) {
  return static_cast<uint32_t>(lhs) == static_cast<uint32_t>(rhs);
}
// necessary to prevent ambiguous overload resolutions
static inline bool operator==(BuiltinSymbol lhs, Symbol rhs) {
  return static_cast<uint32_t>(lhs) == static_cast<uint32_t>(rhs);
}
static inline bool operator==(Symbol lhs, BuiltinSymbol rhs) {
  return static_cast<uint32_t>(lhs) == static_cast<uint32_t>(rhs);
}

}}

// make symbol behave like an integer in hash tables
namespace std {
  template<>
  struct hash<torch::jit::Symbol> {
    std::size_t operator()(torch::jit::Symbol s) const {
      return std::hash<uint32_t>()(static_cast<uint32_t>(s));
    }
  };
}
