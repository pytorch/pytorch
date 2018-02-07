#pragma once
#include <vector>
#include <stdint.h>
#include <string>
#include <unordered_map>

namespace torch { namespace jit {


#define FORALL_BUILTIN_SYMBOLS(_) \
_(PythonOp) \
_(CppOp) \
_(Param) \
_(Select) \
_(Return) \
_(Eval) \
_(add) \
_(Add) \
_(Div) \
_(Mul) \
_(Neg) \
_(Sub) \
_(Pow) \
_(Sigmoid) \
_(Tanh) \
_(mul) \
_(neg) \
_(sigmoid) \
_(tanh) \
_(Constant) \
_(cat) \
_(Slice) \
_(Squeeze) \
_(Undefined) \
_(FusionGroup) \
_(Gemm) \
_(SubConstant) \
_(Scale) \
_(Transpose) \
_(Reshape) \
_(split) \
_(chunk) \
_(Offset) \
_(value) \
_(Subgraph) \
_(BatchNormalization) \
_(Conv) \
_(PackPadded) \
_(PadPacked) \
_(ConvTranspose) \
_(is_test) \
_(epsilon) \
_(expand) \
_(Expand) \
_(order) \
_(momentum) \
_(consumed_inputs) \
_(kernels) \
_(kernel_shape) \
_(kernel) \
_(scale) \
_(strides) \
_(stride) \
_(pads) \
_(pad) \
_(RNN) \
_(LSTM) \
_(GRU) \
_(beta) \
_(alpha) \
_(dilations) \
_(dilation) \
_(broadcast) \
_(axis) \
_(size) \
_(dim) \
_(perm) \
_(shape) \
_(axes) \
_(group) \
_(inplace) \
_(transA) \
_(transB) \
_(other) \
_(__and__) \
_(__lshift__) \
_(__or__) \
_(__rshift__) \
_(__xor__) \
_(abs) \
_(acos) \
_(asin) \
_(atan) \
_(atan2) \
_(ceil) \
_(clamp) \
_(cos) \
_(cosh) \
_(div) \
_(eq) \
_(equal) \
_(exp) \
_(expm1) \
_(floor) \
_(fmod) \
_(frac) \
_(ge) \
_(gt) \
_(le) \
_(lerp) \
_(lgamma) \
_(log) \
_(log1p) \
_(lt) \
_(max) \
_(min) \
_(ne) \
_(ones) \
_(pow) \
_(reciprocal) \
_(remainder) \
_(round) \
_(rsqrt) \
_(sin) \
_(sinh) \
_(sqrt) \
_(sub) \
_(tan) \
_(trunc) \
_(zeros) \
_(exponent) \
_(device) \
_(ReplaceIfUndef) \
_(is_zero) \
_(GraphExecutor) \
_(mm) \
_(t)

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
  explicit Symbol(const std::string & s);
  explicit Symbol(uint32_t value)
  : value(value) {}

  operator uint32_t() const {
    return value;
  }
  const char * toString() const;
private:
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

inline Symbol operator "" _sym(const char * s, size_t) {
  return Symbol(s);
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
