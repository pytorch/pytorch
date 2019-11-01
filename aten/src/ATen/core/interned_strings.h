#pragma once
#include <vector>
#include <cstdint>
#include <string>
#include <unordered_map>
#include <algorithm>

#include <c10/macros/Macros.h>

#if !defined(C10_MOBILE) || defined(FEATURE_TORCH_MOBILE)
#include <ATen/core/aten_interned_strings.h>
#endif

namespace c10 {

#if !defined(C10_MOBILE) || defined(FEATURE_TORCH_MOBILE)
#define FORALL_NS_SYMBOLS(_)         \
  _(namespaces, prim)                \
  _(namespaces, aten)                \
  _(namespaces, onnx)                \
  _(namespaces, attr)                \
  _(namespaces, scope)               \
  _(namespaces, user)                \
  _(namespaces, _caffe2)             \
  _(namespaces, dimname)             \
  _(namespaces, namespaces)          \
  _(prim, Assign)                    \
  _(prim, BroadcastingChunk)         \
  _(prim, BroadcastSizes)            \
  _(prim, Constant)                  \
  _(prim, ChunkSizes)                \
  _(prim, Drop)                      \
  _(prim, Eval)                      \
  _(prim, Expand) /* onnx */         \
  _(prim, FusionGroup)               \
  _(prim, DifferentiableGraph)       \
  _(prim, If)                        \
  _(prim, Jump) /* debug */          \
  _(prim, JumpNZ) /* debug */        \
  _(prim, JumpZ) /* debug */         \
  _(prim, Load)                      \
  _(prim, Loop)                      \
  _(prim, Param)                     \
  _(prim, PackPadded) /* onnx */     \
  _(prim, PadPacked) /* onnx */      \
  _(prim, Placeholder) /* debug */   \
  _(prim, Print)                     \
  _(prim, PythonOp)                  \
  _(prim, IgnoredPythonOp)           \
  _(prim, Reverse)                   \
  _(prim, Return)                    \
  _(prim, ReturnStmt)                \
  _(prim, BreakStmt)                 \
  _(prim, ContinueStmt)              \
  _(prim, Store)                     \
  _(prim, AutogradZero)              \
  _(prim, AutogradAnyNonZero)        \
  _(prim, Starred)                   \
  _(prim, TupleConstruct)            \
  _(prim, TupleUnpack)               \
  _(prim, TupleIndex)                \
  _(prim, TupleSlice)                \
  _(prim, ListConstruct)             \
  _(prim, ListUnpack)                \
  _(prim, DictConstruct)             \
  _(prim, StringIndex)               \
  _(prim, NumToTensor)               \
  _(prim, Uninitialized)             \
  _(prim, ImplicitTensorToNum)       \
  _(aten, Bool)                      \
  _(aten, Int)                       \
  _(aten, Float)                     \
  _(aten, str)                       \
  _(prim, device)                    \
  _(prim, dtype)                     \
  _(prim, shape)                     \
  _(prim, requires_grad)             \
  _(prim, AutogradAdd)               \
  _(prim, GradOf)                    \
  _(aten, grad)                      \
  _(aten, backward)                  \
  _(prim, Guard)                     \
  _(prim, BailOut)                   \
  _(prim, FusedConcat)               \
  _(prim, ConstantChunk)             \
  _(prim, MMTreeReduce)              \
  _(prim, MMBatchSide)               \
  _(prim, min)                       \
  _(prim, max)                       \
  _(prim, abs)                       \
  _(aten, divmod)                    \
  _(prim, zip)                       \
  _(prim, enumerate)                 \
  _(prim, range)                     \
  _(prim, rangelist)                 \
  _(prim, isinstance)                \
  _(prim, unchecked_cast)            \
  _(aten, _grad_sum_to_size)         \
  _(aten, _size_if_not_equal)        \
  _(aten, _ncf_unsqueeze)            \
  _(aten, warn)                      \
  _(aten, sorted)                    \
  _(aten, floordiv)                  \
  _(aten, __range_length)            \
  _(aten, __derive_index)            \
  _(aten, __round_to_zero_floordiv)  \
  _(aten, _unwrap_optional)          \
  _(prim, fork)                      \
  _(prim, forkClosure)               \
  _(prim, RaiseException)            \
  _(prim, Function)                  \
  _(prim, CreateObject)              \
  _(prim, SetAttr)                   \
  _(prim, GetAttr)                   \
  _(prim, profile)                   \
  _(prim, AddStatValue)              \
  _(prim, TimePoint)                 \
  _(prim, CallFunction)              \
  _(prim, CallMethod)                \
  _(prim, LoopContinuation)          \
  _(prim, annotate)                  \
  _(prim, TracedModuleForward)       \
  _(prim, TracedFork)                \
  _(prim, TracedAttr)                \
  _(aten, append)                    \
  _(aten, item)                      \
  _(aten, format)                    \
  _(aten, __not__)                   \
  _(aten, __is__)                    \
  _(aten, __isnot__)                 \
  _(aten, copy)                      \
  _(aten, copy_)                     \
  _(aten, t_)                        \
  _(aten, addbmm_)                   \
  _(aten, addcdiv_)                  \
  _(aten, addcmul_)                  \
  _(aten, addmv_)                    \
  _(aten, addr_)                     \
  _(aten, baddbmm_)                  \
  _(aten, ge_)                       \
  _(aten, gt_)                       \
  _(aten, le_)                       \
  _(aten, lerp_)                     \
  _(aten, lt_)                       \
  _(aten, ne_)                       \
  _(aten, transpose_)                \
  _(aten, unsqueeze_)                \
  _(aten, __getitem__)               \
  _(aten, _set_item)                 \
  _(aten, manual_seed)               \
  _(aten, set_)                      \
  _(aten, index_put_)                \
  _(aten, device)                    \
  _(aten, hash)                      \
  _(aten, len)                       \
  _(aten, list)                      \
  _(aten, wait)                      \
  _(aten, save)                      \
  _(aten, keys)                      \
  _(aten, ord)                       \
  _(aten, chr)                       \
  _(aten, hex)                       \
  _(aten, oct)                       \
  _(aten, clear)                     \
  _(aten, setdefault)                \
  _(aten, bin)                       \
  _(prim, unchecked_unwrap_optional) \
  _(aten, __contains__)              \
  _(prim, BailoutTemplate)           \
  FORALL_ATEN_BASE_SYMBOLS(_)        \
  _(onnx, Add)                       \
  _(onnx, Concat)                    \
  _(onnx, Constant)                  \
  _(onnx, ConstantFill)              \
  _(onnx, Div)                       \
  _(onnx, GRU)                       \
  _(onnx, Gather)                    \
  _(onnx, Gemm)                      \
  _(onnx, LSTM)                      \
  _(onnx, Mul)                       \
  _(onnx, Pow)                       \
  _(onnx, RNN)                       \
  _(onnx, Shape)                     \
  _(onnx, Size)                      \
  _(onnx, Slice)                     \
  _(onnx, Squeeze)                   \
  _(onnx, Sub)                       \
  _(onnx, Transpose)                 \
  _(onnx, Unsqueeze)                 \
  _(onnx, Loop)                      \
  _(onnx, If)                        \
  _(onnx, Reshape)                   \
  _(onnx, Expand)                    \
  _(onnx, Equal)                     \
  _(onnx, Greater)                   \
  _(onnx, Less)                      \
  _(onnx, Not)                       \
  _(onnx, ATen)                      \
  _(onnx, Split)                     \
  _(onnx, ConstantOfShape)           \
  _(onnx, Cast)                      \
  _(onnx, Mod)                       \
  FORALL_ATTR_BASE_SYMBOLS(_)        \
  _(attr, Subgraph)                  \
  _(attr, ReverseSubgraph)           \
  _(attr, f_real_outputs)            \
  _(attr, df_input_vjps)             \
  _(attr, df_input_captured_inputs)  \
  _(attr, df_input_captured_outputs) \
  _(attr, df_output_vjps)            \
  _(attr, axes)                      \
  _(attr, axis)                      \
  _(attr, broadcast)                 \
  _(attr, direction)                 \
  _(attr, ends)                      \
  _(attr, inplace)                   \
  _(attr, input_as_shape)            \
  _(attr, is_zero)                   \
  _(attr, perm)                      \
  _(attr, sizes)                     \
  _(attr, starts)                    \
  _(attr, transA)                    \
  _(attr, transB)                    \
  _(attr, name)                      \
  _(attr, a)                         \
  _(attr, b)                         \
  _(attr, beg)                       \
  _(attr, idx)                       \
  _(attr, split)                     \
  _(attr, slot)                      \
  _(attr, kinds)                     \
  _(attr, types)                     \
  _(attr, scope)
#else
#define FORALL_NS_SYMBOLS(_) \
  _(namespaces, prim)              \
  _(namespaces, aten)              \
  _(namespaces, onnx)              \
  _(namespaces, attr)              \
  _(namespaces, scope)             \
  _(namespaces, user)              \
  _(namespaces, _caffe2)           \
  _(namespaces, dimname)           \
  _(namespaces, namespaces)
#endif

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
struct CAFFE2_API Symbol {
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
  static Symbol onnx(const std::string & s);
  static Symbol prim(const std::string & s);
  static Symbol user(const std::string & s);
  static Symbol caffe2(const std::string & s);
  static Symbol dimname(const std::string & s);
  // TODO: eliminate me
  static Symbol scope(const std::string & s);

  bool is_attr() const;
  bool is_aten() const;
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

enum class _keys : unique_t {
    #define DEFINE_KEY(ns, s) ns##_##s,
    FORALL_NS_SYMBOLS(DEFINE_KEY)
    #undef DEFINE_KEY
    num_symbols
};

#define DEFINE_SYMBOL(s) \
  constexpr Symbol s(static_cast<unique_t>(_keys::s));

#undef DEFINE_SYMBOL

#define DEFINE_SYMBOL(ns, s) \
  namespace ns { constexpr Symbol s(static_cast<unique_t>(_keys::ns##_##s)); }
FORALL_NS_SYMBOLS(DEFINE_SYMBOL)
#undef DEFINE_SYMBOL

inline Symbol Symbol::attr(const std::string & s) { return Symbol::fromQualString("attr::" + s); }
inline Symbol Symbol::aten(const std::string & s)  { return Symbol::fromQualString("aten::" + s); }
inline Symbol Symbol::onnx(const std::string & s)  { return Symbol::fromQualString("onnx::" + s); }
inline Symbol Symbol::prim(const std::string & s)  { return Symbol::fromQualString("prim::" + s); }
inline Symbol Symbol::scope(const std::string & s) { return Symbol::fromQualString("scope::" + s); }
inline Symbol Symbol::user(const std::string & s) { return Symbol::fromQualString("user::" + s); }
inline Symbol Symbol::caffe2(const std::string & s) { return Symbol::fromQualString("_caffe2::" + s); }
inline Symbol Symbol::dimname(const std::string & s) { return Symbol::fromQualString("dimname::" + s); }
inline bool Symbol::is_attr() const { return ns() == namespaces::attr; }
inline bool Symbol::is_aten() const { return ns() == namespaces::aten; }
inline bool Symbol::is_prim() const { return ns() == namespaces::prim; }
inline bool Symbol::is_onnx() const { return ns() == namespaces::onnx; }
inline bool Symbol::is_user() const { return ns() == namespaces::user; }
inline bool Symbol::is_caffe2() const { return ns() == namespaces::_caffe2; }
inline bool Symbol::is_dimname() const { return ns() == namespaces::dimname; }

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
