#pragma once
#include <vector>
#include <cstdint>
#include <string>
#include <unordered_map>
#include <algorithm>

#include <c10/macros/Macros.h>

#include <ATen/core/aten_interned_strings.h>

namespace c10 {

#define FORALL_NS_SYMBOLS(_)         \
  _(namespaces, prim)                \
  _(namespaces, aten)                \
  _(namespaces, cuda)                \
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
  _(prim, ReductionSizes)            \
  _(prim, Constant)                  \
  _(prim, ChunkSizes)                \
  _(prim, ConstantMKLDNNTensor)      \
  _(prim, BroadcastMKLDNNTensors)    \
  _(prim, MKLDNNGroup)               \
  _(prim, MKLDNNHardSwish)           \
  _(prim, MKLDNNHardSigmoid)         \
  _(prim, MKLDNNHardTanh)            \
  _(prim, MKLDNNClamp)               \
  _(prim, Drop)                      \
  _(prim, Eval)                      \
  _(prim, Expand) /* onnx */         \
  _(prim, FusionGroup)               \
  _(prim, CudaFusionGroup)           \
  _(prim, CudaFusionGuard)           \
  _(prim, FunctionalGraph)           \
  _(prim, DifferentiableGraph)       \
  _(prim, TensorExprGroup)           \
  _(prim, StaticSubgraph)            \
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
  _(prim, ComprehensionScope)        \
  _(prim, Store)                     \
  _(prim, AutogradZero)              \
  _(prim, AutogradAnyNonZero)        \
  _(prim, AutogradAllNonZero)        \
  _(prim, AutogradAllZero)           \
  _(prim, Starred)                   \
  _(prim, TupleConstruct)            \
  _(prim, TupleUnpack)               \
  _(prim, TupleIndex)                \
  _(prim, TupleSlice)                \
  _(prim, ListConstruct)             \
  _(prim, ListUnpack)                \
  _(prim, DictConstruct)             \
  _(prim, ModuleContainerIndex)      \
  _(prim, EnumName)                  \
  _(prim, EnumValue)                 \
  _(prim, StringIndex)               \
  _(prim, NumToTensor)               \
  _(prim, Uninitialized)             \
  _(prim, With)                      \
  _(prim, Enter)                     \
  _(prim, Exit)                      \
  _(aten, Bool)                      \
  _(aten, Int)                       \
  _(aten, FloatImplicit)             \
  _(aten, ComplexImplicit)           \
  _(aten, IntImplicit)               \
  _(aten, ScalarImplicit)            \
  _(aten, Float)                     \
  _(aten, Complex)                   \
  _(aten, str)                       \
  _(aten, is_pinned)                 \
  _(aten, Delete)                    \
  _(aten, relu_)                     \
  _(aten, gelu_)                     \
  _(aten, relu6)                     \
  _(aten, relu6_)                    \
  _(aten, dropout_)                  \
  _(aten, sigmoid_)                  \
  _(prim, device)                    \
  _(prim, dtype)                     \
  _(prim, layout)                    \
  _(prim, id)                        \
  _(prim, requires_grad)             \
  _(prim, MakeTestTensor) /* test */ \
  _(prim, AutogradAdd)               \
  _(prim, GradOf)                    \
  _(aten, grad)                      \
  _(aten, backward)                  \
  _(prim, Guard)                     \
  _(prim, BailOut)                   \
  _(prim, TypeCheck)                 \
  _(prim, RequiresGradCheck)         \
  _(prim, FallbackGraph)             \
  _(prim, FusedConcat)               \
  _(prim, ConstantChunk)             \
  _(prim, MMTreeReduce)              \
  _(prim, MMBatchSide)               \
  _(prim, list)                      \
  _(prim, dict)                      \
  _(prim, min)                       \
  _(prim, max)                       \
  _(prim, abs)                       \
  _(aten, divmod)                    \
  _(prim, zip)                       \
  _(prim, enumerate)                 \
  _(prim, range)                     \
  _(prim, rangelist)                 \
  _(prim, isinstance)                \
  _(prim, tolist)                    \
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
  _(aten, is_scripting)              \
  _(aten, _unwrap_optional)          \
  _(prim, fork)                      \
  _(prim, forkClosure)               \
  _(prim, RaiseException)            \
  _(prim, Closure)                   \
  _(prim, CreateObject)              \
  _(prim, SetAttr)                   \
  _(prim, GetAttr)                   \
  _(prim, HasAttr)                   \
  _(prim, profile)                   \
  _(prim, profile_ivalue)            \
  _(prim, AddStatValue)              \
  _(prim, TimePoint)                 \
  _(prim, CallFunction)              \
  _(prim, CallMethod)                \
  _(prim, LoopContinuation)          \
  _(prim, annotate)                  \
  _(prim, TracedModuleForward)       \
  _(prim, TracedFork)                \
  _(prim, TracedAttr)                \
  _(prim, rpc_async)                 \
  _(prim, rpc_sync)                  \
  _(prim, rpc_remote)                \
  _(prim, is_cuda)                   \
  _(aten, abs_)                      \
  _(aten, absolute)                  \
  _(aten, absolute_)                 \
  _(aten, acos)                      \
  _(aten, acos_)                     \
  _(aten, arccos)                    \
  _(aten, arccos_)                   \
  _(aten, acosh)                     \
  _(aten, acosh_)                    \
  _(aten, arccosh)                   \
  _(aten, arccosh_)                  \
  _(aten, asin)                      \
  _(aten, asin_)                     \
  _(aten, arcsin)                    \
  _(aten, arcsin_)                   \
  _(aten, asinh)                     \
  _(aten, asinh_)                    \
  _(aten, arcsinh)                   \
  _(aten, arcsinh_)                  \
  _(aten, atan)                      \
  _(aten, atan_)                     \
  _(aten, arctan)                    \
  _(aten, arctan_)                   \
  _(aten, atanh)                     \
  _(aten, atanh_)                    \
  _(aten, arctanh)                   \
  _(aten, arctanh_)                  \
  _(aten, clamp)                     \
  _(aten, clamp_)                    \
  _(aten, clip)                      \
  _(aten, clip_)                     \
  _(aten, det)                       \
  _(aten, linalg_det)                \
  _(aten, matrix_power)              \
  _(aten, linalg_matrix_power)       \
  _(aten, chain_matmul)              \
  _(aten, linalg_multi_dot)          \
  _(aten, linalg_norm)               \
  _(aten, linalg_vector_norm)        \
  _(aten, linalg_matrix_norm)        \
  _(aten, append)                    \
  _(aten, item)                      \
  _(aten, format)                    \
  _(aten, percentFormat)             \
  _(aten, __not__)                   \
  _(aten, __is__)                    \
  _(aten, __isnot__)                 \
  _(aten, copy)                      \
  _(aten, copy_)                     \
  _(aten, div)                       \
  _(aten, div_)                      \
  _(aten, divide)                    \
  _(aten, divide_)                   \
  _(aten, true_divide)               \
  _(aten, true_divide_)              \
  _(aten, t_)                        \
  _(aten, addbmm_)                   \
  _(aten, addcdiv_)                  \
  _(aten, addcmul_)                  \
  _(aten, addmv_)                    \
  _(aten, addr_)                     \
  _(aten, baddbmm_)                  \
  _(aten, ge)                        \
  _(aten, ge_)                       \
  _(aten, greater_equal)             \
  _(aten, greater_equal_)            \
  _(aten, gt)                        \
  _(aten, gt_)                       \
  _(aten, greater)                   \
  _(aten, greater_)                  \
  _(aten, le)                        \
  _(aten, le_)                       \
  _(aten, less_equal)                \
  _(aten, less_equal_)               \
  _(aten, lerp_)                     \
  _(aten, lt)                        \
  _(aten, lt_)                       \
  _(aten, less)                      \
  _(aten, less_)                     \
  _(aten, isnan)                     \
  _(aten, mul)                       \
  _(aten, mul_)                      \
  _(aten, multiply)                  \
  _(aten, multiply_)                 \
  _(aten, ne)                        \
  _(aten, ne_)                       \
  _(aten, not_equal)                 \
  _(aten, not_equal_)                \
  _(aten, _ger)                      \
  _(aten, ger)                       \
  _(aten, outer)                     \
  _(aten, orgqr)                     \
  _(aten, linalg_householder_product)\
  _(aten, transpose)                 \
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
  _(aten, dict)                      \
  _(aten, wait)                      \
  _(aten, save)                      \
  _(aten, sub)                       \
  _(aten, sub_)                      \
  _(aten, subtract)                  \
  _(aten, subtract_)                 \
  _(aten, keys)                      \
  _(aten, ord)                       \
  _(aten, chr)                       \
  _(aten, hex)                       \
  _(aten, oct)                       \
  _(aten, clear)                     \
  _(aten, trunc)                     \
  _(aten, trunc_)                    \
  _(aten, fix)                       \
  _(aten, fix_)                      \
  _(aten, to_mkldnn)                 \
  _(aten, positive)                  \
  _(aten, neg)                       \
  _(aten, neg_)                      \
  _(aten, negative)                  \
  _(aten, negative_)                 \
  _(aten, setdefault)                \
  _(aten, bin)                       \
  _(aten, pop)                       \
  _(aten, insert)                    \
  _(aten, vstack)                    \
  _(aten, row_stack)                 \
  _(prim, unchecked_unwrap_optional) \
  _(aten, __contains__)              \
  _(prim, BailoutTemplate)           \
  _(prim, grad)                      \
  _(aten, zero_)                     \
  _(aten, fill_)                     \
  _(aten, masked_fill_)              \
  _(cuda, _set_device)               \
  _(cuda, set_stream)                \
  _(cuda, _current_device)           \
  _(cuda, synchronize)               \
  _(aten, swapaxes)                  \
  _(aten, swapaxes_)                 \
  _(aten, swapdims)                  \
  _(aten, swapdims_)                 \
  _(aten, movedim)                   \
  _(aten, moveaxis)                  \
  _(aten, lgamma)                    \
  _(aten, special_gammaln)           \
  _(aten, erf)                       \
  _(aten, special_erf)               \
  _(aten, erfc)                      \
  _(aten, special_erfc)              \
  _(aten, erfinv)                    \
  _(aten, special_erfinv)            \
  _(aten, logit)                     \
  _(aten, special_logit)             \
  _(aten, sigmoid)                   \
  _(aten, special_expit)             \
  _(aten, expm1)                     \
  _(aten, special_expm1)             \
  _(aten, exp2)                      \
  _(aten, special_exp2)              \
  _(aten, special_i0e)               \
  _(aten, has_torch_function)        \
  _(aten, hardswish)                 \
  _(aten, hardswish_)                \
  _(aten, hardsigmoid_)              \
  _(aten, hardtanh_)                 \
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
  _(onnx, GreaterOrEqual)            \
  _(onnx, Less)                      \
  _(onnx, LessOrEqual)               \
  _(onnx, Not)                       \
  _(onnx, ATen)                      \
  _(onnx, Split)                     \
  _(onnx, ConstantOfShape)           \
  _(onnx, Cast)                      \
  _(onnx, Mod)                       \
  _(onnx, Sqrt)                      \
  _(onnx, SplitToSequence)           \
  _(onnx, SequenceAt)                \
  _(onnx, SequenceConstruct)         \
  _(onnx, SequenceEmpty)             \
  _(onnx, SequenceInsert)            \
  _(onnx, SequenceErase)             \
  _(onnx, ConcatFromSequence)        \
  _(onnx, Identity)                  \
  _(onnx, SoftmaxCrossEntropyLoss)   \
  _(onnx, NegativeLogLikelihoodLoss) \
  _(onnx, LogSoftmax)                \
  _(onnx, ReduceL1)                  \
  _(onnx, ReduceL2)                  \
  _(onnx, Conv)                      \
  _(onnx, BatchNormalization)        \
  _(onnx, ReduceProd)                \
  _(onnx, Neg)                       \
  _(onnx, NonZero)                   \
  _(onnx, Range)                     \
  _(onnx, Tile)                      \
  _(onnx, Where)                     \
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
  _(attr, num_none)                  \
  _(attr, num_present)               \
  _(attr, perm)                      \
  _(attr, sizes)                     \
  _(attr, starts)                    \
  _(attr, profiled_type)             \
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
  _(attr, scope)                     \
  _(attr, keepdims)                  \
  _(attr, cache_id)                  \
  _(attr, new_axis)                  \
  _(attr, warn_id)

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
inline Symbol Symbol::cuda(const std::string & s)  { return Symbol::fromQualString("cuda::" + s); }
inline Symbol Symbol::onnx(const std::string & s)  { return Symbol::fromQualString("onnx::" + s); }
inline Symbol Symbol::prim(const std::string & s)  { return Symbol::fromQualString("prim::" + s); }
inline Symbol Symbol::scope(const std::string & s) { return Symbol::fromQualString("scope::" + s); }
inline Symbol Symbol::user(const std::string & s) { return Symbol::fromQualString("user::" + s); }
inline Symbol Symbol::caffe2(const std::string & s) { return Symbol::fromQualString("_caffe2::" + s); }
inline Symbol Symbol::dimname(const std::string & s) { return Symbol::fromQualString("dimname::" + s); }
inline bool Symbol::is_attr() const { return ns() == namespaces::attr; }
inline bool Symbol::is_aten() const { return ns() == namespaces::aten; }
inline bool Symbol::is_cuda() const { return ns() == namespaces::cuda; }
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
