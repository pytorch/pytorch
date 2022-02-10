#pragma once
#include <vector>
#include <cstdint>
#include <string>
#include <unordered_map>
#include <algorithm>

#include <c10/macros/Macros.h>

#include <ATen/core/aten_interned_strings.h>
#include <ATen/core/symbol.h>

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
  _(prim, StaticRuntimeCopyOuts)     \
  _(prim, Drop)                      \
  _(prim, Eval)                      \
  _(prim, Expand) /* onnx */         \
  _(prim, FusionGroup)               \
  _(prim, CudaFusionGroup)           \
  _(prim, CudaFusionGuard)           \
  _(prim, FunctionalGraph)           \
  _(prim, add_optional)              \
  _(prim, DifferentiableGraph)       \
  _(prim, TensorExprGroup)           \
  _(prim, TensorExprDynamicGroup)    \
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
  _(prim, VarConcat)                 \
  _(prim, VarStack)                  \
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
  _(aten, Delete)                    \
  _(aten, gelu_)                     \
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
  _(aten, append)                    \
  _(aten, as_tensor)                 \
  _(aten, adaptive_avg_pool2d_backward) \
  _(aten, dim)                       \
  _(aten, format)                    \
  _(aten, percentFormat)             \
  _(aten, __not__)                   \
  _(aten, __is__)                    \
  _(aten, __isnot__)                 \
  _(aten, _ger)                      \
  _(aten, __getitem__)               \
  _(aten, _set_item)                 \
  _(aten, manual_seed)               \
  _(aten, device)                    \
  _(aten, hash)                      \
  _(aten, len)                       \
  _(aten, list)                      \
  _(aten, dict)                      \
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
  _(aten, pop)                       \
  _(aten, insert)                    \
  _(aten, tensor)                    \
  _(prim, unchecked_unwrap_optional) \
  _(aten, __contains__)              \
  _(prim, BailoutTemplate)           \
  _(prim, grad)                      \
  _(cuda, _set_device)               \
  _(cuda, set_stream)                \
  _(cuda, _current_device)           \
  _(cuda, synchronize)               \
  _(aten, has_torch_function)        \
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
  _(onnx, MatMul)                    \
  _(onnx, Mul)                       \
  _(onnx, Pow)                       \
  _(onnx, RNN)                       \
  _(onnx, Shape)                     \
  _(onnx, Size)                      \
  _(onnx, Slice)                     \
  _(onnx, Softmax)                   \
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
  _(onnx, ReduceMean)                \
  _(onnx, ReduceProd)                \
  _(onnx, Relu)                      \
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
  _(attr, symbolic_shape_inputs)     \
  _(attr, allow_stack_outputs)       \
  _(attr, striding_inputs_desc)      \
  _(attr, striding_outputs_desc)     \
  _(attr, broadcast)                 \
  _(attr, direction)                 \
  _(attr, ends)                      \
  _(attr, inplace)                   \
  _(attr, input_as_shape)            \
  _(attr, is_zero)                   \
  _(attr, num_none)                  \
  _(attr, num_present)               \
  _(attr, perm)                      \
  _(attr, starts)                    \
  _(attr, profiled_type)             \
  _(attr, transA)                    \
  _(attr, transB)                    \
  _(attr, name)                      \
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
  _(attr, warn_id)                   \
  _(attr, allowzero)                 \
  _(attr, seen_none)

enum class _keys : unique_t {
    #define DEFINE_KEY(ns, s) ns##_##s,
    FORALL_NS_SYMBOLS(DEFINE_KEY)
    #undef DEFINE_KEY
    num_symbols
};

#define DEFINE_SYMBOL(ns, s) \
  namespace ns { constexpr Symbol s(static_cast<unique_t>(_keys::ns##_##s)); }
FORALL_NS_SYMBOLS(DEFINE_SYMBOL)
#undef DEFINE_SYMBOL

} // namespace c10
