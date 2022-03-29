#pragma once

#include <c10/macros/Macros.h>

#if defined(EXPOSE_C2_OPS) || \
    !defined(CAFFE2_IS_XPLAT_BUILD) && !defined(C10_MOBILE)
#include <ATen/core/dispatch/OperatorOptions.h>
#include <ATen/core/boxing/BoxedKernel.h>
#include <caffe2/core/tensor.h>
#include <vector>

namespace c10 {
struct FunctionSchema;
struct IValue;
class OperatorHandle;
using Stack = std::vector<IValue>;
}

namespace caffe2 {
namespace detail {

constexpr const char* PREALLOCATED_OUTPUT_ARGNAME =
    "_caffe2_preallocated_outputs";

using _CallCaffe2OpFunc = std::vector<caffe2::Tensor>(
    const c10::FunctionSchema& schema,
    c10::ArrayRef<c10::IValue> inputs,
    std::vector<caffe2::Tensor> &&outputs);

template <class Caffe2Operator>
inline std::vector<caffe2::Tensor> _call_caffe2_op(
    const c10::FunctionSchema& schema,
    c10::ArrayRef<c10::IValue> inputs,
    std::vector<caffe2::Tensor> &&outputs) {
  Caffe2Operator op(schema, inputs, std::move(outputs), -1);
  op.Run(-1);
  return std::move(op).move_output_tensors();
}

TORCH_API void call_caffe2_op_from_c10(
    const c10::OperatorHandle &opHandle,
    c10::Stack *stack,
    _CallCaffe2OpFunc *call_op);

template <typename Caffe2Operator>
void boxed_caffe2_operator(const OperatorHandle& opHandle, c10::Stack* stack) {
  call_caffe2_op_from_c10(
      opHandle,
      stack,
      &_call_caffe2_op<Caffe2Operator>);
}

template <c10::DispatchKey key>
struct TORCH_API RegisterDefinition {
  RegisterDefinition(const char *name, c10::BoxedKernel kernel);
};

extern template struct RegisterDefinition<c10::DispatchKey::CPU>;
extern template struct RegisterDefinition<c10::DispatchKey::CUDA>;
extern template struct RegisterDefinition<c10::DispatchKey::HIP>;

struct TORCH_API RegisterSchema {
  RegisterSchema(
    const char *schema_str,
    c10::optional<c10::AliasAnalysisKind> optional_alias_analysis_kind);
};

} // namespace detail
} // namespace caffe2

#define C10_DECLARE_EXPORT_CAFFE2_OP_TO_C10(OperatorName)

#define C10_EXPORT_CAFFE2_OP_TO_C10_SCHEMA_ONLY(                        \
    OperatorName, OperatorSchema, OptionalAliasAnalysisKind)            \
  /* Register the op schema with the c10 dispatcher */                  \
  static const caffe2::detail::RegisterSchema                           \
  C10_ANONYMOUS_VARIABLE(RegisterSchema_static_init_)(                  \
    OperatorSchema, OptionalAliasAnalysisKind);

#define _C10_EXPORT_CAFFE2_OP_TO_C10_KEY(                                     \
    OperatorName, OperatorClass, Key)                                         \
  /* Register call_caffe2_op_from_c10 as a kernel with the c10 dispatcher */  \
  static const caffe2::detail::RegisterDefinition<c10::DispatchKey::Key>      \
    C10_ANONYMOUS_VARIABLE(Register##Key##Definition_static_init_)(           \
        "_caffe2::" #OperatorName,                                            \
        c10::BoxedKernel::makeFromFunction<                                   \
            &::caffe2::detail::boxed_caffe2_operator<OperatorClass>>());

#define C10_EXPORT_CAFFE2_OP_TO_C10_CPU_KERNEL_ONLY(                    \
    OperatorName, OperatorClass)                                        \
  _C10_EXPORT_CAFFE2_OP_TO_C10_KEY(OperatorName, OperatorClass, CPU)

#define C10_EXPORT_CAFFE2_OP_TO_C10_CPU(                                \
    OperatorName, OperatorSchema, OperatorClass)                        \
  C10_EXPORT_CAFFE2_OP_TO_C10_SCHEMA_ONLY(                              \
    OperatorName, OperatorSchema, c10::nullopt)                         \
  C10_EXPORT_CAFFE2_OP_TO_C10_CPU_KERNEL_ONLY(OperatorName, OperatorClass)

#define C10_EXPORT_CAFFE2_OP_TO_C10_CUDA(OperatorName, OperatorClass)   \
  _C10_EXPORT_CAFFE2_OP_TO_C10_KEY(OperatorName, OperatorClass, CUDA)


// You should never manually call the C10_EXPORT_CAFFE2_OP_TO_C10_HIP macro .
// The C10_EXPORT_CAFFE2_OP_TO_C10_CUDA macro from above will be automatically
// rewritten to C10_EXPORT_CAFFE2_OP_TO_C10_HIP by hipify .
#define C10_EXPORT_CAFFE2_OP_TO_C10_HIP(OperatorName, OperatorClass)    \
  _C10_EXPORT_CAFFE2_OP_TO_C10_KEY(OperatorName, OperatorClass, HIP)


#else
// Don't use c10 dispatcher on mobile because of binary size
#define C10_DECLARE_EXPORT_CAFFE2_OP_TO_C10(OperatorName)
#define C10_EXPORT_CAFFE2_OP_TO_C10_SCHEMA_ONLY( \
    OperatorName, OperatorSchema, OptionalAliasAnalysisKind)
#define C10_EXPORT_CAFFE2_OP_TO_C10_CPU_KERNEL_ONLY(OperatorName, OperatorClass)
#define C10_EXPORT_CAFFE2_OP_TO_C10_CPU( \
    OperatorName, OperatorSchema, OperatorClass)
#define C10_EXPORT_CAFFE2_OP_TO_C10_CUDA(OperatorName, OperatorClass)
#define C10_EXPORT_CAFFE2_OP_TO_C10_HIP(OperatorName, OperatorClass)
#endif
