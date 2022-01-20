#pragma once

#include <c10/macros/Macros.h>

#if defined(EXPOSE_C2_OPS) || \
    !defined(CAFFE2_IS_XPLAT_BUILD) && !defined(C10_MOBILE)
#include <caffe2/core/tensor.h>
#include <ATen/core/boxing/BoxedKernel.h>
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
  explicit RegisterSchema(const char *schema_str);
};

} // namespace detail
} // namespace caffe2

/**
 * To register a caffe2 operator caffe2::MyOperator with the c10 dispatcher,
 * call:
 *
 * In caffe2/operators/MyOperator.h:
 *
 * > C10_DECLARE_EXPORT_CAFFE2_OP_TO_C10(C10MyOperator) // C10MyOperator is the
 * name
 *                                              // used by c10 for this operator
 *
 * In caffe2/operators/MyOperator.cc
 *
 * > C10_EXPORT_CAFFE2_OP_TO_C10_CPU (
 * >    C10MyOperator,
 * >    "_caffe2::C10MyOperator(Tensor input1, int argument2, float argument3)
 * -> (Tensor output1, Tensor output2)" > caffe2::MyOperator<caffe2::CPUContext>
 * // This is the caffe2 operator >                                           //
 * class template > )
 *
 * In caffe2/operators/MyOperator.cu
 *
 * > C10_EXPORT_CAFFE2_OP_TO_C10_CUDA(C10MyOperator ,
 *   caffe2::MyOperator<caffe2::CUDAContext>)
 *
 * Notes:
 * - all macros must be defined in the top level namespace, not in namespace
 *   caffe2.
 * - all operators must call C10_DECLARE_EXPORT_CAFFE2_OP_TO_C10 and
 *   C10_EXPORT_CAFFE2_OP_TO_C10_CPU .
 * - calling C10_EXPORT_CAFFE2_OP_TO_C10_CUDA is optional and can be omitted i f
 *   you don't want to expose the operator for CUDA operations.
 * - caffe2 arguments must come after caffe2 inputs, in other words, any tensor
 *   inputs must precede any non-tensor inputs.
 *
 * More complex use cases:
 * - If your operator has a variable number of input tensors, make the first (!)
 *   input an input of type TensorList. There must be no other tensor inputs.
 */
#define C10_DECLARE_EXPORT_CAFFE2_OP_TO_C10(OperatorName)

#define C10_EXPORT_CAFFE2_OP_TO_C10_SCHEMA_ONLY(OperatorName, OperatorSchema) \
  /* Register the op schema with the c10 dispatcher */                        \
  static const caffe2::detail::RegisterSchema                                 \
    C10_ANONYMOUS_VARIABLE(RegisterSchema_static_init_)(OperatorSchema);

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
  C10_EXPORT_CAFFE2_OP_TO_C10_SCHEMA_ONLY(OperatorName, OperatorSchema) \
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
#define C10_EXPORT_CAFFE2_OP_TO_C10_SCHEMA_ONLY(OperatorName, OperatorSchema)
#define C10_EXPORT_CAFFE2_OP_TO_C10_CPU_KERNEL_ONLY(OperatorName, OperatorClass)
#define C10_EXPORT_CAFFE2_OP_TO_C10_CPU( \
    OperatorName, OperatorSchema, OperatorClass)
#define C10_EXPORT_CAFFE2_OP_TO_C10_CUDA(OperatorName, OperatorClass)
#define C10_EXPORT_CAFFE2_OP_TO_C10_HIP(OperatorName, OperatorClass)
#endif
