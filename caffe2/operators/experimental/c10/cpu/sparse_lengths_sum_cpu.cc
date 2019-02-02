#include <ATen/core/dispatch/KernelRegistration.h>
#include "caffe2/operators/experimental/c10/schemas/sparse_lengths_sum.h"
#include "caffe2/perfkernels/embedding_lookup.h"
#include "caffe2/utils/math.h"
#include "caffe2/core/tensor.h"

using caffe2::Tensor;

namespace caffe2 {
namespace {

template <typename InputType, typename IndexType>
void sparse_lengths_sum_op_cpu_impl_(
    const at::Tensor& dataInput_,
    const at::Tensor& indicesInput_,
    const at::Tensor& lengthsInput_,
    const at::Tensor& output_) {
  Tensor dataInput{C10Tensor(dataInput_)};
  Tensor indicesInput{C10Tensor(indicesInput_)};
  Tensor lengthsInput{C10Tensor(lengthsInput_)};
  Tensor output{C10Tensor(output_)};

  using T = float;
  constexpr bool USE_MEAN = false;
  constexpr bool USE_POSITIONAL_WEIGHT = false;

  CAFFE_ENFORCE_EQ(1, indicesInput.dim(), "INDICES must be a vector");
  CAFFE_ENFORCE_EQ(1, lengthsInput.dim(), "LENGTHS must be a vector");
  const int64_t N = dataInput.size(0);
  const int D = dataInput.size_from_dim(1);
  const int64_t M = lengthsInput.size(0);
  const int64_t indices_size = indicesInput.numel();

  auto shape = dataInput.sizes().vec();
  shape[0] = M;
  output.Resize(shape);
  T* out_data = output.template mutable_data<T>();

  const InputType* in_data = dataInput.template data<InputType>();
  const IndexType* indices = indicesInput.template data<IndexType>();
  const int* lengths = lengthsInput.template data<int>();
  const T* in_weight = nullptr;

  // delegate work to perfkernel that branches based on architecture
  caffe2::EmbeddingLookup<IndexType, InputType, T, USE_POSITIONAL_WEIGHT>(
      D,
      M,
      indices_size,
      N,
      in_data,
      indices,
      lengths,
      in_weight,
      nullptr, // scale_bias field is only used in SparseLengths8BitsRowwiseOp
      USE_MEAN,
      out_data);
}

template<typename IndexType>
void sparse_lengths_sum_op_cpu_impl(
  const at::Tensor& dataInput,
  const at::Tensor& indicesInput,
  const at::Tensor& lengthsInput,
  const at::Tensor& output) {
  switch (dataInput.scalar_type()) {
    case ScalarType::Float: return sparse_lengths_sum_op_cpu_impl_<float, IndexType>(dataInput, indicesInput, lengthsInput, output);
    case ScalarType::Half: return sparse_lengths_sum_op_cpu_impl_<at::Half, IndexType>(dataInput, indicesInput, lengthsInput, output);
    default: throw std::runtime_error(string() + "Unsupported dtype for input data " + toString(dataInput.scalar_type()));
  }
}

void sparse_lengths_sum_op_cpu(
  const at::Tensor& dataInput,
  const at::Tensor& indicesInput,
  const at::Tensor& lengthsInput,
  const at::Tensor& output) {
  switch (indicesInput.scalar_type()) {
    case ScalarType::Int: return sparse_lengths_sum_op_cpu_impl<int>(dataInput, indicesInput, lengthsInput, output);
    case ScalarType::Long: return sparse_lengths_sum_op_cpu_impl<int64_t>(dataInput, indicesInput, lengthsInput, output);
    default: throw std::runtime_error(string() + "Unsupported dtype for input indices " + toString(dataInput.scalar_type()));
  }
}

} // namespace
} // namespace caffe2

namespace c10 {
C10_REGISTER_KERNEL(caffe2::ops::SparseLengthsSum)
    .kernel<decltype(caffe2::sparse_lengths_sum_op_cpu), &caffe2::sparse_lengths_sum_op_cpu>()
    .dispatchKey(CPUTensorId());
} // namespace c10
