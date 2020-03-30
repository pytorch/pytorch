#include <ATen/core/op_registration/op_registration.h>
#include "caffe2/core/export_c10_op_to_caffe2.h"
#include "caffe2/core/tensor.h"
#include "caffe2/perfkernels/embedding_lookup.h"
#include "caffe2/utils/math.h"

using caffe2::Tensor;

namespace caffe2 {
namespace {

template <typename InputType, typename IndexType>
void sparse_lengths_sum_op_cpu_impl_(
    const at::Tensor& dataInput_,
    const at::Tensor& indicesInput_,
    const at::Tensor& lengthsInput_,
    const at::Tensor& output_) {
  Tensor dataInput(dataInput_);
  Tensor indicesInput(indicesInput_);
  Tensor lengthsInput(lengthsInput_);
  Tensor output(output_);

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

static auto registry = c10::RegisterOperators().op(
    "_c10_experimental::SparseLengthsSum",
    c10::RegisterOperators::options()
      .kernel<
        decltype(sparse_lengths_sum_op_cpu),
        &sparse_lengths_sum_op_cpu>(DispatchKey::CPUTensorId));

} // namespace

C10_EXPORT_C10_OP_TO_CAFFE2_CPU(
    "_c10_experimental::SparseLengthsSum",
    C10SparseLengthsSum_DontUseThisOpYet)

} // namespace caffe2
