#include <ATen/core/dispatch/KernelRegistration.h>
#include "caffe2/operators/experimental/c10/schemas/batch_gather.h"
#include "caffe2/utils/math.h"
#include "caffe2/core/tensor.h"

using caffe2::BaseContext;
using caffe2::Tensor;
using std::vector;

namespace caffe2 {
namespace {

template <class TInd>
void batch_gather_op_cpu_impl(
    const at::Tensor& data_,
    const at::Tensor& indices_,
    const at::Tensor& output_) {
  Tensor data{C10Tensor(data_)};
  Tensor indices{C10Tensor(indices_)};
  Tensor output{C10Tensor(output_)};
  CPUContext context;

  CAFFE_ENFORCE_GE(data.dim(), 2, "DATA should be at least 2-D");

  vector<int64_t> shape;
  shape.push_back(data.size(0));
  shape.insert(shape.end(), indices.sizes().begin(), indices.sizes().end());
  shape.insert(shape.end(), data.sizes().begin() + 2, data.sizes().end());
  output.Resize(shape);

  auto block_size = data.size_from_dim(2);
  auto block_bytesize = block_size * data.dtype().itemsize();
  auto N = indices.numel();
  auto data_batch_bytesize = data.size_from_dim(1) * data.dtype().itemsize();
  auto gathered_batch_bytesize =
      N * data.size_from_dim(2) * data.dtype().itemsize();
  const TInd* idxs = indices.template data<TInd>();
  auto src_base = static_cast<const char*>(data.raw_data());
  auto out = static_cast<char*>(output.raw_mutable_data(data.dtype()));

  for (auto batch = 0; batch < data.size(0); ++batch) {
    for (auto i = 0; i < N; ++i) {
      auto idx = idxs[i];
      CAFFE_ENFORCE(
          0 <= idx && idx < data.size(1),
          "INDICES element is out of DATA bounds, id=",
          idx,
          " data_dim=",
          data.size(1));
      auto src = src_base + idx * block_bytesize + batch * data_batch_bytesize;
      auto dst = out + i * block_bytesize + batch * gathered_batch_bytesize;
      context.CopyItemsSameDevice(data.dtype(), block_size, src, dst);
    }
  }
}

void batch_gather_op_cpu(const at::Tensor& data,
    const at::Tensor& indices,
    const at::Tensor& output) {
  switch (data.scalar_type()) {
    case ScalarType::Int: return batch_gather_op_cpu_impl<int>(data, indices, output);
    case ScalarType::Long: return batch_gather_op_cpu_impl<int64_t>(data, indices, output);
    default: throw std::runtime_error(string() + "Unsupported dtype: " + toString(data.scalar_type()));
  }
}
} // namespace
} // namespace caffe2

namespace c10 {
C10_REGISTER_KERNEL(caffe2::ops::BatchGather)
    .kernel<decltype(caffe2::batch_gather_op_cpu), &caffe2::batch_gather_op_cpu>()
    .dispatchKey(CPUTensorId());
} // namespace c10
