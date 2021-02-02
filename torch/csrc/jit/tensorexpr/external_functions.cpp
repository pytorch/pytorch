#include <torch/csrc/jit/tensorexpr/external_functions.h>

#include <ATen/Functions.h>
#include <ATen/NativeFunctions.h>
#include <torch/csrc/jit/tensorexpr/external_functions_registry.h>

namespace torch {
namespace jit {
namespace tensorexpr {

// A helper function to construct a vector of tensors from raw buffer arguments
std::vector<at::Tensor> constructTensors(
    int64_t bufs_num,
    void** buf_data,
    int64_t* buf_ranks,
    int64_t* buf_dims,
    int8_t* buf_dtypes) {
  std::vector<void*> buf_data_vec;
  std::vector<std::vector<int64_t>> buf_dims_vec;
  std::vector<c10::ScalarType> buf_dtypes_vec;
  int64_t buf_dims_idx = 0;
  for (int64_t i = 0; i < bufs_num; i++) {
    buf_data_vec.push_back(buf_data[i]);
    buf_dims_vec.push_back({});
    for (int64_t dim = 0; dim < buf_ranks[i]; dim++) {
      buf_dims_vec[i].push_back(buf_dims[buf_dims_idx++]);
    }
    buf_dtypes_vec.push_back(static_cast<c10::ScalarType>(buf_dtypes[i]));
  }

  std::vector<at::Tensor> tensors;
  for (size_t i = 0; i < buf_data_vec.size(); i++) {
    auto options = at::TensorOptions()
                       .dtype(buf_dtypes_vec[i])
                       .layout(at::kStrided)
                       .device(at::kCPU) // TODO: support GPUs too
                       .requires_grad(false);
    tensors.emplace_back(
        at::from_blob(buf_data_vec[i], buf_dims_vec[i], options));
  }
  return tensors;
}

void nnc_aten_conv2d(
    int64_t bufs_num,
    void** buf_data,
    int64_t* buf_ranks,
    int64_t* buf_dims,
    int8_t* buf_dtypes,
    int64_t args_num,
    int64_t* extra_args) {
  std::vector<at::Tensor> tensors =
      constructTensors(bufs_num, buf_data, buf_ranks, buf_dims, buf_dtypes);

  at::Tensor& r = tensors[0];
  const at::Tensor& x = tensors[1];
  const at::Tensor& w = tensors[2];
  int64_t strideH = 1, strideW = 1;
  int64_t paddingH = 0, paddingW = 0;
  int64_t dilationH = 1, dilationW = 1;
  int64_t groups = 1;
  if (args_num > 0) {
    // Check that if the extra arguments are provided, then the bias tensor is
    // also present
    TORCH_INTERNAL_ASSERT(args_num == 7 && bufs_num == 4);
    strideH = extra_args[0];
    strideW = extra_args[1];
    paddingH = extra_args[2];
    paddingW = extra_args[3];
    dilationH = extra_args[4];
    dilationW = extra_args[5];
    groups = extra_args[6];
  }
  if (bufs_num > 3) {
    const at::Tensor& b = tensors[3];
    r = at::native::conv2d(
        x,
        w,
        b,
        {strideH, strideW},
        {paddingH, paddingW},
        {dilationH, dilationW},
        groups);
  } else {
    r = at::native::conv2d(x, w);
  }

  // TODO: can i haz an out version of the conv2d?
  memcpy(buf_data[0], r.data_ptr(), r.element_size() * r.numel());
}

void nnc_aten_matmul(
    int64_t bufs_num,
    void** buf_data,
    int64_t* buf_ranks,
    int64_t* buf_dims,
    int8_t* buf_dtypes,
    int64_t args_num,
    int64_t* extra_args) {
  std::vector<at::Tensor> tensors =
      constructTensors(bufs_num, buf_data, buf_ranks, buf_dims, buf_dtypes);

  at::Tensor& r = tensors[0];
  const at::Tensor& x = tensors[1];
  const at::Tensor& w = tensors[2];
  at::matmul_out(r, x, w);
}

static RegisterNNCExternalFunction nnc_conv2d(
    "nnc_aten_conv2d",
    nnc_aten_conv2d);
static RegisterNNCExternalFunction nnc_matmul(
    "nnc_aten_matmul",
    nnc_aten_matmul);

} // namespace tensorexpr
} // namespace jit
} // namespace torch
