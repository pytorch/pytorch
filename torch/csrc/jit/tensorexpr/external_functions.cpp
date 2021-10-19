#include <torch/csrc/jit/tensorexpr/external_functions.h>

#include <ATen/Functions.h>
#include <ATen/NativeFunctions.h>
#include <ATen/core/dispatch/Dispatcher.h>
#include <ATen/native/xnnpack/OpContext.h>
#include <c10/util/irange.h>
#include <torch/csrc/jit/tensorexpr/exceptions.h>
#include <torch/csrc/jit/tensorexpr/external_functions_registry.h>

namespace torch {
namespace jit {
namespace tensorexpr {

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
  for (const auto i : c10::irange(bufs_num)) {
    buf_data_vec.push_back(buf_data[i]);
    buf_dims_vec.emplace_back();
    for (const auto dim : c10::irange(buf_ranks[i])) {
      (void)dim;
      buf_dims_vec[i].push_back(buf_dims[buf_dims_idx++]);
    }
    buf_dtypes_vec.push_back(static_cast<c10::ScalarType>(buf_dtypes[i]));
  }

  std::vector<at::Tensor> tensors;
  for (const auto i : c10::irange(buf_data_vec.size())) {
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

#ifdef C10_MOBILE
extern "C" {
#endif

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
  if (args_num > 0) {
    // Check that if the extra arguments are provided, then the bias tensor is
    // also present
    TORCH_INTERNAL_ASSERT(args_num == 7 && bufs_num == 4);
    const at::Tensor& b = tensors[3];

    int64_t strideH = extra_args[0];
    int64_t strideW = extra_args[1];
    int64_t paddingH = extra_args[2];
    int64_t paddingW = extra_args[3];
    int64_t dilationH = extra_args[4];
    int64_t dilationW = extra_args[5];
    int64_t groups = extra_args[6];

    try {
      r = at::conv2d(
          x,
          w,
          b,
          {strideH, strideW},
          {paddingH, paddingW},
          {dilationH, dilationW},
          groups);
    } catch (...) {
    }
  } else {
    try {
      r = at::conv2d(x, w);
    } catch (...) {
    }
  }

  // TODO: can i haz an out version of the conv2d?
  memcpy(buf_data[0], r.data_ptr(), r.element_size() * r.numel());
}

void nnc_aten_conv1d(
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
  if (args_num > 0) {
    // Check that if the extra arguments are provided, then the bias tensor is
    // also present
    TORCH_INTERNAL_ASSERT(args_num == 4 && bufs_num == 4);
    const at::Tensor& b = tensors[3];

    int64_t stride = extra_args[0];
    int64_t padding = extra_args[1];
    int64_t dilation = extra_args[2];
    int64_t groups = extra_args[3];

    try {
      r = at::conv1d(x, w, b, {stride}, {padding}, {dilation}, groups);
    } catch (...) {
    }
  } else {
    try {
      r = at::conv1d(x, w);
    } catch (...) {
    }
  }

  memcpy(buf_data[0], r.data_ptr(), r.element_size() * r.numel());
}

void nnc_aten_adaptive_avg_pool2d(
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
  int64_t H = extra_args[0];
  int64_t W = H;
  if (args_num > 1) {
    W = extra_args[1];
  }
  try {
    at::adaptive_avg_pool2d_out(r, x, {H, W});
  } catch (...) {
  }
}

void nnc_aten_mean(
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
  std::vector<int64_t> mean_dims(args_num);
  if (args_num > 0) {
    memcpy(mean_dims.data(), extra_args, sizeof(int64_t) * args_num);
  }
  try {
    at::mean_out(r, x, mean_dims);
  } catch (...) {
  }
}

void nnc_aten_max_red(
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
  int64_t max_dim = extra_args[0];
  bool keep_dim = extra_args[1];
  try {
    r =  std::get<0>(at::max(x, max_dim, keep_dim));
  } catch (...) {
  }
  memcpy(buf_data[0], r.data_ptr(), r.element_size() * r.numel());
}

void nnc_aten_addmm(
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
  const at::Tensor& y = tensors[2];
  const at::Tensor& z = tensors[3];
  // TODO: handle other alpha and beta dtypes, e.g. alpha=0.6, beta=0.2
  int64_t alpha = extra_args[0], beta = extra_args[1];

  try {
    at::addmm_out(r, x, y, z, alpha, beta);
  } catch (...) {
  }
}

// Only provides first output, the second output is just a copy of one of the
// inputs
void nnc_aten_triangular_solve(
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
  at::Tensor r2 = tensors[2].clone();
  const at::Tensor& input = tensors[1];
  const at::Tensor& A = tensors[2];
  try {
    at::triangular_solve_out(
        r, r2, input, A, extra_args[0], extra_args[2], extra_args[3]);
  } catch (...) {
  }
}

#ifdef USE_XNNPACK

void nnc_prepacked_linear_clamp_run(
    int64_t bufs_num,
    void** buf_data,
    int64_t* buf_ranks,
    int64_t* buf_dims,
    int8_t* buf_dtypes,
    int64_t args_num,
    int64_t* extra_args) {
  using namespace at::native::xnnpack;

  std::vector<at::Tensor> tensors =
      constructTensors(bufs_num - 1, buf_data, buf_ranks, buf_dims, buf_dtypes);

  const at::Tensor& x = tensors[1];
  auto context = reinterpret_cast<LinearOpContext*>(buf_data[2]);
  at::Tensor output = context->run(x);
  memcpy(
      buf_data[0], output.data_ptr(), output.element_size() * output.numel());
}

void nnc_prepacked_conv2d_clamp_run(
    int64_t bufs_num,
    void** buf_data,
    int64_t* buf_ranks,
    int64_t* buf_dims,
    int8_t* buf_dtypes,
    int64_t args_num,
    int64_t* extra_args) {
  using namespace at::native::xnnpack;

  std::vector<at::Tensor> tensors =
      constructTensors(bufs_num - 1, buf_data, buf_ranks, buf_dims, buf_dtypes);

  const at::Tensor& x = tensors[1];
  auto context = reinterpret_cast<Conv2dOpContext*>(buf_data[2]);
  at::Tensor output = context->run(x);
  memcpy(
      buf_data[0], output.data_ptr(), output.element_size() * output.numel());
}

#endif // USE_XNNPACK

void nnc_aten_embedding(
    int64_t bufs_num,
    void** buf_data,
    int64_t* buf_ranks,
    int64_t* buf_dims,
    int8_t* buf_dtypes,
    int64_t args_num,
    int64_t* extra_args) {
  std::vector<at::Tensor> tensors =
      constructTensors(bufs_num, buf_data, buf_ranks, buf_dims, buf_dtypes);

  /*
struct TORCH_API embedding {
  using schema = at::Tensor (const at::Tensor &, const at::Tensor &, int64_t, bool, bool);
  using ptr_schema = schema*;
  // See Note [static constexpr char* members for windows NVCC]
  STATIC_CONSTEXPR_STR_INL_EXCEPT_WIN_CUDA(name, "aten::embedding")
  STATIC_CONSTEXPR_STR_INL_EXCEPT_WIN_CUDA(overload_name, "")
  STATIC_CONSTEXPR_STR_INL_EXCEPT_WIN_CUDA(schema_str, "embedding(Tensor weight, Tensor indices, int padding_idx=-1, bool scale_grad_by_freq=False, bool sparse=False) -> Tensor")
  static at::Tensor call(const at::Tensor & weight, const at::Tensor & indices, int64_t padding_idx, bool scale_grad_by_freq, bool sparse);
  static at::Tensor redispatch(c10::DispatchKeySet dispatchKeySet, const at::Tensor & weight, const at::Tensor & indices, int64_t padding_idx, bool scale_grad_by_freq, bool sparse);
};
   */
  at::Tensor& r = tensors[0];
  const at::Tensor& weight = tensors[1];
  const at::Tensor& indices = tensors[2];
  try {
    r = at::embedding(weight, indices);
  } catch (...) {
  }
  // TODO: have to copy output because at::embedding doesnt have an out variant
  // and NNC's external calls don't support allocations
  memcpy(buf_data[0], r.data_ptr(), r.element_size() * r.numel());
}

#ifndef C10_MOBILE

const static RegisterNNCExternalFunction nnc_conv2d(
    "nnc_aten_conv2d",
    nnc_aten_conv2d);
const static RegisterNNCExternalFunction nnc_conv1d(
    "nnc_aten_conv1d",
    nnc_aten_conv1d);
const static RegisterNNCExternalFunction nnc_adaptive_avg_pool2d(
    "nnc_aten_adaptive_avg_pool2d",
    nnc_aten_adaptive_avg_pool2d);
const static RegisterNNCExternalFunction nnc_mean(
    "nnc_aten_mean",
    nnc_aten_mean);
const static RegisterNNCExternalFunction nnc_max_red("nnc_aten_max_red", nnc_aten_max_red);
const static RegisterNNCExternalFunction nnc_addmm(
    "nnc_aten_addmm",
    nnc_aten_addmm);

const static RegisterNNCExternalFunction nnc_triangular_solve(
    "nnc_aten_triangular_solve",
    nnc_aten_triangular_solve);

const static RegisterNNCExternalFunction nnc_embedding(
    "nnc_aten_embedding",
    nnc_aten_embedding);

#ifdef USE_XNNPACK
const static RegisterNNCExternalFunction reg_nnc_prepacked_linear_clamp_run(
    "nnc_prepacked_linear_clamp_run",
    nnc_prepacked_linear_clamp_run);
const static RegisterNNCExternalFunction reg_nnc_prepacked_conv2d_clamp_run(
    "nnc_prepacked_conv2d_clamp_run",
    nnc_prepacked_conv2d_clamp_run);
#endif // USE_XNNPACK

#endif // C10_MOBILE

#ifdef C10_MOBILE
} // extern "C"
#endif

} // namespace tensorexpr
} // namespace jit
} // namespace torch
