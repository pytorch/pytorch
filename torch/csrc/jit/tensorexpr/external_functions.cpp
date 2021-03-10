#include <ideep/abstract_types.hpp>
#include <torch/csrc/jit/tensorexpr/external_functions.h>

#include <ATen/Functions.h>
#include <ATen/NativeFunctions.h>
#include <torch/csrc/jit/tensorexpr/external_functions_registry.h>
#include <ideep.hpp>
#include <ATen/native/mkldnn/MKLDNNCommon.h>
#include <ATen/native/ConvUtils.h>
#include <tuple>
#include <unordered_map>
#include "c10/core/ScalarType.h"
#include "c10/util/Exception.h"
#include <torch/csrc/jit/jit_log.h>

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
    buf_dims_vec.emplace_back();
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

std::tuple<std::vector<ideep::tensor>, bool> constructItensors(
    int64_t bufs_num,
    void** buf_data,
    int64_t* buf_ranks,
    int64_t* buf_dims,
    int8_t* buf_dtypes) {

  std::vector<void*> buf_data_vec;
  std::vector<std::vector<int64_t>> buf_dims_vec;
  std::vector<c10::ScalarType> buf_dtypes_vec;

  std::vector<ideep::tensor> empty {};

  int64_t buf_dims_idx = 0;
  for (int64_t i = 0; i < bufs_num; i++) {
    buf_data_vec.push_back(buf_data[i]);
    buf_dims_vec.emplace_back();
    for (int64_t dim = 0; dim < buf_ranks[i]; dim++) {
      buf_dims_vec[i].push_back(buf_dims[buf_dims_idx++]);
    }
    buf_dtypes_vec.push_back(static_cast<c10::ScalarType>(buf_dtypes[i]));
  }

  // TODO: MKLDNN can support more types but let's start with floats
  std::unordered_map<c10::ScalarType, ideep::data_type> c102it {{c10::ScalarType::Float, ideep::data_type::f32}};
  auto tag = ideep::format_tag::undef;
  // TODO: support more layouts         0            1             2    3              4  
  std::vector<ideep::format_tag> tags {tag, ideep::format_tag::a, tag, tag, ideep::format_tag::nchw};
  std::vector<ideep::tensor> tensors;
  for (size_t i = 0; i < buf_data_vec.size(); i++) {
    TORCH_INTERNAL_ASSERT(buf_dims_vec[i].size() == 1 || buf_dims_vec[i].size() == 4);
    if (c102it.count(buf_dtypes_vec[i]) == 0) {
      return std::make_tuple(empty, false);
    }
    tensors.emplace_back(ideep::tensor(buf_dims_vec[i], c102it.at(buf_dtypes_vec[i]), tags[buf_dims_vec[i].size()], buf_data_vec[i]));
  }
  return std::make_tuple(tensors, true);
}

void nnc_aten_conv2d_out(
  int64_t bufs_num,
  void** buf_data,
  int64_t* buf_ranks,
  int64_t* buf_dims,
  int8_t* buf_dtypes,
  int64_t args_num,
  int64_t* extra_args) {

    std::vector<ideep::tensor> tensors;
    bool args_valid_for_mkldnn;
    std::tie(tensors, args_valid_for_mkldnn) =
      constructItensors(bufs_num, buf_data, buf_ranks, buf_dims, buf_dtypes);

    if (!args_valid_for_mkldnn) {
      nnc_aten_conv2d(bufs_num, buf_data, buf_ranks, buf_dims, buf_dtypes, args_num, extra_args);
      return;
    }

    auto&  r = tensors[0];
    const auto&  x = tensors[1];
    const auto&  w = tensors[2];
    int64_t strideH = 1;
    int64_t strideW = 1; 
    int64_t paddingH = 0; 
    int64_t paddingW = 0; 
    int64_t dilationH = 1; 
    int64_t dilationW = 1; 
    int64_t groups = 1;

    if (args_num > 0) {
      TORCH_INTERNAL_ASSERT(args_num == 7);
      strideH = extra_args[0];
      strideW = extra_args[1];
      paddingH = extra_args[2];
      paddingW = extra_args[3];
      dilationH = extra_args[4];
      dilationW = extra_args[5];
      groups = extra_args[6];
    }

    std::vector<int64_t> output_sizes = at::native::conv_output_size(x.get_dims(), w.get_dims(), {paddingH, paddingW}, {strideH, strideW}, {dilationH, dilationW});
    ideep::tensor y;
    try {
      if (bufs_num == 4) {
            const auto& b = tensors[3];
            ideep::convolution_forward::compute(
              x, w, b, {output_sizes.cbegin(), output_sizes.cend()}, y, {strideH, strideW}, {dilationH, dilationW}, {paddingH, paddingW}, {paddingH, paddingW},
              groups);
      } else {
            ideep::convolution_forward::compute(
              x, w, {output_sizes.cbegin(), output_sizes.cend()}, y, {strideH, strideW}, {dilationH, dilationW}, {paddingH, paddingW}, {paddingH, paddingW},
              groups);
      }
      y.reorder_to(r);
    } catch (...) {
      GRAPH_DEBUG("Exception thrown while executing ideep::convolution_forward::compute");
    }

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
      r = at::native::conv2d(
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
      r = at::native::conv2d(x, w);
    } catch (...) {
    }
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
  try {
    at::matmul_out(r, x, w);
  } catch (...) {
  }
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
  int64_t W = extra_args[1];
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
  int64_t dim = extra_args[0];
  try {
    at::mean_out(r, x, {dim});
  } catch (...) {
  }
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
  try {
    at::addmm_out(r, x, y, z, extra_args[0], extra_args[1]);
  } catch (...) {
  }
}

static RegisterNNCExternalFunction nnc_conv2d(
    "nnc_aten_conv2d",
    nnc_aten_conv2d_out);
static RegisterNNCExternalFunction nnc_matmul(
    "nnc_aten_matmul",
    nnc_aten_matmul);
static RegisterNNCExternalFunction nnc_adaptive_avg_pool2d(
    "nnc_aten_adaptive_avg_pool2d",
    nnc_aten_adaptive_avg_pool2d);
static RegisterNNCExternalFunction nnc_mean("nnc_aten_mean", nnc_aten_mean);
static RegisterNNCExternalFunction nnc_addmm("nnc_aten_addmm", nnc_aten_addmm);

} // namespace tensorexpr
} // namespace jit
} // namespace torch
