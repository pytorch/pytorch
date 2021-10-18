#include <torch/csrc/jit/tensorexpr/external_functions.h>

#include <ATen/ATen.h>
#include <ATen/Functions.h>
#include <ATen/NativeFunctions.h>
#include <ATen/core/Tensor.h>
#include <ATen/core/dispatch/Dispatcher.h>
#include <ATen/native/quantized/cpu/conv_packed_params.h>
#include <ATen/native/quantized/cpu/conv_serialization.h>
#include <ATen/native/xnnpack/OpContext.h>
#include <ATen/quantized/Quantizer.h>
#include <c10/core/TensorOptions.h>
#include <c10/util/irange.h>
#include <torch/csrc/jit/jit_log.h>
#include <torch/csrc/jit/tensorexpr/exceptions.h>
#include <torch/csrc/jit/tensorexpr/external_functions_registry.h>

#include <iostream>

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

at::Tensor quantized_conv2d(
    at::Tensor qx,
    c10::intrusive_ptr<ConvPackedParamsBase<2>> packed_weight,
    double scale,
    int64_t zero) {
  auto qconv2d_op = c10::Dispatcher::singleton()
                        .findSchemaOrThrow("_quantized::conv2d", "")
                        .typed<at::Tensor(
                            at::Tensor,
                            const c10::intrusive_ptr<ConvPackedParamsBase<2>>&,
                            double,
                            int64_t)>();
  return qconv2d_op.call(qx, packed_weight, scale, zero);
}

void nnc_aten_quantized_conv2d(
    int64_t bufs_num,
    void** buf_data,
    int64_t* buf_ranks,
    int64_t* buf_dims,
    int8_t* buf_dtypes,
    int64_t args_num,
    int64_t* extra_args) {
  std::vector<at::Tensor> tensors =
      constructTensors(bufs_num, buf_data, buf_ranks, buf_dims, buf_dtypes);
  const double x_qscale = ((double*)extra_args)[0];
  const int64_t x_qzero = extra_args[1];
  const c10::ScalarType x_qdtype = static_cast<c10::ScalarType>(extra_args[2]);
  at::Tensor qx = at::from_blob_quantized_per_tensor_affine(
      buf_data[1],
      tensors[1].sizes(),
      [](void*) {},
      x_qscale,
      x_qzero,
      at::TensorOptions(toQIntType(x_qdtype)));
  auto convPackedParams =
      reinterpret_cast<ConvPackedParamsBase<2>*>(buf_data[2]);
  const double out_qscale = ((double*)extra_args)[3];
  const int64_t out_qzero = extra_args[4];
  auto r = convPackedParams->apply(qx, out_qscale, out_qzero);
  memcpy(buf_data[0], r.data_ptr(), r.element_size() * r.numel());
}

void nnc_aten_quantized_conv2d_relu(
    int64_t bufs_num,
    void** buf_data,
    int64_t* buf_ranks,
    int64_t* buf_dims,
    int8_t* buf_dtypes,
    int64_t args_num,
    int64_t* extra_args) {
  std::vector<at::Tensor> tensors =
      constructTensors(bufs_num, buf_data, buf_ranks, buf_dims, buf_dtypes);
  const double x_qscale = ((double*)extra_args)[0];
  const int64_t x_qzero = extra_args[1];
  const c10::ScalarType x_qdtype = static_cast<c10::ScalarType>(extra_args[2]);
  at::Tensor qx = at::from_blob_quantized_per_tensor_affine(
      buf_data[1],
      tensors[1].sizes(),
      [](void*) {},
      x_qscale,
      x_qzero,
      at::TensorOptions(toQIntType(x_qdtype)));
  auto convPackedParams =
      reinterpret_cast<ConvPackedParamsBase<2>*>(buf_data[2]);
  const double out_qscale = ((double*)extra_args)[3];
  const int64_t out_qzero = extra_args[4];
  auto r = convPackedParams->apply(qx, out_qscale, out_qzero);
  memcpy(buf_data[0], r.data_ptr(), r.element_size() * r.numel());
}

at::Tensor quantized_add(
    at::Tensor qa,
    at::Tensor qb,
    double out_scale,
    int64_t out_zero) {
  auto qadd_op =
      c10::Dispatcher::singleton()
          .findSchemaOrThrow("quantized::add", "")
          .typed<at::Tensor(at::Tensor, at::Tensor, double, int64_t)>();
  return qadd_op.call(qa, qb, out_scale, out_zero);
}

void nnc_aten_quantized_add(
    int64_t bufs_num,
    void** buf_data,
    int64_t* buf_ranks,
    int64_t* buf_dims,
    int8_t* buf_dtypes,
    int64_t args_num,
    int64_t* extra_args) {
  std::vector<at::Tensor> tensors =
      constructTensors(bufs_num, buf_data, buf_ranks, buf_dims, buf_dtypes);

  const double a_qscale = ((double*)extra_args)[0];
  const int64_t a_qzero = extra_args[1];
  const c10::ScalarType a_qdtype = static_cast<c10::ScalarType>(extra_args[2]);
  at::Tensor qa = at::from_blob_quantized_per_tensor_affine(
      buf_data[1],
      tensors[1].sizes(),
      [](void*) {},
      a_qscale,
      a_qzero,
      at::TensorOptions(toQIntType(a_qdtype)));
  const double b_qscale = ((double*)extra_args)[3];
  const int64_t b_qzero = extra_args[4];
  const c10::ScalarType b_qdtype = static_cast<c10::ScalarType>(extra_args[5]);
  at::Tensor qb = at::from_blob_quantized_per_tensor_affine(
      buf_data[2],
      tensors[2].sizes(),
      [](void*) {},
      b_qscale,
      b_qzero,
      at::TensorOptions(toQIntType(b_qdtype)));
  const double out_qscale = ((double*)extra_args)[6];
  const int64_t out_qzero = extra_args[7];
  auto r = quantized_add(qa, qb, out_qscale, out_qzero);
  memcpy(buf_data[0], r.data_ptr(), r.element_size() * r.numel());
}

c10::intrusive_ptr<ConvPackedParamsBase<2>> quantized_conv2d_prepack(
    at::Tensor qweight,
    c10::optional<at::Tensor> bias,
    c10::List<int64_t> stride,
    c10::List<int64_t> padding,
    c10::List<int64_t> dilation,
    int64_t groups) {
  auto qconv2d_prepack_op =
      c10::Dispatcher::singleton()
          .findSchemaOrThrow("quantized::conv2d_prepack", "")
          .typed<c10::intrusive_ptr<ConvPackedParamsBase<2>>(
              at::Tensor,
              c10::optional<at::Tensor>,
              c10::List<int64_t>,
              c10::List<int64_t>,
              c10::List<int64_t>,
              int64_t)>();
  return qconv2d_prepack_op.call(
      qweight, bias, stride, padding, dilation, groups);
}

void nnc_aten_quantized_conv2d_prepack(
    int64_t bufs_num,
    void** buf_data,
    int64_t* buf_ranks,
    int64_t* buf_dims,
    int8_t* buf_dtypes,
    int64_t args_num,
    int64_t* extra_args) {
  std::vector<at::Tensor> tensors =
      constructTensors(bufs_num, buf_data, buf_ranks, buf_dims, buf_dtypes);

  const double w_qscale = extra_args[7];
  const int64_t w_qzero = extra_args[8];
  const c10::ScalarType w_qdtype = static_cast<c10::ScalarType>(extra_args[9]);
  auto qw = at::from_blob_quantized_per_tensor_affine(
      buf_data[1],
      tensors[1].sizes(),
      [](void*) {},
      w_qscale,
      w_qzero,
      at::TensorOptions(toQIntType(w_qdtype)));
  auto b = tensors[2];

  int64_t strideH = extra_args[0];
  int64_t strideW = extra_args[1];
  int64_t paddingH = extra_args[2];
  int64_t paddingW = extra_args[3];
  int64_t dilationH = extra_args[4];
  int64_t dilationW = extra_args[5];
  int64_t groups = extra_args[6];
  c10::List<int64_t> strides = {strideH, strideW};
  c10::List<int64_t> paddings = {paddingH, paddingW};
  c10::List<int64_t> dilations = {dilationH, dilationW};
  c10::intrusive_ptr<ConvPackedParamsBase<2>> prepacked =
      quantized_conv2d_prepack(qw, b, strides, paddings, dilations, groups);
  TORCH_INTERNAL_ASSERT(
      prepacked, buildErrorMessage("Quantized conv2d prepack failed"));
  ConvParamsSerializationType serialized = serialize_conv<2>(prepacked);
  static std::vector<ConvParamsSerializationType> cache;
  cache.push_back(serialized);
  memcpy(buf_data[0], &serialized, sizeof(serialized));
}

void nnc_aten_upsample_nearest2d(
    int64_t bufs_num,
    void** buf_data,
    int64_t* buf_ranks,
    int64_t* buf_dims,
    int8_t* buf_dtypes,
    int64_t args_num,
    int64_t* extra_args) {
  std::vector<at::Tensor> tensors =
      constructTensors(bufs_num, buf_data, buf_ranks, buf_dims, buf_dtypes);
  at::Tensor x = tensors[0];
  const double x_qscale = ((double*)extra_args)[0];
  const int64_t x_qzero = extra_args[1];
  const int64_t x_qdtype = extra_args[2];
  const auto is_quantized = x_qdtype != -1;
  if (is_quantized) {
    x = at::from_blob_quantized_per_tensor_affine(
        buf_data[1],
        tensors[1].sizes(),
        [](void*) {},
        x_qscale,
        x_qzero,
        at::TensorOptions(toQIntType(static_cast<c10::ScalarType>(x_qdtype))));
  }

  int64_t output_size_h = extra_args[3];
  int64_t output_size_w = extra_args[4];
  double scale_factor_h = ((double*)extra_args)[5];
  double scale_factor_w = ((double*)extra_args)[6];

  c10::optional<at::IntArrayRef> output_size_arg = c10::nullopt;
  if (output_size_h != -1) {
    output_size_arg = {output_size_h, output_size_w};
  }
  c10::optional<at::ArrayRef<double>> scale_factors_arg = c10::nullopt;

  c10::optional<double> scales_h = c10::nullopt;
  c10::optional<double> scales_w = c10::nullopt;
  if (scale_factor_h != -1.f) {
    scales_h = scale_factor_h;
    scale_factors_arg = {scale_factor_h, scale_factor_w};
  }
  if (scale_factor_w != -1.f) {
    scales_w = scale_factor_w;
  }

  auto r = at::upsample_nearest2d(x, output_size_arg, scale_factors_arg);
  memcpy(buf_data[0], r.data_ptr(), r.element_size() * r.numel());
}

void nnc_aten_quantize_per_tensor(
    int64_t bufs_num,
    void** buf_data,
    int64_t* buf_ranks,
    int64_t* buf_dims,
    int8_t* buf_dtypes,
    int64_t args_num,
    int64_t* extra_args) {
  std::vector<at::Tensor> tensors =
      constructTensors(bufs_num, buf_data, buf_ranks, buf_dims, buf_dtypes);
  at::Tensor x = tensors[1];
  GRAPH_DEBUG("nnc_aten_quantize_per_tensor:", x);
  const double qscale = ((double*)extra_args)[0];
  const int64_t qzero = extra_args[1];
  const c10::ScalarType qdtype = static_cast<c10::ScalarType>(extra_args[2]);
  auto r = at::quantize_per_tensor(x, qscale, qzero, qdtype);
  memcpy(buf_data[0], r.data_ptr(), r.element_size() * r.numel());
}

void nnc_aten_dequantize(
    int64_t bufs_num,
    void** buf_data,
    int64_t* buf_ranks,
    int64_t* buf_dims,
    int8_t* buf_dtypes,
    int64_t args_num,
    int64_t* extra_args) {
  std::vector<at::Tensor> tensors =
      constructTensors(bufs_num, buf_data, buf_ranks, buf_dims, buf_dtypes);
  const double qscale = ((double*)extra_args)[0];
  const int64_t qzero = extra_args[1];
  const int64_t qdtype = extra_args[2];
  at::Tensor qx = at::from_blob_quantized_per_tensor_affine(
      buf_data[1],
      tensors[1].sizes(),
      [](void*) {},
      qscale,
      qzero,
      at::TensorOptions(toQIntType(static_cast<c10::ScalarType>(qdtype))));
  auto r = at::dequantize(qx);
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

#ifndef C10_MOBILE

const static RegisterNNCExternalFunction nnc_conv2d(
    "nnc_aten_conv2d",
    nnc_aten_conv2d);
const static RegisterNNCExternalFunction nnc_quantized_conv2d(
    "nnc_quantized_conv2d",
    nnc_aten_quantized_conv2d);
const static RegisterNNCExternalFunction nnc_quantized_conv2d_relu(
    "nnc_quantized_conv2d_relu",
    nnc_aten_quantized_conv2d_relu);
const static RegisterNNCExternalFunction nnc_quantized_add(
    "nnc_quantized_add",
    nnc_aten_quantized_add);
const static RegisterNNCExternalFunction nnc_quantize_per_tensor(
    "nnc_quantize_per_tensor",
    nnc_aten_quantize_per_tensor);
const static RegisterNNCExternalFunction nnc_dequantize(
    "nnc_dequantize",
    nnc_aten_dequantize);
const static RegisterNNCExternalFunction nnc_quantized_conv2d_prepack(
    "nnc_quantized_conv2d_prepack",
    nnc_aten_quantized_conv2d_prepack);
const static RegisterNNCExternalFunction nnc_upsample_nearest2d(
    "nnc_upsample_nearest2d",
    nnc_aten_upsample_nearest2d);
const static RegisterNNCExternalFunction nnc_adaptive_avg_pool2d(
    "nnc_aten_adaptive_avg_pool2d",
    nnc_aten_adaptive_avg_pool2d);
const static RegisterNNCExternalFunction nnc_mean(
    "nnc_aten_mean",
    nnc_aten_mean);
const static RegisterNNCExternalFunction nnc_addmm(
    "nnc_aten_addmm",
    nnc_aten_addmm);

const static RegisterNNCExternalFunction nnc_triangular_solve(
    "nnc_aten_triangular_solve",
    nnc_aten_triangular_solve);

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
