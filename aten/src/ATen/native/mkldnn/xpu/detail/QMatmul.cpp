#include <ATen/Tensor.h>
#include <ATen/core/Tensor.h>
#include <c10/core/ScalarType.h>

#include <ATen/native/mkldnn/xpu/detail/Attr.h>
#include <ATen/native/mkldnn/xpu/detail/oneDNNContext.h>

#include <oneapi/dnnl/dnnl.hpp>

namespace at::native::onednn {

at::Tensor broadcast_bias2D(
    at::Tensor& dst,
    at::Tensor& bias,
    int64_t m,
    int64_t n) {
  switch (bias.dim()) {
    case 1:
      TORCH_CHECK(
          bias.size(0) == n || bias.size(0) == 1,
          "matmul supports [n] or [1] when bias dim is 1, but b.size() is:",
          bias.size(0));
      break;
    case 2:
      if ((bias.size(0) == m && bias.size(1) == n) ||
          (bias.size(0) == m && bias.size(1) == 1) ||
          (bias.size(0) == m && bias.size(1) == 1))
        return bias; // No need to broadcast
      TORCH_CHECK(
          bias.size(0) == 1 && bias.size(1) == 1,
          "matmul supports [m, n] or [1, n] or [m, 1] or [1, 1] when bias dim is 2 ...")
      break;
    case 0:
      TORCH_CHECK(
          bias.numel() == 1, "matmul supports 1 numel when bias dim is [] ...");
      break;
    default:
      TORCH_CHECK(0, "unsupported bias dim in matmul ...");
  }
  bias = bias.expand({1, n}).contiguous();
  return bias;
}

at::Tensor broadcast_bias3D(
    at::Tensor& dst,
    at::Tensor bias,
    int64_t mb,
    int64_t m,
    int64_t n) {
  switch (bias.dim()) {
    case 1:
      TORCH_CHECK(
          bias.size(0) == n || bias.size(0) == 1,
          "matmul supports [n] or [1] when bias dim is 1, but b.size() is:",
          bias.size(0));
      break;
    case 3:
      TORCH_CHECK(
          are_expandable({mb, m, n}, bias.sizes()),
          "matmul bias must be expandable to:",
          dst.sizes(),
          " but got:",
          bias.sizes());
      break;
    case 0:
      TORCH_CHECK(
          bias.numel() == 1, "matmul supports 1 numel when bias dim is [] ...");
      break;
    default:
      TORCH_CHECK(0, "unsupported bias dim in matmul ...");
  }
  bias = bias.expand({mb, m, n}).contiguous();
  return bias;
}

at::Tensor broadcast_bias(
    at::Tensor& dst,
    at::Tensor bias,
    int64_t mb,
    int64_t m,
    int64_t n) {
  if (dst.dim() == 2) {
    return broadcast_bias2D(dst, bias, m, n);
  } else {
    return broadcast_bias3D(dst, bias, mb, m, n);
  }
}

void quantized_matmul(
    at::Tensor mat1, // act
    double input_scale,
    int64_t input_zero_point,
    at::Tensor mat2, // weight
    at::Tensor& weight_scales,
    at::Tensor& weight_zero_points,
    at::Tensor& bias,
    at::Tensor result, // output
    double output_scale,
    int64_t output_zero_point,
    std::optional<c10::ScalarType> output_dtype,
    std::optional<at::Tensor> other, // extra input for binary-post-op
    double other_scale,
    int64_t other_zero_point,
    const c10::string_view& binary_post_op,
    double binary_alpha,
    const c10::string_view& unary_post_op,
    torch::List<std::optional<at::Scalar>>& unary_post_op_args,
    c10::string_view unary_post_op_algorithm,
    bool m2_trans) {
  // [Note] Quantized Matrix Multiplication at XPU
  // The following code integrates oneDNN quantized gemm. The quantization
  // config we support:
  // activation: s8&u8; per tensor calibrated; symmetric&asymmetric
  // weight: s8; per_tensor/per_channel calibrated; symmetric
  auto attr = Attr(1.0 / output_scale, output_zero_point);
  construct_attr_by_post_op(
      binary_post_op,
      binary_alpha,
      input_scale,
      input_zero_point,
      other,
      unary_post_op,
      unary_post_op_args,
      unary_post_op_algorithm,
      attr);

  size_t dims = result.dim();
  at::Device cur_device = at::Device(at::kXPU, c10::xpu::current_device());
  auto engine = GpuEngineManager::Instance().get_engine(cur_device);
  auto stream = GpuStreamManager::Instance().get_stream();

  at::Tensor m1 = is_onednn_matmul_strides(mat1) ? mat1 : mat1.contiguous();
  at::Tensor m2 = is_onednn_matmul_strides(mat2) ? mat2 : mat2.contiguous();
  at::Tensor dst =
      is_onednn_matmul_strides(result, true) ? result : result.contiguous();

  int64_t m = dst.size(-2);
  int64_t n = dst.size(-1);
  int64_t k = m1.size(-1);
  int64_t mb = 1;

  if (dims == 3) {
    mb = dst.size(0);
    TORCH_CHECK(
        mb == m1.size(0) && mb == m2.size(0),
        "batch size mismatch, dst mb: ",
        mb,
        "m1 mb",
        m1.size(0),
        " m2 mb: ",
        m2.size(0));
  }

  bool with_bias = false;
  at::Tensor b = bias;
  if (b.defined()) {
    with_bias = true;
    b = broadcast_bias(dst, b, mb, m, n);
  }
  // bias is fused in post-op for quantized path
  b = b.contiguous(); // avoid reorder 2 times

  auto m1_usr_dt = get_onednn_dtype(m1);
  auto m2_usr_dt = get_onednn_dtype(m2);
  auto dst_usr_dt = get_onednn_dtype(dst);

  auto m1_dt = m1_usr_dt;
  auto m2_dt = m2_usr_dt;
  auto dst_dt = dst_usr_dt;
  dnnl::memory::data_type bias_dt;

  dnnl::memory::desc m1_md, m1_usr_md;
  dnnl::memory::desc m2_md, m2_usr_md;
  dnnl::memory::desc dst_md, dst_usr_md;
  dnnl::memory::desc b_md;

  dnnl::memory::dims m1_dims, m2_dims, dst_dims, bias_dims;
  dnnl::memory::dims m1_strides, m2_strides, dst_strides, bias_strides;
  if (dims == 2) {
    m1_dims = {m, k};
    m2_dims = {k, n}; // (n, 1) (1, n)
    dst_dims = {m, n};

    m1_strides = {m1.stride(0), m1.stride(1)};
    if (m2_trans) {
      m2_strides = {m2.stride(0), m2.stride(1)};
    } else {
      m2_strides = {m2.stride(1), m2.stride(0)};
    }
    dst_strides = {dst.stride(0), dst.stride(1)};
  } else {
    m1_dims = {mb, m, k};
    m2_dims = {mb, k, n};
    dst_dims = {mb, m, n};

    m1_strides = {m1.stride(0), m1.stride(1), m1.stride(2)};
    if (m2_trans) {
      m2_strides = {m2.stride(0), m2.stride(1), m2.stride(2)};
    } else {
      m2_strides = {m2.stride(0), m2.stride(2), m2.stride(1)};
    }
    dst_strides = {dst.stride(0), dst.stride(1), dst.stride(2)};
  }

  if (with_bias) {
    bias_dims = get_onednn_dims(b);
    bias_dt = get_onednn_dtype(b);
    bias_strides = get_onednn_strides(b);
  }

  std::unordered_map<int, dnnl::memory> args;

  dnnl::post_ops po;
  po = attr.extract_post_ops(dst);
  bool m1_need_zp = (input_zero_point != 0);
  bool dst_need_zp = (output_zero_point != 0);
  bool wgh_is_per_channel = weight_scales.numel() > 1;

  dnnl::matmul matmul_p;
  dnnl::matmul::primitive_desc matmul_pd;

  m1_md = dnnl::memory::desc(m1_dims, m1_dt, m1_strides);
  m2_md = dnnl::memory::desc(m2_dims, m2_dt, m2_strides);
  dst_md = dnnl::memory::desc(dst_dims, dst_dt, dst_strides);
  dnnl::primitive_attr pattr;
  pattr.set_post_ops(po);
  pattr.set_scratchpad_mode(dnnl::scratchpad_mode::user);

  at::Tensor m2_sc;
  if (!wgh_is_per_channel) {
    pattr.set_scales_mask(DNNL_ARG_WEIGHTS, 0);
  } else {
    pattr.set_scales_mask(DNNL_ARG_WEIGHTS, 1 << 1);
  }

  at::Tensor m1_sc;
  dnnl::memory::desc m1_sc_md = dnnl::memory::desc(
      {1}, dnnl::memory::data_type::f32, dnnl::memory::format_tag::x);
  int mask_ac = 0;
  pattr.set_scales_mask(DNNL_ARG_SRC, mask_ac);
  if (m1_need_zp) {
    pattr.set_zero_points_mask(DNNL_ARG_SRC, mask_ac);
  }
  pattr.set_scales_mask(DNNL_ARG_DST, mask_ac);
  if (dst_need_zp) {
    pattr.set_zero_points_mask(DNNL_ARG_DST, mask_ac);
  }

  if (with_bias) {
    b_md = dnnl::memory::desc(bias_dims, bias_dt, bias_strides);
    matmul_pd =
        dnnl::matmul::primitive_desc(engine, m1_md, m2_md, b_md, dst_md, pattr);
  } else {
    matmul_pd =
        dnnl::matmul::primitive_desc(engine, m1_md, m2_md, dst_md, pattr);
  }

  matmul_p = dnnl::matmul(matmul_pd);

  m1_usr_md = dnnl::memory::desc(m1_dims, m1_usr_dt, m1_strides);
  m2_usr_md = dnnl::memory::desc(m2_dims, m2_usr_dt, m2_strides);
  dst_usr_md = dnnl::memory::desc(dst_dims, dst_usr_dt, dst_strides);

  auto m1_usr_m = make_onednn_memory(m1_usr_md, engine, m1.data_ptr());
  auto m2_usr_m = make_onednn_memory(m2_usr_md, engine, m2.data_ptr());
  auto dst_usr_m = make_onednn_memory(dst_usr_md, engine, dst.data_ptr());

  auto expected_m1_md = matmul_pd.src_desc();
  auto expected_m2_md = matmul_pd.weights_desc();
  auto expected_dst_md = matmul_pd.dst_desc();

  dnnl::memory m1_m = m1_usr_m, m2_m = m2_usr_m, dst_m = dst_usr_m;
  at::Tensor m1_, m2_, dst_;

  int scratchpad_size = matmul_pd.scratchpad_desc().get_size();
  at::Tensor scratchpad_tensor =
      at::empty({scratchpad_size}, m1.options().dtype(at::kByte), c10::nullopt);
  auto scratchpad_memory = make_onednn_memory(
      matmul_pd.scratchpad_desc(), engine, scratchpad_tensor.data_ptr());
  args.insert({DNNL_ARG_SCRATCHPAD, scratchpad_memory});

  if (attr.with_binary())
    attr.construct_post_binary(matmul_pd, args);

  args.insert({DNNL_ARG_SRC, m1_m});
  args.insert({DNNL_ARG_WEIGHTS, m2_m});
  args.insert({DNNL_ARG_DST, dst_m});
  if (b.defined()) {
    auto b_m = make_onednn_memory(b_md, engine, b.data_ptr());
    args.insert({DNNL_ARG_BIAS, b_m});
  }

  // Add scale/zp md
  weight_scales = weight_scales.to(at::kFloat);
  dnnl::memory m2_sc_m, m2_zp_m;
  dnnl::memory::desc m2_sc_md = dnnl::memory::desc(
      get_onednn_dims(weight_scales),
      dnnl::memory::data_type::f32,
      dnnl::memory::format_tag::x);
  m2_sc_m = make_onednn_memory(m2_sc_md, engine, weight_scales.data_ptr());
  args.insert({DNNL_ARG_ATTR_SCALES | DNNL_ARG_WEIGHTS, m2_sc_m});

  dnnl::memory m1_sc_m, m1_zp_m;
  Tensor m1_sc_tensor, m1_zp_tensor;
  m1_sc_m = dnnl_memory_from_host_scalar(
      static_cast<float>(input_scale), m1_sc_tensor, engine);
  m1_zp_m = dnnl_memory_from_host_scalar(
      static_cast<int32_t>(input_zero_point), m1_zp_tensor, engine);

  args.insert({DNNL_ARG_ATTR_SCALES | DNNL_ARG_SRC, m1_sc_m});
  if (m1_need_zp) {
    args.insert({DNNL_ARG_ATTR_ZERO_POINTS | DNNL_ARG_SRC, m1_zp_m});
  }

  dnnl::memory dst_sc_m, dst_zp_m;
  Tensor dst_sc_tensor, dst_zp_tensor;
  dst_sc_m = dnnl_memory_from_host_scalar(
      static_cast<float>(output_scale), dst_sc_tensor, engine);
  dst_zp_m = dnnl_memory_from_host_scalar(
      static_cast<int32_t>(output_zero_point), dst_zp_tensor, engine);
  args.insert({DNNL_ARG_ATTR_SCALES | DNNL_ARG_DST, dst_sc_m});
  if (dst_need_zp) {
    args.insert({DNNL_ARG_ATTR_ZERO_POINTS | DNNL_ARG_DST, dst_zp_m});
  }

  auto qmatmul_event = dnnl::sycl_interop::execute(matmul_p, stream, args);

  if (!dst.is_same(result))
    result.copy_(dst);
}

} // namespace at::native::onednn
