
#include <c10/xpu/XPUFunctions.h>

#include <ATen/ATen.h>
#include <ATen/record_function.h>

#include <Attr.h>
#include <Utils.h>

#include <oneapi/dnnl/dnnl.hpp>

namespace at::native::onednn {

sycl::event matmul(
    at::Tensor& result,
    const at::Tensor& mat1,
    const at::Tensor& mat2,
    const at::Tensor& b_raw,
    bool m2_trans,
    Attr attr,
    const std::vector<sycl::event>& deps) {
  int64_t dims = result.dim();
  TORCH_CHECK(
      dims == 2 || dims == 3,
      "oneDNN matmul only works with 2D or 3D, got ",
      dims);
  TORCH_CHECK(
      dims == mat1.dim() && dims == mat2.dim(),
      "oneDNN input matrixes must have the same ranks");
  TORCH_CHECK(result.defined(), "oneDNN matmul result should be defined");

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

  // validate bias and make it compatible with oneDNN implementation
  bool with_bias = false;
  at::Tensor b = b_raw;
  if (b.defined()) {
    with_bias = true;
    if (b.dim() == 1) {
      TORCH_CHECK(
          b.size(0) == n || b.size(0) == 1,
          "matmul supports [n] or [1] when bias dim is 1 ...");
      if (b.size(0) == 0) {
        with_bias = false;
      } else if (m1.dim() == 3) {
        b = b.expand({mb, m, n}).contiguous();
      } else if (m1.dim() == 2) {
        b = b.expand({1, n}).contiguous();
      }
    } else if (b.dim() == 2) {
      TORCH_CHECK(
          (b.size(0) == m && b.size(1) == n) ||
              (b.size(0) == 1 && b.size(1) == n) ||
              (b.size(0) == m && b.size(1) == 1) ||
              (b.size(0) == 1 && b.size(1) == 1),
          "matmul supports [m, n] or [1, n] or [m, 1] or [1, 1] when bias dim is 2 ...");
      if (b.size(0) == 1 && b.size(1) == 1)
        b = b.expand({1, n}).contiguous();
    } else if (b.dim() == 3) {
      TORCH_CHECK(
          at::are_expandable({mb, m, n}, b.sizes()),
          "matmul bias must be expandable to:",
          dst.sizes(),
          " but got:",
          b.sizes());
      b = b.expand({mb, m, n}).contiguous();
    } else if (b.dim() == 0) {
      TORCH_CHECK(
          b.numel() == 1, "matmul supports 1 numel when bias dim is [] ...");
      if (m1.dim() == 3) {
        b = b.expand({mb, m, n}).contiguous();
      } else {
        b = b.expand({1, n}).contiguous();
      }
    } else {
      TORCH_CHECK(0, "unsupported bias dim in matmul ...");
    }
  }

  b = b.contiguous(); // avoid reorder 2 times

  // xpu matmul support both ab/ba shape for m2 tensor, we don't check any more
  auto m1_usr_dt = get_onednn_dtype(m1);
  auto m2_usr_dt = get_onednn_dtype(m2);
  auto dst_usr_dt = get_onednn_dtype(dst);

  auto m1_dt = m1_usr_dt;
  auto m2_dt = m2_usr_dt;
  auto dst_dt = dst_usr_dt;
  dnnl::memory::data_type bias_dt;

  dnnl::memory::desc m1_md, m1_usr_md, m1_any_md;
  dnnl::memory::desc m2_md, m2_usr_md, m2_any_md;
  dnnl::memory::desc dst_md, dst_usr_md, dst_any_md;
  dnnl::memory::desc bias_md;

  // Naive Master weight
  if (m1_dt == dnnl::memory::data_type::bf16 &&
      m2_dt == dnnl::memory::data_type::f32) {
    m2_dt = dnnl::memory::data_type::bf16;
    dst_dt = dnnl::memory::data_type::bf16;
  } else if (
      m1_dt == dnnl::memory::data_type::f32 &&
      m2_dt == dnnl::memory::data_type::bf16) {
    m1_dt = dnnl::memory::data_type::bf16;
    dst_dt = dnnl::memory::data_type::bf16;
  }

  dnnl::memory::dims m1_dims, m2_dims, dst_dims, bias_dims;
  dnnl::memory::dims m1_strides, m2_strides, dst_strides, bias_strides;
  if (dims == 2) {
    m1_dims = {m, k};
    m2_dims = {k, n};
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

  dnnl::post_ops po = attr.extract_post_ops(dst);

  std::unordered_map<int, dnnl::memory> args;
  dnnl::matmul matmul_p;
  dnnl::matmul::primitive_desc matmul_pd;

  // STEP1: create memory desc
  m1_md = dnnl::memory::desc(m1_dims, m1_dt, m1_strides);
  m2_md = dnnl::memory::desc(m2_dims, m2_dt, m2_strides);
  dst_md = dnnl::memory::desc(dst_dims, dst_dt, dst_strides);

  // STEP2: creat attribute
  dnnl::primitive_attr pattr;
  pattr.set_post_ops(po);

#if ONEDNN_SUPPORT_DETERMINISTIC
  if (at::globalContext().deterministicAlgorithms() ||
      at::globalContext().deterministicMkldnn())
    pattr.set_deterministic(true);
#endif

  // scratchpad
  pattr.set_scratchpad_mode(dnnl::scratchpad_mode::user);

  if (m1_dt == dnnl::memory::data_type::f32) {
    pattr.set_fpmath_mode(dnnl::fpmath_mode::strict);
  }

  // STEP3: create primitive
  if (with_bias) {
    bias_md = dnnl::memory::desc(bias_dims, bias_dt, bias_strides);
    matmul_pd = dnnl::matmul::primitive_desc(
        engine, m1_md, m2_md, bias_md, dst_md, pattr);
  } else {
    matmul_pd =
        dnnl::matmul::primitive_desc(engine, m1_md, m2_md, dst_md, pattr);
  }

  matmul_p = dnnl::matmul(matmul_pd);

  m1_usr_md = dnnl::memory::desc(m1_dims, m1_usr_dt, m1_strides);
  m2_usr_md = dnnl::memory::desc(m2_dims, m2_usr_dt, m2_strides);
  dst_usr_md = dnnl::memory::desc(dst_dims, dst_usr_dt, dst_strides);

  // STEP4: create memory
  auto m1_usr_m = make_onednn_memory(m1_usr_md, engine, m1.data_ptr());
  auto m2_usr_m = make_onednn_memory(m2_usr_md, engine, m2.data_ptr());
  auto dst_usr_m = make_onednn_memory(dst_usr_md, engine, dst.data_ptr());

  auto expected_m1_md = matmul_pd.src_desc();
  auto expected_m2_md = matmul_pd.weights_desc();
  auto expected_dst_md = matmul_pd.dst_desc();

  dnnl::memory m1_m = m1_usr_m, m2_m = m2_usr_m, dst_m = dst_usr_m;
  at::Tensor m1_, m2_, dst_;

  if (attr.with_binary())
    attr.construct_post_binary(matmul_pd, args);

  size_t scratchpad_size = matmul_pd.scratchpad_desc().get_size();
  at::Tensor scratchpad_tensor = at::empty(
      {static_cast<int64_t>(scratchpad_size)},
      m1.options().dtype(at::kByte),
      std::nullopt);
  auto scratchpad_memory = make_onednn_memory(
      matmul_pd.scratchpad_desc(), engine, scratchpad_tensor.data_ptr());
  args.insert({DNNL_ARG_SCRATCHPAD, scratchpad_memory});

  args.insert({DNNL_ARG_SRC, m1_m});
  args.insert({DNNL_ARG_WEIGHTS, m2_m});
  args.insert({DNNL_ARG_DST, dst_m});
  if (with_bias) {
    auto bias_m = make_onednn_memory(bias_md, engine, b.data_ptr());
    args.insert({DNNL_ARG_BIAS, bias_m});
  }

  sycl::event matmul_event =
      dnnl::sycl_interop::execute(matmul_p, stream, args, deps);

  if (!dst.is_same(result))
    result.copy_(dst);

  return matmul_event;
}

sycl::event scaled_matmul(
    at::Tensor& result,
    const at::Tensor& mat1,
    const at::Tensor& mat2,
    const c10::optional<at::Tensor>& bias,
    const at::Tensor& scale_a,
    const at::Tensor& scale_b,
    Attr attr,
    const std::vector<sycl::event>& deps = {}) {
  at::Device cur_device = at::Device(at::kXPU, c10::xpu::current_device());
  dnnl::engine engine = GpuEngineManager::Instance().get_engine(cur_device);
  auto stream = GpuStreamManager::Instance().get_stream();

  // Validation checks have passed lets resize the output to actual size
  IntArrayRef mat1_sizes = mat1.sizes();
  IntArrayRef mat2_sizes = mat2.sizes();
  int64_t K = mat1_sizes[1], M = mat1_sizes[0], N = mat2_sizes[1];

  std::vector<int64_t> src_dims = {M, K};
  std::vector<int64_t> weight_dims = {K, N};
  std::vector<int64_t> dst_dims = {M, N};

  dnnl::memory src = at::native::onednn::make_onednn_memory(
      {src_dims,
       at::native::onednn::get_onednn_dtype(mat1),
       mat1.strides().vec()},
      engine,
      mat1.data_ptr());
  auto mat2_c = mat2.contiguous();
  dnnl::memory weight = at::native::onednn::make_onednn_memory(
      {weight_dims,
       at::native::onednn::get_onednn_dtype(mat2_c),
       mat2_c.strides().vec()},
      engine,
      mat2_c.data_ptr());
  dnnl::memory dst = at::native::onednn::make_onednn_memory(
      {dst_dims,
       at::native::onednn::get_onednn_dtype(result),
       result.strides().vec()},
      engine,
      result.data_ptr());
  bool with_bias = bias.has_value();
  dnnl::memory::desc bias_desc = dnnl::memory::desc();
  dnnl::memory onednn_bias;
  if (with_bias) {
    auto bias_value = bias.value();
    if (bias_value.dim() == 1) {
      TORCH_CHECK(
          bias_value.size(0) == N || bias_value.size(0) == 1,
          "matmul supports [n] or [1] when bias dim is 1 ...");
      auto b_reshape = bias_value.reshape({1, bias_value.size(0)});
      onednn_bias = at::native::onednn::make_onednn_memory(
          {b_reshape.sizes().vec(),
           at::native::onednn::get_onednn_dtype(b_reshape),
           b_reshape.strides().vec()},
          engine,
          b_reshape.data_ptr());
    } else {
      onednn_bias = at::native::onednn::make_onednn_memory(
          {bias_value.sizes().vec(),
           at::native::onednn::get_onednn_dtype(bias_value),
           bias_value.strides().vec()},
          engine,
          bias_value.data_ptr());
    }
  }

  dnnl::primitive_attr op_attr = dnnl::primitive_attr();
  op_attr.set_scales_mask(DNNL_ARG_SRC, scale_a.numel() == 1 ? 0 : 1 << 1);
  op_attr.set_scales_mask(DNNL_ARG_WEIGHTS, scale_b.numel() == 1 ? 0 : 1 << 1);
  op_attr.set_scratchpad_mode(dnnl::scratchpad_mode::user);
  // TODO: Remove this try/catch when oneDNN provides API to notify
  // framework whether current platform can run FP8 primitives.
  dnnl::matmul::primitive_desc primitive_desc;
  try {
    primitive_desc = with_bias ? dnnl::matmul::primitive_desc(
                                     engine,
                                     src.get_desc(),
                                     weight.get_desc(),
                                     onednn_bias.get_desc(),
                                     dst.get_desc(),
                                     op_attr)
                               : dnnl::matmul::primitive_desc(
                                     engine,
                                     src.get_desc(),
                                     weight.get_desc(),
                                     dst.get_desc(),
                                     op_attr);
  } catch (dnnl::error& e) {
    if (e.status == dnnl_unimplemented)
      throw std::runtime_error("Onednn cannot create primitive.");
    // on any other error just re-throw
    throw;
  }
  auto primitive = dnnl::matmul(primitive_desc);

  // Prepare args and execute primitive
  dnnl::memory scratchpad = at::native::onednn::make_onednn_memory(
      primitive_desc.scratchpad_desc(), engine, nullptr);
  std::unordered_map<int, dnnl::memory> args;
  args.insert({DNNL_ARG_SRC, src});
  args.insert({DNNL_ARG_WEIGHTS, weight});
  args.insert({DNNL_ARG_DST, dst});
  args.insert({DNNL_ARG_SCRATCHPAD, scratchpad});
  if (with_bias) {
    args.insert({DNNL_ARG_BIAS, onednn_bias});
  }
  // auto sycl_queue = dnnl::sycl_interop::get_queue(stream);
  dnnl::memory src_scales_t = scale_a.numel() == 1
      ? at::native::onednn::make_onednn_memory(
            {{1}, dnnl::memory::data_type::f32, {1}},
            engine,
            scale_a.data_ptr())
      : at::native::onednn::make_onednn_memory(
            {{scale_a.numel(), 1}, dnnl::memory::data_type::f32, {1}},
            engine,
            scale_a.data_ptr());
  args.insert({DNNL_ARG_ATTR_SCALES | DNNL_ARG_SRC, src_scales_t});
  dnnl::memory wei_scales_t = scale_b.numel() == 1
      ? at::native::onednn::make_onednn_memory(
            {{1}, dnnl::memory::data_type::f32, {1}},
            engine,
            scale_b.data_ptr())
      : at::native::onednn::make_onednn_memory(
            {{scale_b.numel(), 1}, dnnl::memory::data_type::f32, {1}},
            engine,
            scale_b.data_ptr());
  args.insert({DNNL_ARG_ATTR_SCALES | DNNL_ARG_WEIGHTS, wei_scales_t});

  auto matmul_forward = dnnl::matmul(primitive_desc);
  sycl::event matmul_fwd_event =
      dnnl::sycl_interop::execute(matmul_forward, stream, args);
  return matmul_fwd_event;
}

} // namespace at::native::onednn
