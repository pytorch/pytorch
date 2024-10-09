#include <c10/xpu/XPUFunctions.h>

#include <ATen/native/mkldnn/xpu/detail/Attr.h>
#include <ATen/native/mkldnn/xpu/detail/Utils.h>

#include <oneapi/dnnl/dnnl.hpp>
#include <cstdint>


namespace at::native::onednn {

inline Tensor resize_as_onednn_mat1(const Tensor& mat1, const Tensor& output) {
  auto output_ = output.flatten(0, -2);
  int n = output_.sizes()[1];
  auto sizes = mat1.sym_sizes().vec();
  sizes[sizes.size() - 1] = n;
  return output.view_symint(sizes);
}

sycl::event woq_matmul_int4(
    Tensor& result, // dst, [M, N]
    const Tensor& mat1_, // src, [M, K]
    const Tensor& mat2_, // quantized weight, [K/8, N]
    const Tensor& scale, // [K/group_size, N]
    const Tensor& zp, // [k/group_size, N/8]
    int64_t group_size,
    bool m2_trans,
    Attr attr,
    const c10::optional<Tensor>& g_idx,
    const std::vector<sycl::event>& deps,
    Tensor b_raw = at::Tensor()) {
  Tensor mat1;
  if (g_idx.has_value()) {
    mat1 = mat1_.index_select(-1, g_idx.value()).flatten(0, -2);
  } else {
    mat1 = mat1_.flatten(0, -2);
  }
  auto mat2 = mat2_.flatten(0, -2);
  int m = mat1.sizes()[0];
  int n = mat2.sizes()[1];
  int k = mat1.sizes()[1];
  result = at::empty({m, n}, mat1_.options());
  size_t dims = result.dim();
  TORCH_CHECK(
      dims == 2 || dims == 3,
      "oneDNN matmul only works with 2D or 3D, got ",
      dims);
  TORCH_CHECK(
      dims == mat1.dim() && dims == mat2.dim(),
      "oneDNN input matrixes must have the same ranks");
  TORCH_CHECK(result.defined(), "oneDNN matmul result should be defined");

  at::Device curDevice = at::Device(at::kXPU, at::xpu::current_device());
  auto engine = GpuEngineManager::Instance().get_engine(curDevice);
  auto engine_index = curDevice.index();
  auto stream = GpuStreamManager::Instance().get_stream();

  // make them all contiguous
  Tensor m1 = is_onednn_matmul_strides(mat1) ? mat1 : mat1.contiguous();
  Tensor m2 = is_onednn_matmul_strides(mat2) ? mat2 : mat2.contiguous();
  Tensor scale_ = is_onednn_matmul_strides(scale) ? scale : scale.contiguous();
  Tensor zp_ = is_onednn_matmul_strides(zp) ? zp : zp.contiguous();
  Tensor dst = is_onednn_matmul_strides(result, true) ? result : result.contiguous();

  m = dst.size(-2);
  n = dst.size(-1);
  k = m1.size(-1);
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
  Tensor b = b_raw;
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
          are_expandable({mb, m, n}, b.sizes()),
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

  // bias is fused in post-op for quantized path
  b = b.contiguous(); // avoid reorder 2 times

  // torch tensor of int32 dtype, corresponding to oneDNN s32
  auto m1_usr_dt = get_onednn_dtype(m1); // half <==> f16
  //   auto m2_usr_dt = dnnl::memory::data_type::s4; // int32, representing 8xint4
  auto m2_usr_dt = get_onednn_dtype(m2);
  auto scale_user_dt = get_onednn_dtype(scale_); // half <==> fp16
  //   auto zp_user_dt = dnnl::memory::data_type::s4; // int32, representing 8xint4
  auto zp_user_dt = get_onednn_dtype(zp_);
  auto dst_usr_dt = get_onednn_dtype(dst); // half <==> f16

  auto m1_dt = m1_usr_dt;
  auto m2_dt = dnnl::memory::data_type::u4;
  auto scale_dt = scale_user_dt;
  auto zp_dt = dnnl::memory::data_type::u4;
  auto dst_dt = dst_usr_dt;
  dnnl::memory::data_type bias_dt;

  dnnl::memory::desc m1_md, m1_usr_md;
  dnnl::memory::desc m2_md, m2_usr_md;
  dnnl::memory::desc scale_md, scale_usr_md;
  dnnl::memory::desc zp_md, zp_usr_md;
  dnnl::memory::desc dst_md, dst_usr_md;
  dnnl::memory::desc b_md;

  // STEP1: create dnnl::memory desc
  dnnl::memory::dims m1_dims, m2_dims, m2_usr_dims, scale_dims, zp_dims, zp_usr_dims,
      dst_dims, bias_dims;
  dnnl::memory::dims m1_strides, m2_strides, m2_usr_strides, scale_strides,
      zp_strides, zp_usr_strides, dst_strides, bias_strides;

  const uint64_t num_groups = (uint64_t)(k / group_size);
  const uint64_t compressed_n = (uint64_t)(n / 8);
  const uint64_t compressed_k = (uint64_t)(k / 8);

  m2_usr_dims = {compressed_k, n};
  scale_dims = {num_groups, n};
  zp_dims = {1};
  zp_usr_dims = {1};

  m2_usr_strides = {1, compressed_k};
  scale_strides = {scale_.stride(1), scale_.stride(0)};
  zp_strides = {1};
  zp_usr_strides = {1};

  if (dims == 2) {
    m1_dims = {m, k};
    m2_dims = {k, n};
    dst_dims = {m, n};

    m1_strides = {m1.stride(0), m1.stride(1)};
    m2_strides = {n, 1};
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

  dnnl::post_ops po = attr.extract_post_ops(dst);

  dnnl::matmul matmul_p;
  dnnl::matmul::primitive_desc matmul_pd;

  m1_usr_md = dnnl::memory::desc(m1_dims, m1_usr_dt, m1_strides);
  m2_usr_md = dnnl::memory::desc(m2_usr_dims, m2_usr_dt, m2_usr_strides);
  scale_usr_md = dnnl::memory::desc(scale_dims, scale_user_dt, scale_strides);
  zp_usr_md = dnnl::memory::desc(zp_usr_dims, zp_user_dt, zp_usr_strides);
  dst_usr_md = dnnl::memory::desc(dst_dims, dst_usr_dt, dst_strides);
  // STEP4: create dnnl::memory
  auto m1_usr_m = make_onednn_memory(m1_usr_md, engine, m1.data_ptr());
  auto m2_usr_m = make_onednn_memory(m2_usr_md, engine, m2.data_ptr());


  void* handle_b = m2_usr_m.get_data_handle();
  dnnl::memory m2_u4_m(
      {{k, n}, dnnl::memory::data_type::u4, dnnl::memory::format_tag::ba},
      engine,
      handle_b);

  m1_md = dnnl::memory::desc(m1_dims, m1_dt, m1_strides);
  m2_md = dnnl::memory::desc(m2_dims, m2_dt, m2_strides);
  scale_md = dnnl::memory::desc(scale_dims, scale_dt, scale_strides);
  zp_md = dnnl::memory::desc(zp_dims, zp_dt, zp_strides);
  dst_md = dnnl::memory::desc(dst_dims, dst_dt, dst_strides);
  
  // STEP2: creat attribute
  dnnl::primitive_attr pattr;
  pattr.set_post_ops(po);

  pattr.set_scratchpad_mode(dnnl::scratchpad_mode::user);

  // Set scales with multiple scales along K dimension and with groups along
  // K.
  pattr.set_scales(
      DNNL_ARG_WEIGHTS,
      /* mask */ (1 << 0) + (1 << 1),
      {group_size, 1},
      scale_dt);
  // Set a single zero point with s8 data type.
  pattr.set_zero_points(
      DNNL_ARG_WEIGHTS,
      /* mask */ 0,
      {},
      dnnl::memory::data_type::s8);
  // Set fpmath mode with `apply_to_int=true` to apply fpmath mode behavior to
  // integral primitives (in this example, matmul).
  pattr.set_fpmath_mode(dnnl::fpmath_mode::f16, true);

  if (with_bias) {
    b_md = dnnl::memory::desc(bias_dims, bias_dt, bias_strides);
    matmul_pd = dnnl::matmul::primitive_desc(
        engine, m1_md, m2_u4_m.get_desc(), b_md, dst_md, pattr);
  } else {
    matmul_pd = dnnl::matmul::primitive_desc(
        engine, m1_md, m2_u4_m.get_desc(), dst_md, pattr);
  }

  matmul_p = dnnl::matmul(matmul_pd);

  auto dst_usr_m = make_onednn_memory(dst_usr_md, engine, dst.data_ptr());
  auto scale_usr_m = make_onednn_memory(scale_usr_md, engine, scale.data_ptr());
  auto zp_usr_m = make_onednn_memory(zp_usr_md, engine, zp.data_ptr());

  dnnl::memory m1_m = m1_usr_m, m2_m = m2_u4_m, dst_m = dst_usr_m;
  dnnl::memory scale_m = scale_usr_m; // zp_m = zp_u4_m;
  Tensor m1_, m2_, zp_new, dst_;


  int scratchpad_size = matmul_pd.scratchpad_desc().get_size();
  Tensor scratchpad_tensor = at::empty(
      {scratchpad_size}, m1.options().dtype(at::kByte), c10::nullopt);
  auto scratchpad_memory = make_onednn_memory(
      matmul_pd.scratchpad_desc(), engine, scratchpad_tensor.data_ptr());
  args.insert({DNNL_ARG_SCRATCHPAD, scratchpad_memory});

  if (attr.with_binary())
    attr.construct_post_binary(matmul_pd, args);

  args.insert({DNNL_ARG_SRC, m1_m});
  args.insert({DNNL_ARG_WEIGHTS, m2_u4_m});
  args.insert({DNNL_ARG_DST, dst_m});
  if (b.defined()) {
    auto b_m = make_onednn_memory(b_md, engine, b.data_ptr());
    args.insert({DNNL_ARG_BIAS, b_m});
  }
  // add scale & zp
  // dnnl::memory zp_m({{1}, dnnl::memory::data_type::s8, {1}}, engine);
  args.insert({DNNL_ARG_ATTR_SCALES | DNNL_ARG_WEIGHTS, scale_m});
  args.insert({DNNL_ARG_ATTR_ZERO_POINTS | DNNL_ARG_WEIGHTS, zp_usr_m});

  sycl::event matmul_event = dnnl::sycl_interop::execute(matmul_p, stream, args, deps);
  if (!dst.is_same(result))
    result.copy_(dst);
  result = resize_as_onednn_mat1(mat1_, result);

  return matmul_event;
}



}