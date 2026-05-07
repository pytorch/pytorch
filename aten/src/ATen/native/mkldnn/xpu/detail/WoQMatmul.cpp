#include <c10/xpu/XPUFunctions.h>

#include <ATen/native/mkldnn/xpu/detail/Attr.h>
#include <ATen/native/mkldnn/xpu/detail/DnnlExt.h>
#include <ATen/native/mkldnn/xpu/detail/Utils.h>

#include <oneapi/dnnl/dnnl.hpp>
#include <cstdint>

namespace at::native::onednn {

void woq_matmul_int4_impl(
    Tensor& result,
    const Tensor& mat1_,
    const Tensor& mat2_,
    const Tensor& scale,
    const Tensor& zp,
    int64_t group_size) {
  auto& engine = GpuEngineManager::Instance().get_engine();
  auto& stream = GpuStreamManager::Instance().get_stream();

  Tensor m1 = mat1_;
  Tensor m2 = mat2_;
  Tensor scale_ = scale;
  Tensor zp_ = zp;
  Tensor dst = result;

  int m = m1.size(-2); // M
  int n = dst.size(-1); // N
  int k = m1.size(-1); // K

  // Construct usr md from input
  // xxx_usr_md would describe the real layout of inputs
  auto m1_usr_dt = get_onednn_dtype(m1); // e.g., half <==> f16
  auto m2_usr_dt = get_onednn_dtype(m2); // int32 tensor, pack 8 int4
  auto scale_usr_dt = get_onednn_dtype(scale_); // bf16
  auto zp_usr_dt = get_onednn_dtype(zp_); // s8 expected currently
  auto dst_usr_dt = get_onednn_dtype(dst); // bf16

  dnnl::memory::dims m1_usr_dims, m2_usr_dims, scale_usr_dims, zp_usr_dims,
      dst_usr_dims;
  dnnl::memory::dims m1_usr_strides, m2_usr_strides, scale_usr_strides,
      zp_usr_strides, dst_usr_strides;
  int compressed_k = k / 8;
  int num_groups = (int)(k / group_size);
  m1_usr_dims = {m, k};
  m1_usr_strides = {m1.stride(0), m1.stride(1)};
  m2_usr_dims = {compressed_k, n};
  m2_usr_strides = {1, compressed_k}; // k dim contiguous, 4bit pack into s32

  scale_usr_dims = {num_groups, n};
  scale_usr_strides = {n, 1};
  zp_usr_dims = {num_groups, n};
  zp_usr_strides = {n, 1};
  dst_usr_dims = {m, n};
  dst_usr_strides = {dst.stride(0), dst.stride(1)};

  dnnl::memory::desc m1_usr_md, m2_usr_md, scale_usr_md, zp_usr_md, dst_usr_md;

  m1_usr_md = dnnl::memory::desc(m1_usr_dims, m1_usr_dt, m1_usr_strides);
  m2_usr_md = dnnl::memory::desc(m2_usr_dims, m2_usr_dt, m2_usr_strides);
  scale_usr_md =
      dnnl::memory::desc(scale_usr_dims, scale_usr_dt, scale_usr_strides);
  zp_usr_md = dnnl::memory::desc(zp_usr_dims, zp_usr_dt, zp_usr_strides);
  dst_usr_md = dnnl::memory::desc(dst_usr_dims, dst_usr_dt, dst_usr_strides);

  // create usr memory
  auto dst_usr_m = make_onednn_memory(dst_usr_md, engine, dst.data_ptr());
  auto scale_usr_m = make_onednn_memory(scale_usr_md, engine, scale.data_ptr());
  auto zp_usr_m = make_onednn_memory(zp_usr_md, engine, zp.data_ptr());

  // Construct md for primitive creation
  // The xxx_md describes what kinds of matmul the oneDNN does.
  // The problem for this op is [m, k] x [k, n] => [m, n] matmul.
  auto m1_dt = m1_usr_dt; // bf16
  // Tell oneDNN the weight dtype we want manipulate is u4,
  // library needs infer how to unpack u4 data based on the m2_usr_md (s32).
  auto m2_dt = dnnl::memory::data_type::u4;
  auto scale_dt = scale_usr_dt; // bf16
  // Tell oneDNN the zp dtype we want manipulate is s8
  // library needs infer how to unpack s8 data based on the m2_usr_md.
  auto zp_dt = zp_usr_dt; // should be s8, currently
  auto dst_dt = dst_usr_dt;

  dnnl::memory::desc m1_md, m2_md, scale_md, zp_md, dst_md;
  dnnl::memory::dims m1_dims, m2_dims, scale_dims, zp_dims, dst_dims;
  dnnl::memory::dims m1_strides, m2_strides, scale_strides, zp_strides,
      dst_strides;

  m1_dims = m1_usr_dims; // {m, k}
  m1_strides = m1_usr_strides; // {k, 1}
  m2_dims = {k, n};
  m2_strides = {n, 1};
  scale_dims = scale_usr_dims; // {k//group_size, n}
  scale_strides = scale_usr_strides;
  zp_dims = zp_usr_dims;
  zp_strides = zp_usr_strides;
  dst_dims = dst_usr_dims;
  dst_strides = dst_usr_strides;

  m1_md = dnnl::memory::desc(m1_dims, m1_dt, m1_strides);
  m2_md = dnnl::memory::desc(m2_dims, m2_dt, m2_strides);
  scale_md = dnnl::memory::desc(scale_dims, scale_dt, scale_strides);
  zp_md = dnnl::memory::desc(zp_dims, zp_dt, zp_strides);
  dst_md = dnnl::memory::desc(dst_dims, dst_dt, dst_strides);

  std::unordered_map<int, dnnl::memory> args;

  dnnl::matmul matmul_p;
  dnnl::matmul::primitive_desc matmul_pd;

  auto m1_usr_m = make_onednn_memory(m1_usr_md, engine, m1.data_ptr());
  auto m2_usr_m = make_onednn_memory(m2_usr_md, engine, m2.data_ptr());

  void* handle_b = m2_usr_m.get_data_handle();
  // reinterpret m2_usr_memory as u4
  dnnl::memory m2_u4_m(
      {{k, n}, dnnl::memory::data_type::u4, dnnl::memory::format_tag::ba},
      engine,
      handle_b);

  dnnl::primitive_attr pattr;
  pattr.set_scratchpad_mode(dnnl::scratchpad_mode::user);
#if ONEDNN_SUPPORT_DETERMINISTIC
  if (at::globalContext().deterministicAlgorithms() ||
      at::globalContext().deterministicMkldnn()) {
    pattr.set_deterministic(true);
  }
#endif

  // Set scales with multiple scales along K dimension and with groups along  K.
  pattr.set_scales(
      DNNL_ARG_WEIGHTS,
      /* mask */ (1 << 0) + (1 << 1),
      {group_size, 1},
      scale_dt);
  // Set a single zero point with s8 data type.
  pattr.set_zero_points(
      DNNL_ARG_WEIGHTS,
      (1 << 0) + (1 << 1),
      {group_size, 1},
      dnnl::memory::data_type::s8);

  if (m1_dt == dnnl::memory::data_type::f16)
    pattr.set_fpmath_mode(dnnl::fpmath_mode::f16, true);
  else if (m1_dt == dnnl::memory::data_type::bf16)
    pattr.set_fpmath_mode(dnnl::fpmath_mode::bf16, true);

  matmul_pd = dnnl::matmul::primitive_desc(
      engine, m1_md, m2_u4_m.get_desc(), dst_md, pattr);
  matmul_p = dnnl::matmul(matmul_pd);

  dnnl::memory m1_m = m1_usr_m, m2_m = m2_u4_m, dst_m = dst_usr_m;
  dnnl::memory scale_m = scale_usr_m; // zp_m = zp_u4_m;
  Tensor m1_, m2_, zp_new, dst_;

  int scratchpad_size = matmul_pd.scratchpad_desc().get_size();
  Tensor scratchpad_tensor =
      at::empty({scratchpad_size}, m1.options().dtype(at::kByte), std::nullopt);
  auto scratchpad_memory = make_onednn_memory(
      matmul_pd.scratchpad_desc(), engine, scratchpad_tensor.data_ptr());
  args.insert({DNNL_ARG_SCRATCHPAD, scratchpad_memory});

  args.insert({DNNL_ARG_SRC, m1_m});
  args.insert({DNNL_ARG_WEIGHTS, m2_u4_m});
  args.insert({DNNL_ARG_DST, dst_m});
  args.insert({DNNL_ARG_ATTR_SCALES | DNNL_ARG_WEIGHTS, scale_m});
  args.insert({DNNL_ARG_ATTR_ZERO_POINTS | DNNL_ARG_WEIGHTS, zp_usr_m});
  dnnl::sycl_interop::execute(matmul_p, stream, args);
}

static inline void set_quant_primitive_attr(
    primitive_attr& pattr,
    const Tensor& scale,
    const Tensor& zp,
    const int64_t group_size) {
  // set scale and zero point for matmul args
  pattr.set_scales(
      DNNL_ARG_WEIGHTS,
      /* mask */ (1 << 0) + (1 << 1),
      {group_size, 1},
      get_onednn_dtype(scale));
  pattr.set_zero_points(
      DNNL_ARG_WEIGHTS,
      /* mask */ (1 << 0) + (1 << 1),
      {group_size, 1},
      memory::data_type::s8);
}

void woq_matmul_int4_impl_cache(
    Tensor& result,
    const Tensor& mat1,
    const Tensor& mat2,
    const Tensor& scale,
    const Tensor& zp,
    int64_t group_size) {
  auto a_sz = mat1.sizes();
  auto c_sz = result.sizes();

  const int m =
      std::reduce(a_sz.begin(), a_sz.end() - 1, 1, std::multiplies<int64_t>());
  const int n = *(c_sz.end() - 1);
  const int k = *(a_sz.end() - 1);

  const int64_t ldb = mat2.strides()[mat2.dim() - 2] * 8; // for int4 matmul
  const int64_t lda = mat1.strides()[mat1.dim() - 2];
  const int64_t ldc = result.strides()[result.dim() - 2];

  bias_type_t b_type = bias_type_t::none;
  trans_type_t tt = trans_type_t::nt; // only support nt for int4 matmul

  joint_dtypes_t jd;
  if (mat1.scalar_type() == at::ScalarType::Half) {
    jd = joint_dtypes_t::f16_int4;
  } else if (mat1.scalar_type() == at::ScalarType::BFloat16) {
    jd = joint_dtypes_t::bf16_int4;
  } else {
    TORCH_INTERNAL_ASSERT(
        false, "Unsupported data type for int4 matmul: ", mat1.scalar_type());
  }

  auto f_attr = [&](primitive_attr& pattr) {
    pattr.set_scratchpad_mode(dnnl::scratchpad_mode::user);

    if (jd == joint_dtypes_t::f16_int4) {
      pattr.set_fpmath_mode(dnnl::fpmath_mode::f16, true);
    } else if (jd == joint_dtypes_t::bf16_int4) {
      pattr.set_fpmath_mode(dnnl::fpmath_mode::bf16, true);
    }

    set_quant_primitive_attr(pattr, scale, zp, group_size);

#if ONEDNN_SUPPORT_DETERMINISTIC
    if (at::globalContext().deterministicAlgorithms() ||
        at::globalContext().deterministicMkldnn()) {
      pattr.set_deterministic(true);
    }
#endif
  };

  int64_t zp_group_size = group_size;
  auto device_id = c10::xpu::current_device();
  auto& matmul_ext = matmul_primitive_create_and_cache(
      jd,
      tt,
      b_type,
      m,
      n,
      k,
      lda,
      ldb,
      ldc,
      device_id,
      f_attr,
      group_size,
      zp_group_size);

  auto& engine = GpuEngineManager::Instance().get_engine();

  int arg_off = 0;
  // set scale and zero point for matmul args
  matmul_ext.set_attribute(
      arg_off++,
      DNNL_ARG_ATTR_SCALES | DNNL_ARG_WEIGHTS,
      scale.data_ptr(),
      [&]() {
        return make_onednn_memory(
            get_onednn_md(scale), engine, scale.data_ptr());
      });

  // set zp_md for asymmetric quantization
  matmul_ext.set_attribute(
      arg_off++,
      DNNL_ARG_ATTR_ZERO_POINTS | DNNL_ARG_WEIGHTS,
      zp.data_ptr(),
      [&]() {
        int num_groups = k / group_size;
        memory zp_usr_m(
            {{num_groups, n}, memory::data_type::s8, {n, 1}},
            engine,
            zp.data_ptr());
        return zp_usr_m;
      });

  // set general args
  std::vector<std::pair<int, void*>> arg_handles;
  arg_handles.reserve(8);

  arg_handles.emplace_back(DNNL_ARG_SRC, mat1.data_ptr());
  arg_handles.emplace_back(DNNL_ARG_WEIGHTS, mat2.data_ptr());
  arg_handles.emplace_back(DNNL_ARG_DST, result.data_ptr());

  int scratchpad_size = matmul_ext.get_scratchpad_size();
  Tensor scratchpad_tensor = at::empty(
      {scratchpad_size}, mat1.options().dtype(at::kByte), std::nullopt);
  arg_handles.emplace_back(DNNL_ARG_SCRATCHPAD, scratchpad_tensor.data_ptr());

  auto& strm = GpuStreamManager::Instance().get_stream();
  auto qint4_matmul_event =
      matmul_ext.execute(strm, engine, std::move(arg_handles), arg_off);
}

void woq_matmul_int4(
    Tensor& result, // torchao: [M, K], dtype: fp16,bf16
    const Tensor& mat1_, // torchao: [M, K], dtype: fp16,bf16
    const Tensor& mat2_, // torchao quantized weight, [K/8, N], dtype: uint4x8
    const Tensor& scale, // torchao: [K/group_size, N], dtype: fp16,bf16
    const Tensor& zp, // torchao: [K/group_size, N], dtype: int8
    int64_t group_size,
    bool pri_cache) {
  size_t dims = result.dim();
  TORCH_CHECK(
      dims == 2, "INT4 matmul at XPU only works with 2D input, got ", dims);
  TORCH_CHECK(result.defined(), "oneDNN matmul result should be defined");

  const int device_id = c10::xpu::current_device();
  at::Device cur_device = at::Device(at::kXPU, device_id);
  TORCH_CHECK(
      cur_device == mat1_.device(),
      "_weight_int4pack_mm_with_scales_and_zeros input should be on current device.");

  if (pri_cache) {
    woq_matmul_int4_impl_cache(result, mat1_, mat2_, scale, zp, group_size);
  } else {
    woq_matmul_int4_impl(result, mat1_, mat2_, scale, zp, group_size);
  }
}

} // namespace at::native::onednn
