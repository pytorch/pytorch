#include <c10/xpu/XPUFunctions.h>
#include <ATen/ATen.h>
#include <ATen/core/Tensor.h>
#include <c10/core/ScalarType.h>

#include <ATen/native/mkldnn/xpu/detail/oneDNNContext.h>
#include <ATen/native/mkldnn/xpu/detail/LRUCache.h>
#include <ATen/native/mkldnn/xpu/detail/Utils.h>
#include <ATen/native/mkldnn/xpu/detail/DnnlExt.h>

#include <oneapi/dnnl/dnnl.hpp>
#include <cstdint>
#include <optional>

using namespace dnnl;
using namespace at::native::onednn;

namespace at::native::onednn {

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

  if (zp.dim() == 1) {
    pattr.set_zero_points(
        DNNL_ARG_WEIGHTS,
        /* mask */ 0,
        {},
        memory::data_type::s8);
  } else {
    pattr.set_zero_points(
        DNNL_ARG_WEIGHTS,
        /* mask */ (1 << 0) + (1 << 1),
        {group_size, 1},
        memory::data_type::s8);
  }
}

void dnnl_matmul_w4a16_pri_cache(
    Tensor& result, 
    const Tensor& mat1_, 
    const Tensor& mat2, 
    const std::optional<Tensor>& bias,
    const Tensor& scale, 
    const Tensor& zp, 
    int64_t group_size,
    const std::optional<Tensor>& g_idx) {
  size_t dims = result.dim();
  TORCH_CHECK(
      dims == 2, "INT4 matmul at XPU only works with 2D input, got ", dims);
  TORCH_CHECK(result.defined(), "oneDNN matmul result should be defined");

  const int device_id = c10::xpu::current_device();
  at::Device cur_device = at::Device(at::kXPU, device_id);
  TORCH_CHECK(
      cur_device == mat1_.device(),
      "_weight_int4pack_mm_with_scales_and_zeros input should be on current device.");
  
  auto mat1 = g_idx.has_value() ? mat1_.index_select(-1, g_idx.value()) : mat1_;

  auto a_sz = mat1.sizes();
  auto c_sz = result.sizes();

  const int m = std::reduce(
      a_sz.begin(), a_sz.end() - 1, 1, std::multiplies<int64_t>());
  const int n = *(c_sz.end() - 1); 
  const int k = *(a_sz.end() - 1);

  // get device, engine, stream
  auto engine = GpuEngineManager::Instance().get_engine(cur_device);

  bias_type_t b_type;
  if (bias.has_value()) {
    auto& b = bias.value();
    const auto nuelm = b.numel();
    if (nuelm == 1) {
      b_type = bias_type_t::_scalar;
    } else if (nuelm == m * n) {
      b_type = bias_type_t::_mn;
    } else if (b.size(b.dim() - 1) == n && nuelm == n) {
      b_type = bias_type_t::_n;
    } else if (b.size(b.dim() - 1) == 1 && nuelm == m) {
      b_type = bias_type_t::_m;
    } else {
      TORCH_CHECK(0, "unsupported bias dim in matmul ...", b.sizes());
    }
  } else {
    b_type = bias_type_t::_none;
  }

  const int64_t ldb = mat2.strides()[mat2.dim() - 2] * 8; // for int4 matmul
  const int64_t lda = mat1.strides()[mat1.dim() - 2];
  const int64_t ldc = result.strides()[result.dim() - 2];

  trans_type_t tt = trans_type_t::_nt; // only support nt for int4 matmul

  joint_dtypes_t jd;
  if (mat1.scalar_type() == at::ScalarType::Half) {
    jd = joint_dtypes_t::_f16_int4;
  } else if (mat1.scalar_type() == at::ScalarType::BFloat16) {
    jd = joint_dtypes_t::_bf16_int4;
  } else {
    TORCH_INTERNAL_ASSERT(
        false, "Unsupported data type for int4 matmul: ", mat1.scalar_type());
  }
  
  auto f_attr = [&](primitive_attr& pattr) {
    pattr.set_scratchpad_mode(dnnl::scratchpad_mode::user);
    
    if (jd == joint_dtypes_t::_f16_int4) {
      pattr.set_fpmath_mode(dnnl::fpmath_mode::f16, true);
    } else if (jd == joint_dtypes_t::_bf16_int4) {
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

  auto& matmul_ext = matmul_primitive_create_and_cache(
      jd, tt, b_type, m, n, k, lda, ldb, ldc, device_id, f_attr);

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

  // set zp_md for symmetric quantization
  if (zp.dim() == 1) {
    matmul_ext.set_attribute(
        arg_off++,
        DNNL_ARG_ATTR_ZERO_POINTS | DNNL_ARG_WEIGHTS,
        zp.data_ptr(),
        [&]() {
          return make_onednn_memory(get_onednn_md(zp), engine, zp.data_ptr());
        });
  } else {
    // set zp_md for asymmetric quantization
    matmul_ext.set_attribute(
        arg_off++,
        DNNL_ARG_ATTR_ZERO_POINTS | DNNL_ARG_WEIGHTS,
        zp.data_ptr(),
        [&]() {
          int n = mat2.sizes()[1];
          int k = mat1.sizes()[1];

          const int64_t num_groups = (uint64_t)(k / group_size);
          memory zp_B_u4_m(
              {{num_groups, n}, memory::data_type::u4, {n, 1}},
              engine,
              zp.data_ptr());
          return zp_B_u4_m;
        });
  }

  // set general args
  std::vector<std::pair<int, void*>> arg_handles;
  arg_handles.reserve(8);

  arg_handles.emplace_back(DNNL_ARG_SRC, mat1.data_ptr());
  arg_handles.emplace_back(DNNL_ARG_WEIGHTS, mat2.data_ptr());
  arg_handles.emplace_back(DNNL_ARG_DST, result.data_ptr());
  if (bias.has_value()) {
    arg_handles.emplace_back(DNNL_ARG_BIAS, bias.value().data_ptr());
  }

  int scratchpad_size = matmul_ext.get_scratchpad_size();
  Tensor scratchpad_tensor = at::empty(
      {scratchpad_size}, mat1.options().dtype(at::kByte), c10::nullopt);
  arg_handles.emplace_back(DNNL_ARG_SCRATCHPAD, scratchpad_tensor.data_ptr());

  auto strm = GpuStreamManager::Instance().get_stream();
  auto qint4_matmul_event = matmul_ext.execute(strm, engine, std::move(arg_handles), arg_off);
}
} // namespace at::native::onednn
