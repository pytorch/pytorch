#include <ATen/ATen.h>
#include <ATen/core/Tensor.h>
#include <c10/core/ScalarType.h>

#include <ATen/native/mkldnn/xpu/detail/oneDNNContext.h>
#include <ATen/native/mkldnn/xpu/detail/LRUCache.h>
#include <ATen/native/mkldnn/xpu/detail/Utils.h>
#include <ATen/native/mkldnn/xpu/detail/DnnlExt.h>

#include <oneapi/dnnl/dnnl.hpp>
#include <cstdint>

#include <chrono>
#include <fstream>
#include <iostream>
#include <string>
#include <tuple>
#include <vector>

using namespace dnnl;
using namespace at::native::onednn;

#ifdef USE_PRIMITIVE_CACHE

namespace at::native::onednn {

class TimeLogger {
 public:
  // Static inline member, can be defined and initialized inside the class
  static inline std::vector<long long> starts;
  static inline std::vector<long long> phase0;
  static inline std::vector<long long> phase1;
  static inline std::vector<long long> phase2;

  // Static method to access the singleton instance
  static TimeLogger& get_instance() {
    static TimeLogger
        instance; // Static local variable to ensure single instance
    return instance;
  }

  void record_start() {
    // Get the current timestamp in nanoseconds
    auto now = std::chrono::high_resolution_clock::now();
    auto duration = now.time_since_epoch();
    auto nanoseconds =
        std::chrono::duration_cast<std::chrono::nanoseconds>(duration).count();

    // Store m, n, k, and timestamp in the static log_data
    starts.push_back(nanoseconds);
  }

  void record_phase0() {
    // Get the current timestamp in nanoseconds
    auto now = std::chrono::high_resolution_clock::now();
    auto duration = now.time_since_epoch();
    auto nanoseconds =
        std::chrono::duration_cast<std::chrono::nanoseconds>(duration).count();

    // Store m, n, k, and timestamp in the static log_data
    phase0.push_back(nanoseconds);
  }

  void record_phase1() {
    // Get the current timestamp in nanoseconds
    auto now = std::chrono::high_resolution_clock::now();
    auto duration = now.time_since_epoch();
    auto nanoseconds =
        std::chrono::duration_cast<std::chrono::nanoseconds>(duration).count();

    // Store m, n, k, and timestamp in the static log_data
    phase1.push_back(nanoseconds);
  }

  void record_phase2() {
    // Get the current timestamp in nanoseconds
    auto now = std::chrono::high_resolution_clock::now();
    auto duration = now.time_since_epoch();
    auto nanoseconds =
        std::chrono::duration_cast<std::chrono::nanoseconds>(duration).count();

    // Store m, n, k, and timestamp in the static log_data
    phase2.push_back(nanoseconds);
  }

  // Destructor to write the log data to the file only once when the program
  // exits
  ~TimeLogger() {
    std::ofstream file("host_time_log.txt", std::ios::app);

    if (!file) {
      std::cerr << "Error opening file!" << std::endl;
      return;
    }

    // Write all stored logs to the file
    for (size_t i = 0; i < starts.size(); i++) {
      auto s = starts[i];
      auto p0 = phase0[i];
      auto p1 = phase1[i];
      auto p2 = phase2[i];

      // Write the log to the file
      file << "host time: " << p0 - s << "," << p1 - p0 << "," << p2 - p1
           << "\n";
    }
  }

 private:
  // Private constructor to prevent direct instantiation (Singleton pattern)
  TimeLogger() {}

  // Prevent copying and assignment
  TimeLogger(const TimeLogger&) = delete;
  TimeLogger& operator=(const TimeLogger&) = delete;
};

inline Tensor resize_as_onednn_mat1(const Tensor& mat1, const Tensor& output) {
  auto output_ = output.flatten(0, -2);
  int n = output_.sizes()[1];
  auto sizes = mat1.sym_sizes().vec();
  sizes[sizes.size() - 1] = n;
  return output.view_symint(sizes);
}

static inline void set_quant_primitive_attr(
    primitive_attr& pattr,
    const Tensor& scale,
    const Tensor& zp,
    const int64_t group_size) {
  // set scale and zero point for matmul args
#ifdef USE_SCRATCHPAD_MODE
  pattr.set_scratchpad_mode(dnnl::scratchpad_mode::user);
#endif

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
        memory::data_type::u4);
  }
  pattr.set_fpmath_mode(dnnl::fpmath_mode::f16, true);
}

template <typename F>
static at::Tensor dnnl_matmul_w4a16_common(
    Tensor& result, // dst, [b, m, n]
    const Tensor& mat1_, // src, [b, m, k]
    const Tensor& mat2, // quantized weight, [K/8, N] transpose
    const c10::optional<Tensor>& bias,
    const Tensor& scale, // [k/group_size, n]
    const Tensor& zp, // [k/group_size, n/8]
    int64_t group_size,
    bool m2_trans,
    const c10::optional<Tensor>& g_idx,
    F pattr,
    Tensor res_flat,
    Tensor res1_flat) {
  // For GPTQ with desc_act=True scenario
  auto mat1 = g_idx.has_value() ? mat1_.index_select(-1, g_idx.value()) : mat1_;

  auto src_sz = mat1.sizes();
  auto o_sz = mat1.sizes().vec();
  auto b_sz = mat2.sizes();
  *(o_sz.end() - 1) = *(b_sz.end() - 1);
  result = at::empty(o_sz, mat1.options());

  const int m = std::reduce(
      src_sz.begin(), src_sz.end() - 1, 1, std::multiplies<int64_t>());
  const int n = b_sz[1]; // presume channel last format
  const int k = *(src_sz.end() - 1);

  // get device, engine, stream
  const int device_id = c10::xpu::current_device();
  at::Device curDevice = at::Device(at::kXPU, device_id);
  auto engine = GpuEngineManager::Instance().get_engine(curDevice);

  joint_dtypes jd;
  if (mat1.scalar_type() == at::ScalarType::Half) {
    jd = joint_dtypes::_f16_int4;
  } else if (mat1.scalar_type() == at::ScalarType::BFloat16) {
    jd = joint_dtypes::_bf16_int4;
  } else {
    TORCH_INTERNAL_ASSERT(
        false, "Unsupported data type for int4 matmul: ", mat1.scalar_type());
  }

  // TODO: add bias datatype according to onednn requirement
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

  const int64_t ldb = mat2.strides()[mat2.dim() - 1] * 8; // for int4 matmul
  const int64_t lda = mat1.strides()[mat1.dim() - 2];
  const int64_t ldc = result.strides()[result.dim() - 2];

  trans_type tt = trans_type::_nt; // only support nt for int4 matmul
  auto& matmul_ext = matmul_primitive_create_and_cache(
      jd, tt, b_type, m, n, k, lda, ldb, ldc, device_id, pattr);

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

          const uint64_t num_groups = (uint64_t)(k / group_size);
          memory zp_B_u4_m(
              {{num_groups, n}, memory::data_type::u4, {n, 1}},
              engine,
              zp.data_ptr());
          return zp_B_u4_m;
        });
  }

  if (res_flat.defined()) {
    matmul_ext.set_attribute(
        arg_off++,
        DNNL_ARG_ATTR_MULTIPLE_POST_OP(0) | DNNL_ARG_SRC_1,
        res_flat.data_ptr(),
        [&]() {
          return make_onednn_memory(
              get_onednn_md(res_flat), engine, res_flat.data_ptr());
        });
  }
  if (res1_flat.defined()) {
    matmul_ext.set_attribute(
        arg_off++,
        DNNL_ARG_ATTR_MULTIPLE_POST_OP(1) | DNNL_ARG_SRC_1,
        res1_flat.data_ptr(),
        [&]() {
          return make_onednn_memory(
              get_onednn_md(res1_flat), engine, res1_flat.data_ptr());
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

#ifdef USE_SCRATCHPAD_MODE
  int scratchpad_size = matmul_ext.get_scratchpad_size();
  Tensor scratchpad_tensor = at::empty(
      {scratchpad_size}, mat1.options().dtype(at::kByte), c10::nullopt);
  arg_handles.emplace_back(DNNL_ARG_SCRATCHPAD, scratchpad_tensor.data_ptr());
#endif

  auto strm = GpuStreamManager::Instance().get_stream();
  auto qint4_matmul_event = matmul_ext.execute(strm, engine, std::move(arg_handles), arg_off);

  return result;
}

static at::Tensor dnnl_matmul_w4a16(
    Tensor& result, // dst, [b, m, n]
    const Tensor& mat1, // src, [b, m, k]
    const Tensor& mat2, // quantized weight, [k, n] transpose
    const c10::optional<Tensor>& bias,
    const Tensor& scale, // [k/group_size, n]
    const Tensor& zp, // [k/group_size, n/8]
    int64_t group_size,
    bool m2_trans,
    const c10::optional<Tensor>& g_idx) {
  RECORD_FUNCTION("dnnl_matmul_w4a16", std::vector<c10::IValue>({mat1, mat2}));

  auto quant = [&](primitive_attr& pattr) {
    set_quant_primitive_attr(pattr, scale, zp, group_size);
  };

  result = dnnl_matmul_w4a16_common(
      result,
      mat1,
      mat2,
      bias,
      scale,
      zp,
      group_size,
      m2_trans,
      g_idx,
      quant,
      at::Tensor(),
      at::Tensor());

  return result;
}

static at::Tensor dnnl_matmul_w4a16_and_silu(
    Tensor& result, // dst, [b, M, N]
    const Tensor& mat1, // src, [b, M, K]
    const Tensor& mat2, // quantized weight, [K/8, N]
    const c10::optional<Tensor>& bias,
    const Tensor& scale, // [K/group_size, N]
    const Tensor& zp, // [k/group_size, N/8]
    int64_t group_size,
    bool m2_trans,
    const c10::optional<Tensor>& g_idx) {
  RECORD_FUNCTION(
      "dnnl_matmul_w4a16_and_silu", std::vector<c10::IValue>({mat1, mat2}));

  auto silu = [&](primitive_attr& pattr) {
    set_quant_primitive_attr(pattr, scale, zp, group_size);
    post_ops po;
    po.append_eltwise(algorithm::eltwise_swish, 1.f, 0.f);
    pattr.set_post_ops(po);
  };

  result = dnnl_matmul_w4a16_common(
      result,
      mat1,
      mat2,
      bias,
      scale,
      zp,
      group_size,
      m2_trans,
      g_idx,
      silu,
      at::Tensor(),
      at::Tensor());

  return result;
}

static at::Tensor dnnl_matmul_w4a16_and_resmul(
    Tensor& result, // dst, [b, M, N]
    const Tensor& mat1, // src, [b, M, K]
    const Tensor& mat2, // quantized weight, [K/8, N]
    const c10::optional<Tensor>& bias,
    const Tensor& scale, // [K/group_size, N]
    const Tensor& zp, // [k/group_size, N/8]
    const Tensor& res,
    int64_t group_size,
    bool m2_trans,
    const c10::optional<Tensor>& g_idx) {
  RECORD_FUNCTION(
      "dnnl_matmul_w4a16_and_resmul", std::vector<c10::IValue>({mat1, mat2}));

  auto res_flat = res.flatten(0, -2);
  auto resmul = [&](primitive_attr& pattr) {
    set_quant_primitive_attr(pattr, scale, zp, group_size);
    post_ops po;
    po.append_binary(algorithm::binary_mul, get_onednn_md(res_flat));
    pattr.set_post_ops(po);
  };

  result = dnnl_matmul_w4a16_common(
      result,
      mat1,
      mat2,
      bias,
      scale,
      zp,
      group_size,
      m2_trans,
      g_idx,
      resmul,
      res_flat,
      at::Tensor());

  return result;
}

static at::Tensor dnnl_matmul_w4a16_and_bias_gelu(
    Tensor& result, // dst, [b, M, N]
    const Tensor& mat1, // src, [b, M, K]
    const Tensor& mat2, // quantized weight, [K/8, N]
    const c10::optional<Tensor>& bias,
    const Tensor& scale, // [K/group_size, N]
    const Tensor& zp, // [k/group_size, N/8]
    int64_t group_size,
    c10::string_view approximate,
    bool m2_trans,
    const c10::optional<Tensor>& g_idx) {
  RECORD_FUNCTION(
      "dnnl_matmul_w4a16_and_bias_gelu",
      std::vector<c10::IValue>({mat1, mat2}));

  auto bias_gelu = [&](primitive_attr& pattr) {
    set_quant_primitive_attr(pattr, scale, zp, group_size);
    post_ops po;
    if (approximate == "none") {
      po.append_eltwise(algorithm::eltwise_gelu_erf, 1.f, 0.f);
    } else if (approximate == "tanh") {
      po.append_eltwise(algorithm::eltwise_gelu_tanh, 1.f, 0.f);
    } else {
      TORCH_INTERNAL_ASSERT(false, "Unsupported gelu algorithm: ", approximate);
    }
    pattr.set_post_ops(po);
  };

  result = dnnl_matmul_w4a16_common(
      result,
      mat1,
      mat2,
      bias,
      scale,
      zp,
      group_size,
      m2_trans,
      g_idx,
      bias_gelu,
      at::Tensor(),
      at::Tensor());

  return result;
}

static at::Tensor dnnl_matmul_w4a16_and_bias_resadd_resadd(
    Tensor& result, // dst, [b, M, N]
    const Tensor& mat1, // src, [b, M, K]
    const Tensor& mat2, // quantized weight, [K/8, N]
    const c10::optional<Tensor>& bias,
    const Tensor& scale, // [K/group_size, N]
    const Tensor& zp, // [k/group_size, N/8]
    const Tensor& res,
    const Tensor& res1,
    int64_t group_size,
    bool m2_trans,
    const c10::optional<Tensor>& g_idx) {
  RECORD_FUNCTION(
      "dnnl_matmul_w4a16_and_bias_resadd_resadd",
      std::vector<c10::IValue>({mat1, mat2}));

  auto res_flat = res.flatten(0, -2);
  auto res1_flat = res1.flatten(0, -2);
  auto bias_resadd_resadd = [&](primitive_attr& pattr) {
    set_quant_primitive_attr(pattr, scale, zp, group_size);
    post_ops po;
    po.append_binary(algorithm::binary_add, get_onednn_md(res_flat));
    po.append_binary(algorithm::binary_add, get_onednn_md(res1_flat));
    pattr.set_post_ops(po);
  };

  result = dnnl_matmul_w4a16_common(
      result,
      mat1,
      mat2,
      bias,
      scale,
      zp,
      group_size,
      m2_trans,
      g_idx,
      bias_resadd_resadd,
      res_flat,
      res1_flat);

  return result;
}

static at::Tensor dnnl_matmul_w4a16_and_silu_mul(
    Tensor& result, // dst, [b, M, N]
    const Tensor& mat1, // src, [b, M, K]
    const Tensor& mat2, // quantized weight, [K/8, N]
    const c10::optional<Tensor>& bias,
    const Tensor& scale, // [K/group_size, N]
    const Tensor& zp, // [k/group_size, N/8]
    const Tensor& res,
    int64_t group_size,
    bool m2_trans,
    const c10::optional<Tensor>& g_idx) {
  RECORD_FUNCTION(
      "dnnl_matmul_w4a16_and_silu_mul", std::vector<c10::IValue>({mat1, mat2}));

  auto res_flat = res.flatten(0, -2);
  auto silu_mul = [&](primitive_attr& pattr) {
    set_quant_primitive_attr(pattr, scale, zp, group_size);
    post_ops po;
    po.append_eltwise(algorithm::eltwise_swish, 1.f, 0.f);
    po.append_binary(algorithm::binary_mul, get_onednn_md(res_flat));
    pattr.set_post_ops(po);
  };

  auto matmul_ext = dnnl_matmul_w4a16_common(
      result,
      mat1,
      mat2,
      bias,
      scale,
      zp,
      group_size,
      m2_trans,
      g_idx,
      silu_mul,
      at::Tensor(),
      res_flat);

  return result;
}

static at::Tensor dnnl_matmul_w4a16_and_bias_silu_mul(
    Tensor& result, // dst, [b, M, N]
    const Tensor& mat1, // src, [b, M, K]
    const Tensor& mat2, // quantized weight, [K/8, N]
    const c10::optional<Tensor>& bias,
    const Tensor& scale, // [K/group_size, N]
    const Tensor& zp, // [k/group_size, N/8]
    const Tensor& res,
    int64_t group_size,
    bool m2_trans,
    const c10::optional<Tensor>& g_idx) {
  RECORD_FUNCTION(
      "dnnl_matmul_w4a16_and_bias_silu_mul",
      std::vector<c10::IValue>({mat1, mat2}));

  auto res_flat = res.flatten(0, -2);
  auto silu_mul_int4 = [&](primitive_attr& pattr) {
    set_quant_primitive_attr(pattr, scale, zp, group_size);
    post_ops po;
    po.append_eltwise(algorithm::eltwise_swish, 1.f, 0.f);
    po.append_binary(algorithm::binary_mul, get_onednn_md(res_flat));
    pattr.set_post_ops(po);
  };

  auto matmul_ext = dnnl_matmul_w4a16_common(
      result,
      mat1,
      mat2,
      bias,
      scale,
      zp,
      group_size,
      m2_trans,
      g_idx,
      silu_mul_int4,
      at::Tensor(),
      res_flat);

  return result;
}

static at::Tensor dnnl_matmul_w4a16_and_add(
    Tensor& result, // dst, [b, M, N]
    const Tensor& mat1, // src, [b, M, K]
    const Tensor& mat2, // quantized weight, [K/8, N]
    const c10::optional<Tensor>& bias,
    const Tensor& scale, // [K/group_size, N]
    const Tensor& zp, // [k/group_size, N/8]
    const Tensor& res,
    int64_t group_size,
    bool m2_trans,
    const c10::optional<Tensor>& g_idx) {
  RECORD_FUNCTION(
      "dnnl_matmul_w4a16_and_add", std::vector<c10::IValue>({mat1, mat2}));

  auto res_flat = res.flatten(0, -2);
  auto bias_add_int4 = [&](primitive_attr& pattr) {
    set_quant_primitive_attr(pattr, scale, zp, group_size);
    post_ops po;
    po.append_binary(algorithm::binary_add, get_onednn_md(res_flat));
    pattr.set_post_ops(po);
  };

  result = dnnl_matmul_w4a16_common(
      result,
      mat1,
      mat2,
      bias,
      scale,
      zp,
      group_size,
      m2_trans,
      g_idx,
      bias_add_int4,
      res_flat,
      at::Tensor());

  return result;
}

static at::Tensor dnnl_matmul_w4a16_and_bias_add(
    Tensor& result, // dst, [b, M, N]
    const Tensor& mat1, // src, [b, M, K]
    const Tensor& mat2, // quantized weight, [K/8, N]
    const c10::optional<Tensor>& bias,
    const Tensor& scale, // [K/group_size, N]
    const Tensor& zp, // [k/group_size, N/8]
    const Tensor& res,
    int64_t group_size,
    bool m2_trans,
    const c10::optional<Tensor>& g_idx) {
  RECORD_FUNCTION(
      "dnnl_matmul_w4a16_and_bias_add", std::vector<c10::IValue>({mat1, mat2}));

  auto res_flat = res.flatten(0, -2);
  auto bias_add_int4 = [&](primitive_attr& pattr) {
    set_quant_primitive_attr(pattr, scale, zp, group_size);
    post_ops po;
    po.append_binary(algorithm::binary_add, get_onednn_md(res_flat));
    pattr.set_post_ops(po);
  };

  result = dnnl_matmul_w4a16_common(
      result,
      mat1,
      mat2,
      bias,
      scale,
      zp,
      group_size,
      m2_trans,
      g_idx,
      bias_add_int4,
      res_flat,
      at::Tensor());

  return result;
}

} // namespace at::native::onednn

#endif // USE_PRIMITIVE_CACHE
