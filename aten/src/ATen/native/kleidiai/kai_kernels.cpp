#include <ATen/native/kleidiai/kai_kernels.h>
#include <ATen/native/kleidiai/kai_pack.h>
#include <ATen/native/kleidiai/kai_ukernel_interface.h>

#include <ATen/Parallel.h>

#include <algorithm>
#include <cfloat>
#include <cmath>
#include <unordered_map>
#if AT_KLEIDIAI_ENABLED()
#include <cpuinfo.h>

namespace at::native::kleidiai {

void kai_pack_int4_rhs(
    const Tensor& weight_packed,
    const Tensor& weight,
    const Tensor& scales,
    const std::optional<Tensor>& bias,
    const int64_t n,
    const int64_t k,
    const int64_t bl) {
  // Prefer Channelwise kernel over Groupwise kernel for conflicting cases
  if (bl == k) {
    // Channelwise
    auto kernel_packet = kai_select_channelwise_matmul_ukernel(
        kai_kernel_id::
            matmul_clamp_f32_qai8dxp1x8_qsi4cxp8x8_1x8x32_neon_dotprod);
    auto& params = kernel_packet.rhs_pack_params;
    params.lhs_zero_point = 1;
    params.rhs_zero_point = 8;

    kai_pack_rhs_channelwise_int4<kai_matmul_ukernel_f32_qa8dxp_qs4cxp>(
        kernel_packet, weight_packed, weight, scales, bias, n, k);
  } else if (!(bl % 32) && !(k % bl)) {
    // Groupwise
    auto kernel_packet = kai_select_groupwise_matmul_ukernel(
        kai_kernel_id::
            matmul_clamp_f32_qai8dxp1x8_qsi4c32p8x8_1x8x32_neon_dotprod);

    const int64_t rhs_stride = kai_roundup(k, 2) / 2;
    const int64_t scale_stride = (kai_roundup(k, bl) / bl) * sizeof(uint16_t);
    auto& params = kernel_packet.rhs_pack_params;
    params.lhs_zero_point = 1;
    params.rhs_zero_point = 8;
    params.scale_dt = kai_datatype::kai_dt_bf16;

    kai_pack_rhs_groupwise_int4<kai_matmul_ukernel_f32_qa8dxp_qs4c32p>(
        kernel_packet,
        weight_packed,
        weight,
        scales,
        bias,
        n,
        k,
        bl,
        rhs_stride,
        scale_stride);
  }
}

size_t kai_pack_rhs_int4_size(
    const int64_t n,
    const int64_t k,
    const int64_t bl) {
  size_t packed_size = n * k;
  // Prefer Channelwise kernel over Groupwise kernel for conflicting cases
  if (bl == k) {
    // Channelwise
    auto kernel_packet = kai_select_channelwise_matmul_ukernel(
        kai_kernel_id::
            matmul_clamp_f32_qai8dxp1x8_qsi4cxp8x8_1x8x32_neon_dotprod);
    const auto& ukernel = kernel_packet.ukernel;
    const size_t nr = ukernel.get_nr();
    const size_t kr = ukernel.get_kr();
    const size_t sr = ukernel.get_sr();
    packed_size = kernel_packet.kai_get_rhs_packed_size(n, k, nr, kr, sr);
  } else if (!(bl % 32) && !(k % bl)) {
    // Groupwise
    auto kernel_packet = kai_select_groupwise_matmul_ukernel(
        kai_kernel_id::
            matmul_clamp_f32_qai8dxp1x8_qsi4c32p8x8_1x8x32_neon_dotprod);
    const auto& ukernel = kernel_packet.ukernel;
    const size_t nr = ukernel.get_nr();
    const size_t kr = ukernel.get_kr();
    const size_t sr = ukernel.get_sr();
    packed_size = kernel_packet.kai_get_rhs_packed_size(
        n, k, nr, kr, sr, bl, kai_datatype::kai_dt_bf16);
  }
  return packed_size;
}

static void matmul_channelwise(
    kai_matmul_ukernel_f32_qa8dxp_qs4cxp& kernel_packet,
    size_t m_increment,
    size_t m_start,
    size_t m_per_thread,
    size_t n_start,
    size_t n_per_thread,
    size_t n,
    size_t k,
    size_t mr,
    size_t nr,
    size_t kr,
    size_t sr,
    size_t dst_stride,
    size_t lhs_stride,
    uint8_t* lhs_native_mtx_f32,
    uint8_t* lhs_packed_mtx_qa8dx,
    uint8_t* rhs_packed_mtx_qs4cx,
    uint8_t* dst_act_mtx_f32) {
  for (size_t m0 = 0; m0 < m_per_thread; m0 += m_increment) {
    const float* src_ptr =
        (const float*)(lhs_native_mtx_f32 +
                       kai_get_lhs_offset_lhs_quant_pack_qai8dxp_f32(
                           m_start + m0, lhs_stride));
    void* lhs_packed_ptr =
        (void*)(lhs_packed_mtx_qa8dx +
                kai_get_lhs_packed_offset_lhs_quant_pack_qai8dxp_f32(
                    0, k, mr, kr, sr));
    const void* rhs_packed_ptr =
        (const void*)((const char*)rhs_packed_mtx_qs4cx +
                      kernel_packet.ukernel.get_rhs_packed_offset(n_start, k));
    float* dst_ptr = (float*)((uint8_t*)dst_act_mtx_f32 +
                              kernel_packet.ukernel.get_dst_offset(
                                  m_start + m0, n_start, dst_stride));

    // Quantize and pack the Input
    kernel_packet.kai_run_lhs_quant_pack(
        m_increment, k, mr, kr, sr, 0, src_ptr, lhs_stride, lhs_packed_ptr);

    // Run Matmul on Int4 packed weights and Quantized Packed Input
    kernel_packet.ukernel.run_matmul(
        m_increment,
        n_per_thread,
        k,
        lhs_packed_ptr,
        rhs_packed_ptr,
        dst_ptr,
        dst_stride,
        sizeof(float),
        -FLT_MAX,
        FLT_MAX);
  }
}

static void matmul_groupwise(
    const kai_matmul_clamp_f32_qai8dxp_qsi4c32p_ukernel& ukernel,
    const size_t m,
    const size_t num_n_per_thread,
    const size_t n_start,
    const size_t k,
    const size_t bl,
    const size_t dst_stride,
    const void* lhs_ptr,
    uint8_t* rhs_packed,
    uint8_t* dst_data) {
  const size_t rhs_packed_offset =
      ukernel.get_rhs_packed_offset(n_start, k, bl);
  const size_t dst_offset = ukernel.get_dst_offset(0, n_start, dst_stride);

  const void* rhs_ptr = (const void*)(rhs_packed + rhs_packed_offset);
  float* dst_ptr = (float*)((uint8_t*)dst_data + dst_offset);

  // Run Matmul on Int4 packed weights and Quantized Packed Input
  ukernel.run_matmul(
      m,
      num_n_per_thread,
      k,
      bl,
      lhs_ptr,
      rhs_ptr,
      dst_ptr,
      dst_stride,
      sizeof(float),
      -FLT_MAX,
      FLT_MAX);
}

struct ThreadDivision {
  int64_t num_threads_x;
  int64_t num_threads_y;
  bool use_gemm; // True if GEMM is selected, false if GEMV is used
};

inline static unsigned int round_down_to_power_of_2(unsigned int n) {
  n |= n >> 1;
  n |= n >> 2;
  n |= n >> 4;
  n |= n >> 8;
  n |= n >> 16;
  return n - (n >> 1);
}

inline static void adjust_max_threads(int64_t& max_threads) {
  // We would not like to round down to nearest power of 2 always
  // There can be possible thread split combination between powers of 2 for odd
  // shapes
  // TODO:: Decide better strategy based on hint of input and weight shapes
  max_threads = round_down_to_power_of_2(max_threads);
}

static std::pair<int64_t, int64_t> split_2d(const int64_t max_threads) {
  int64_t sqrt_threads = std::sqrt(max_threads);

  for (int64_t i = sqrt_threads; i >= 1; --i) {
    if (max_threads % i == 0) {
      return {i, max_threads / i};
    }
  }

  return {1, max_threads}; // Theres still a possibility of 1D blocking when
                           // calling GEMM kernel
}

inline static ThreadDivision get_thread_division(
    int64_t max_threads,
    const int64_t m,
    const int64_t n,
    const int64_t k,
    const int64_t gemm_m_step,
    const int64_t gemm_n_step,
    const int64_t gemv_m_step,
    const int64_t gemv_n_step) {
  adjust_max_threads(max_threads);
  ThreadDivision division{1, 1, false};

  // Split threads 2D for GEMM
  if (m % gemm_m_step == 0 && n % gemm_n_step == 0) {
    while (max_threads > 0) {
      auto [num_thread_y, num_thread_x] = split_2d(max_threads);
      if (m % num_thread_y == 0 && n % num_thread_x == 0) {
        int64_t m_per_thread = m / num_thread_y;
        int64_t n_per_thread = n / num_thread_x;
        if (m_per_thread % gemm_m_step == 0 &&
            n_per_thread % gemm_n_step == 0) {
          division = {num_thread_x, num_thread_y, true};
          return division;
        }
      }
      max_threads -= 2;
    }
  }
  // Split threads 1D for GEMV
  if (n % gemv_n_step == 0) {
    for (; max_threads > 0; max_threads -= 2) {
      if (n % max_threads == 0 && (n / max_threads) % gemv_n_step == 0) {
        division.num_threads_x = max_threads;
        return division;
      }
    }
  }
  return division;
}

static void kai_quant_pack_lhs_int4_mm_groupwise(
    const Tensor& output,
    const Tensor& input,
    const Tensor& weight,
    const int64_t m,
    const int64_t n,
    const int64_t k,
    const int64_t bl) {
  kai_kernel_id id = kai_kernel_id::
      matmul_clamp_f32_qai8dxp1x8_qsi4c32p8x8_1x8x32_neon_dotprod;
  if (cpuinfo_has_arm_i8mm() && m > 1) {
    id =
        kai_kernel_id::matmul_clamp_f32_qai8dxp4x8_qsi4c32p4x8_4x8x32_neon_i8mm;
  }
  auto kernel_packet = kai_select_groupwise_matmul_ukernel(id);

  const auto& ukernel = kernel_packet.ukernel;

  const size_t mr = ukernel.get_mr();
  const size_t kr = ukernel.get_kr();
  const size_t sr = ukernel.get_sr();
  const size_t n_step = ukernel.get_n_step();
  int64_t total_threads = at::get_num_threads();
  int64_t num_threads_x = 1;
  adjust_max_threads(total_threads);
  // Split threads 1D only for now
  if (n % n_step == 0) {
    for (; total_threads > 0; total_threads -= 2) {
      if (n % total_threads == 0 && (n / total_threads) % n_step == 0) {
        num_threads_x = total_threads;
        break;
      }
    }
  }

  const size_t num_n_per_thread = n / num_threads_x;

  const size_t dst_stride = n * sizeof(float);
  float* lhs = reinterpret_cast<float*>(input.data_ptr());
  uint8_t* rhs_packed_mtx_qs4cx = reinterpret_cast<uint8_t*>(weight.data_ptr());

  uint8_t* dst_act_mtx_f32 = reinterpret_cast<uint8_t*>(output.data_ptr());
  const size_t lhs_packed_size =
      kernel_packet.kai_get_lhs_packed_size(m, k, mr, kr, sr);
  auto lhs_packed = std::make_unique<uint8_t[]>(lhs_packed_size);

  // Quantize and pack the Input
  kernel_packet.kai_run_lhs_quant_pack(
      m,
      k,
      mr,
      kr,
      sr,
      0,
      (const float*)lhs,
      k * sizeof(float),
      (void*)lhs_packed.get());

  at::parallel_for(0, num_threads_x, 0, [&](int begin, int end) {
    for (const auto x : c10::irange(begin, end)) {
      matmul_groupwise(
          std::ref(ukernel),
          m,
          num_n_per_thread,
          x * num_n_per_thread,
          k,
          bl,
          dst_stride,
          lhs_packed.get(),
          rhs_packed_mtx_qs4cx,
          dst_act_mtx_f32);
    }
  });
}

static void kai_quant_pack_lhs_int4_mm_channelwise(
    const Tensor& output,
    const Tensor& input,
    const Tensor& weight,
    const int64_t m,
    const int64_t n,
    const int64_t k) {
  // Kernel IDs for GEMM and GEMV
  kai_kernel_id gemm_id =
      kai_kernel_id::matmul_clamp_f32_qai8dxp4x8_qsi4cxp8x8_8x8x32_neon_i8mm;
  kai_kernel_id gemv_id =
      kai_kernel_id::matmul_clamp_f32_qai8dxp1x8_qsi4cxp8x8_1x8x32_neon_dotprod;

  // Get the total number of threads available and choose GEMM or GEMV steps
  const int64_t total_threads = at::get_num_threads();
  auto gemm_kernel_packet = kai_select_channelwise_matmul_ukernel(gemv_id);
  if (cpuinfo_has_arm_i8mm()) {
    gemm_kernel_packet = kai_select_channelwise_matmul_ukernel(gemm_id);
  }
  auto gemv_kernel_packet = kai_select_channelwise_matmul_ukernel(gemv_id);

  // Retrieve m_step and n_step values from GEMM and GEMV kernels
  const int64_t gemm_m_step = gemm_kernel_packet.ukernel.get_m_step();
  const int64_t gemm_n_step = gemm_kernel_packet.ukernel.get_n_step();
  const int64_t gemv_m_step = gemv_kernel_packet.ukernel.get_m_step();
  const int64_t gemv_n_step = gemv_kernel_packet.ukernel.get_n_step();
  // Determine threading and kernel type
  ThreadDivision division = get_thread_division(
      total_threads,
      m,
      n,
      k,
      gemm_m_step,
      gemm_n_step,
      gemv_m_step,
      gemv_n_step);
  // Select appropriate kernel packet based on the chosen kernel type
  auto& kernel_packet =
      division.use_gemm ? gemm_kernel_packet : gemv_kernel_packet;

  // Thread blocking parameters
  const size_t mr = kernel_packet.ukernel.get_mr();
  const size_t nr = kernel_packet.ukernel.get_nr();
  const size_t kr = kernel_packet.ukernel.get_kr();
  const size_t sr = kernel_packet.ukernel.get_sr();
  const size_t m_increment = kernel_packet.ukernel.get_m_step();
  const size_t n_per_thread = n / division.num_threads_x;
  const size_t m_per_thread = m / division.num_threads_y;
  const int64_t num_threads = division.num_threads_y * division.num_threads_x;
  const size_t dst_stride = n * sizeof(float);
  const size_t lhs_stride = k * sizeof(float);

  const size_t lhs_packed_size =
      kernel_packet.kai_get_lhs_packed_size(m_increment, k, mr, kr, sr);

  uint8_t* dst_act_mtx_f32 = reinterpret_cast<uint8_t*>(output.data_ptr());
  uint8_t* lhs_native_mtx_f32 = reinterpret_cast<uint8_t*>(input.data_ptr());
  uint8_t* rhs_packed_mtx_qs4cx = reinterpret_cast<uint8_t*>(weight.data_ptr());
  auto lhs_packed = std::make_unique<uint8_t[]>(lhs_packed_size * num_threads);
  uint8_t* lhs_packed_base = lhs_packed.get();

  at::parallel_for(0, num_threads, 0, [&](int64_t begin, int64_t end) {
    for (const auto i : c10::irange(begin, end)) {
      size_t y = i / division.num_threads_x;
      size_t x = i % division.num_threads_x;
      uint8_t* lhs_packed_ptr =
          lhs_packed_base + (x + y * division.num_threads_x) * lhs_packed_size;
      matmul_channelwise(
          std::ref(kernel_packet),
          m_increment,
          y * m_per_thread,
          m_per_thread,
          x * n_per_thread,
          n_per_thread,
          n,
          k,
          mr,
          nr,
          kr,
          sr,
          dst_stride,
          lhs_stride,
          lhs_native_mtx_f32,
          lhs_packed_ptr,
          rhs_packed_mtx_qs4cx,
          dst_act_mtx_f32);
    }
  });
}

void kai_quant_pack_lhs_int4_mm(
    const Tensor& output,
    const Tensor& input,
    const Tensor& weight,
    const int64_t m,
    const int64_t n,
    const int64_t k,
    const int64_t bl) {
  // Prefer Channelwise kernel over Groupwise kernel for conflicting cases
  if (bl == k) {
    kleidiai::kai_quant_pack_lhs_int4_mm_channelwise(
        output, input, weight, m, n, k);
  } else if (!(bl % 32) && !(k % bl)) {
    kleidiai::kai_quant_pack_lhs_int4_mm_groupwise(
        output, input, weight, m, n, k, bl);
  }
}
} // namespace at::native::kleidiai
#endif
