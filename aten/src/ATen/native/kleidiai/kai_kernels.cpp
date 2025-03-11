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

static inline size_t get_vec_per_thread(
    size_t totalVec,
    size_t totalThread,
    size_t minStep) {
  return kai_roundup((totalVec + totalThread - 1) / totalThread, minStep);
}

static void kai_quant_pack_lhs_int4_mm_groupwise(
    const Tensor& output,
    const Tensor& input,
    const Tensor& weight,
    const int64_t m,
    const int64_t n,
    const int64_t k,
    const int64_t bl) {
  // Kernel IDs for GEMM and GEMV
  constexpr kai_kernel_id gemm_id =
      kai_kernel_id::matmul_clamp_f32_qai8dxp4x8_qsi4c32p4x8_4x8x32_neon_i8mm;
  constexpr kai_kernel_id gemv_id = kai_kernel_id::
      matmul_clamp_f32_qai8dxp1x8_qsi4c32p8x8_1x8x32_neon_dotprod;

  // Get total threads and select kernel
  const int64_t total_threads = at::get_num_threads();
  auto kernel_packet = kai_select_groupwise_matmul_ukernel(gemv_id);
  if (cpuinfo_has_arm_i8mm() && m > 1) {
    kernel_packet = kai_select_groupwise_matmul_ukernel(gemm_id);
  }

  // Thread blocking parameters
  const int64_t n_step = kernel_packet.ukernel.get_n_step();
  const size_t mr = kernel_packet.ukernel.get_mr();
  const size_t kr = kernel_packet.ukernel.get_kr();
  const size_t sr = kernel_packet.ukernel.get_sr();

  const size_t lhs_packed_size =
      kernel_packet.kai_get_lhs_packed_size(m, k, mr, kr, sr);
  auto lhs_packed = std::make_unique<uint8_t[]>(lhs_packed_size);
  uint8_t* dst_act_mtx_f32 = reinterpret_cast<uint8_t*>(output.data_ptr());
  const uint8_t* lhs_native_mtx_f32 =
      reinterpret_cast<const uint8_t*>(input.data_ptr());
  const uint8_t* rhs_packed_mtx_qs4cx =
      reinterpret_cast<const uint8_t*>(weight.data_ptr());
  uint8_t* lhs_packed_base = lhs_packed.get();

  const size_t lhs_stride = k * sizeof(float);
  const size_t dst_stride = n * sizeof(float);
  constexpr size_t dst_stride_col = sizeof(float);

  // LHS quantization packing
  int64_t vec_per_thread = get_vec_per_thread(m, total_threads, mr);
  int64_t num_threads = (m + vec_per_thread - 1) / vec_per_thread;
  const size_t src_stride = vec_per_thread * lhs_stride;

  auto lhs_quant_pack = [=, &kernel_packet](int64_t thread_id) {
    const auto lhs_src_ptr = lhs_native_mtx_f32 + thread_id * src_stride;
    const int64_t m_idx = thread_id * vec_per_thread;
    auto lhs_packed_ptr = lhs_packed_base +
        kai_get_lhs_packed_offset_lhs_quant_pack_qai8dxp_f32(
                              m_idx, k, mr, kr, sr);
    const int64_t vec_num = (thread_id == num_threads - 1)
        ? (m - vec_per_thread * thread_id)
        : vec_per_thread;

    kernel_packet.kai_run_lhs_quant_pack(
        vec_num,
        k,
        mr,
        kr,
        sr,
        0,
        (const float*)lhs_src_ptr,
        lhs_stride,
        lhs_packed_ptr);
  };

  at::parallel_for(
      0, num_threads, /*grain_size=*/1, [&](int64_t begin, int64_t end) {
        for (int64_t thread_id = begin; thread_id < end; ++thread_id) {
          lhs_quant_pack(thread_id);
        }
      });

  // Matrix multiplication
  vec_per_thread = get_vec_per_thread(n, total_threads, n_step);
  num_threads = (n + vec_per_thread - 1) / vec_per_thread;

  auto mm = [=, &kernel_packet](int64_t thread_id) {
    const auto rhs_packed_ptr = rhs_packed_mtx_qs4cx +
        kernel_packet.ukernel.get_rhs_packed_offset(
            thread_id * vec_per_thread, k, bl);
    auto dst_ptr = dst_act_mtx_f32 +
        kernel_packet.ukernel.get_dst_offset(
            0, thread_id * vec_per_thread, dst_stride);
    const int64_t vec_num = (thread_id == num_threads - 1)
        ? (n - vec_per_thread * thread_id)
        : vec_per_thread;

    kernel_packet.ukernel.run_matmul(
        m,
        vec_num,
        k,
        bl,
        lhs_packed_base,
        rhs_packed_ptr,
        (float*)dst_ptr,
        dst_stride,
        dst_stride_col,
        -FLT_MAX,
        FLT_MAX);
  };

  at::parallel_for(
      0, num_threads, /*grain_size=*/1, [&](int64_t begin, int64_t end) {
        for (int64_t thread_id = begin; thread_id < end; ++thread_id) {
          mm(thread_id);
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
  constexpr kai_kernel_id gemm_id =
      kai_kernel_id::matmul_clamp_f32_qai8dxp4x8_qsi4cxp8x8_8x8x32_neon_i8mm;
  constexpr kai_kernel_id gemv_id =
      kai_kernel_id::matmul_clamp_f32_qai8dxp1x8_qsi4cxp8x8_1x8x32_neon_dotprod;

  // Get total threads and select kernel
  const int64_t total_threads = at::get_num_threads();
  auto kernel_packet = kai_select_channelwise_matmul_ukernel(gemv_id);
  if (cpuinfo_has_arm_i8mm() && m > 1) {
    kernel_packet = kai_select_channelwise_matmul_ukernel(gemm_id);
  }

  // Thread blocking parameters
  const int64_t n_step = kernel_packet.ukernel.get_n_step();
  const size_t mr = kernel_packet.ukernel.get_mr();
  const size_t kr = kernel_packet.ukernel.get_kr();
  const size_t sr = kernel_packet.ukernel.get_sr();

  const size_t lhs_packed_size =
      kernel_packet.kai_get_lhs_packed_size(m, k, mr, kr, sr);
  auto lhs_packed = std::make_unique<uint8_t[]>(lhs_packed_size);
  uint8_t* dst_act_mtx_f32 = reinterpret_cast<uint8_t*>(output.data_ptr());
  const uint8_t* lhs_native_mtx_f32 =
      reinterpret_cast<const uint8_t*>(input.data_ptr());
  const uint8_t* rhs_packed_mtx_qs4cx =
      reinterpret_cast<const uint8_t*>(weight.data_ptr());
  uint8_t* lhs_packed_base = lhs_packed.get();

  const size_t lhs_stride = k * sizeof(float);
  const size_t dst_stride = n * sizeof(float);
  constexpr size_t dst_stride_col = sizeof(float);

  // LHS quantization packing
  int64_t vec_per_thread = get_vec_per_thread(m, total_threads, mr);
  int64_t num_threads = (m + vec_per_thread - 1) / vec_per_thread;
  const size_t src_stride = vec_per_thread * lhs_stride;

  auto lhs_quant_pack = [=, &kernel_packet](int64_t thread_id) {
    const auto lhs_src_ptr = lhs_native_mtx_f32 + thread_id * src_stride;
    const int64_t m_idx = thread_id * vec_per_thread;
    auto lhs_packed_ptr = lhs_packed_base +
        kai_get_lhs_packed_offset_lhs_quant_pack_qai8dxp_f32(
                              m_idx, k, mr, kr, sr);
    const int64_t vec_num = (thread_id == num_threads - 1)
        ? (m - vec_per_thread * thread_id)
        : vec_per_thread;

    kernel_packet.kai_run_lhs_quant_pack(
        vec_num,
        k,
        mr,
        kr,
        sr,
        0,
        (const float*)lhs_src_ptr,
        lhs_stride,
        lhs_packed_ptr);
  };

  at::parallel_for(
      0, num_threads, /*grain_size=*/1, [&](int64_t begin, int64_t end) {
        for (int64_t thread_id = begin; thread_id < end; ++thread_id) {
          lhs_quant_pack(thread_id);
        }
      });

  // Matrix multiplication
  vec_per_thread = get_vec_per_thread(n, total_threads, n_step);
  num_threads = (n + vec_per_thread - 1) / vec_per_thread;

  auto mm = [=, &kernel_packet](int64_t thread_id) {
    const auto rhs_packed_ptr = rhs_packed_mtx_qs4cx +
        kernel_packet.ukernel.get_rhs_packed_offset(
            thread_id * vec_per_thread, k);
    auto dst_ptr = dst_act_mtx_f32 +
        kernel_packet.ukernel.get_dst_offset(
            0, thread_id * vec_per_thread, dst_stride);
    const int64_t vec_num = (thread_id == num_threads - 1)
        ? (n - vec_per_thread * thread_id)
        : vec_per_thread;

    kernel_packet.ukernel.run_matmul(
        m,
        vec_num,
        k,
        lhs_packed_base,
        rhs_packed_ptr,
        (float*)dst_ptr,
        dst_stride,
        dst_stride_col,
        -FLT_MAX,
        FLT_MAX);
  };

  at::parallel_for(
      0, num_threads, /*grain_size=*/1, [&](int64_t begin, int64_t end) {
        for (int64_t thread_id = begin; thread_id < end; ++thread_id) {
          mm(thread_id);
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
