#include <ATen/native/kleidiai/kai_kernels.h>
#include <ATen/native/kleidiai/kai_pack.h>
#include <ATen/native/kleidiai/kai_ukernel_interface.h>

#include <ATen/Parallel.h>

#include <algorithm>
#include <cfloat>
#if AT_KLEIDIAI_ENABLED()

namespace at::native::kleidiai {

Tensor kai_pack_int4_rhs(
    const Tensor& weight_packed,
    const Tensor& weight,
    const Tensor& scales,
    const Tensor& bias,
    const int64_t n,
    const int64_t k,
    const int64_t bl) {
  if (32 == bl) {
    // Groupwise 32
    auto kernel_packet = kai_select_groupwise_matmul_ukernel(
        kai_kernel_id::
            matmul_clamp_f32_qsi8d32p1x8_qsi4c32p4x8_1x4x32_neon_dotprod);
    return kai_pack_rhs_groupwise_int4<kai_matmul_ukernel_f32_qa8d32p_qs4c32p>(
        kernel_packet, weight_packed, weight, bias, n, k, bl);
  } else {
    // Channelwise
    auto kernel_packet = kai_select_channelwise_matmul_ukernel(
        kai_kernel_id::
            matmul_clamp_f32_qai8dxp1x8_qsi4cxp8x8_1x8x32_neon_dotprod);
    return kai_pack_rhs_channelwise_int4<kai_matmul_ukernel_f32_qa8dxp_qs4cxp>(
        kernel_packet, weight_packed, weight, scales, bias, n, k);
  }
}

size_t kai_pack_rhs_int4_size(
    const int64_t n,
    const int64_t k,
    const int64_t bl) {
  if (32 == bl) {
    // Groupwise 32
    auto kernel_packet = kai_select_groupwise_matmul_ukernel(
        kai_kernel_id::
            matmul_clamp_f32_qsi8d32p1x8_qsi4c32p4x8_1x4x32_neon_dotprod);
    const auto& ukernel = kernel_packet.ukernel;
    const size_t nr = ukernel.get_nr();
    const size_t kr = ukernel.get_kr();
    return kernel_packet.kai_get_rhs_packed_size(n, k, nr, kr, bl);
  } else {
    // Channelwise
    auto kernel_packet = kai_select_channelwise_matmul_ukernel(
        kai_kernel_id::
            matmul_clamp_f32_qai8dxp1x8_qsi4cxp8x8_1x8x32_neon_dotprod);
    const auto& ukernel = kernel_packet.ukernel;
    const size_t nr = ukernel.get_nr();
    const size_t kr = ukernel.get_kr();
    const size_t sr = ukernel.get_sr();
    return kernel_packet.kai_get_rhs_packed_size(n, k, nr, kr, sr);
  }
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
    uint8_t* lhs_native_mtx_f32,
    uint8_t* lhs_packed_mtx_qa8dx,
    uint8_t* rhs_packed_mtx_qs4cx,
    uint8_t* dst_act_mtx_f32) {
  const size_t dst_stride = n * sizeof(float);
  for (size_t m0 = 0; m0 < m_per_thread; m0 += m_increment) {
    const float* src_ptr =
        (const float*)(lhs_native_mtx_f32 +
                       kai_get_lhs_offset_lhs_quant_pack_qai8dxp_f32(
                           m_start + m0, k * sizeof(float)));
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
        m_increment,
        k,
        mr,
        kr,
        sr,
        0,
        src_ptr,
        k * sizeof(float),
        lhs_packed_ptr);

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
    const kai_matmul_clamp_f32_qsi8d32p_qsi4c32p_ukernel& ukernel,
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

static void kai_quant_pack_lhs_int4_mm_groupwise(
    const Tensor& output,
    const Tensor& input,
    const Tensor& weight,
    const int64_t m,
    const int64_t n,
    const int64_t k,
    const int64_t bl) {
  kai_kernel_id id = kai_kernel_id::
      matmul_clamp_f32_qsi8d32p1x8_qsi4c32p4x8_1x4x32_neon_dotprod;
  if (m > 1) {
    id = kai_kernel_id::
        matmul_clamp_f32_qsi8d32p4x8_qsi4c32p4x8_8x4x32_neon_i8mm;
  }

  auto kernel_packet = kai_select_groupwise_matmul_ukernel(id);

  const auto& ukernel = kernel_packet.ukernel;

  const size_t mr = ukernel.get_mr();
  const size_t kr = ukernel.get_kr();
  const size_t sr = ukernel.get_sr();

  const size_t num_threads_x = at::get_num_threads();

  const size_t num_n_per_thread = n / num_threads_x;

  const size_t dst_stride = n * sizeof(float);
  uint8_t* lhs = reinterpret_cast<uint8_t*>(input.data_ptr());
  uint8_t* rhs_packed_mtx_qs4cx = reinterpret_cast<uint8_t*>(weight.data_ptr());

  uint8_t* dst_act_mtx_f32 = reinterpret_cast<uint8_t*>(output.data_ptr());

  const float* src_ptr = (const float*)((const uint8_t*)lhs);
  const size_t lhs_packed_size =
      kernel_packet.kai_get_lhs_packed_size(m, k, bl, mr, kr, sr);
  void* lhs_packed = static_cast<void*>(aligned_alloc(64, lhs_packed_size));

  // Quantize and pack the Input for LLM
  kernel_packet.kai_run_lhs_quant_pack(
      m, k, bl, mr, kr, sr, 0, src_ptr, sizeof(float) * k, lhs_packed);

  at::parallel_for(0, num_threads_x, 0, [&](int begin, int end) {
    for (int64_t x = begin; x < end; x++) {
      matmul_groupwise(
          std::ref(ukernel),
          m,
          num_n_per_thread,
          x * num_n_per_thread,
          k,
          bl,
          dst_stride,
          lhs_packed,
          rhs_packed_mtx_qs4cx,
          dst_act_mtx_f32);
    }
  });

  free(lhs_packed);
}

static void kai_quant_pack_lhs_int4_mm_channelwise(
    const Tensor& output,
    const Tensor& input,
    const Tensor& weight,
    const int64_t m,
    const int64_t n,
    const int64_t k) {
  kai_kernel_id id =
      kai_kernel_id::matmul_clamp_f32_qai8dxp4x8_qsi4cxp8x8_8x8x32_neon_i8mm;
  const int64_t total_threads = at::get_num_threads();
  int64_t num_threads_y = total_threads;
  int64_t num_threads_x = total_threads;

  auto kernel_packet = kai_select_channelwise_matmul_ukernel(id);

  const bool can_gemm = (m % 8 == 0) ? true : false;

  if (!can_gemm) {
    id = kai_kernel_id::
        matmul_clamp_f32_qai8dxp1x8_qsi4cxp8x8_1x8x32_neon_dotprod;
    kernel_packet = kai_select_channelwise_matmul_ukernel(id);
    num_threads_y = 1;
    while (n % (num_threads_x * 8) != 0) {
      num_threads_x /= 2;
    }
  }
  const auto& ukernel = kernel_packet.ukernel;
  if (can_gemm) {
    auto m_step = ukernel.get_m_step();
    const int64_t max_threads_y = m / m_step;
    // Determine the number of threads for y
    num_threads_y = std::min(max_threads_y, num_threads_y);

    // Adjust num_threads_x and num_threads_y based on 2D blocking
    if (num_threads_y == total_threads && total_threads % 2 == 0) {
      num_threads_x = 2;
      num_threads_y /= num_threads_x;
    } else {
      if (total_threads % num_threads_y != 0) {
        num_threads_x = 1;
        // num_threads_y is odd and extra threads can be used in x direction
        if (total_threads % 2 == 0 && (num_threads_y - 1) % 2 == 0) {
          num_threads_x = total_threads / (num_threads_y - 1);
        }
      } else {
        num_threads_x = total_threads / num_threads_y;
      }
    }
  }

  const size_t mr = ukernel.get_mr();
  const size_t nr = ukernel.get_nr();
  const size_t kr = ukernel.get_kr();
  const size_t sr = ukernel.get_sr();

  const size_t m_increment = ukernel.get_m_step();
  assert((m % num_threads_y) == 0);
  assert((n % num_threads_x) == 0);
  const size_t n_per_thread = n / num_threads_x;
  const size_t m_per_thread = m / num_threads_y;
  const int64_t num_threads = num_threads_y * num_threads_x;

  const size_t lhs_packed_size =
      kernel_packet.kai_get_lhs_packed_size(m_increment, k, mr, kr, sr);

  uint8_t* dst_act_mtx_f32 = reinterpret_cast<uint8_t*>(output.data_ptr());
  uint8_t* lhs_native_mtx_f32 = reinterpret_cast<uint8_t*>(input.data_ptr());
  uint8_t* rhs_packed_mtx_qs4cx = reinterpret_cast<uint8_t*>(weight.data_ptr());
  uint8_t* lhs_packed =
      static_cast<uint8_t*>(aligned_alloc(64, lhs_packed_size * num_threads));

  at::parallel_for(0, num_threads, 0, [&](int64_t begin, int64_t end) {
    for (int64_t i = begin; i < end; ++i) {
      size_t y = i / num_threads_x;
      size_t x = i % num_threads_x;

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
          lhs_native_mtx_f32,
          lhs_packed + (x + y * num_threads_x) * lhs_packed_size,
          rhs_packed_mtx_qs4cx,
          dst_act_mtx_f32);
    }
  });

  free(lhs_packed);
}

void kai_quant_pack_lhs_int4_mm(
    const Tensor& output,
    const Tensor& input,
    const Tensor& weight,
    const int64_t m,
    const int64_t n,
    const int64_t k,
    const int64_t bl) {
  if (32 == bl) {
    kleidiai::kai_quant_pack_lhs_int4_mm_groupwise(
        output, input, weight, m, n, k, bl);
  } else {
    kleidiai::kai_quant_pack_lhs_int4_mm_channelwise(
        output, input, weight, m, n, k);
  }
}

} // namespace at::native::kleidiai
#endif
