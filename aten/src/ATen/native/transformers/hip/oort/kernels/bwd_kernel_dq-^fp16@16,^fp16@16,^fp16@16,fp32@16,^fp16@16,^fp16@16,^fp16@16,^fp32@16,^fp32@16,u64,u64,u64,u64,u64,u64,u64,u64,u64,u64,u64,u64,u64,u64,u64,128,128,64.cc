#define INCBIN_PREFIX g_oort_kernel_for_shim_
#define INCBIN_STYLE INCBIN_STYLE_SNAKE
#include <incbin.h>
#include "oort_kernel.h"
#include "bwd_kernel_dq.h"

INCBIN(bwd_kernel_dq__Pfp16A16_Pfp16A16_Pfp16A16_fp32A16_Pfp16A16_Pfp16A16_Pfp16A16_Pfp32A16_Pfp32A16_u64_u64_u64_u64_u64_u64_u64_u64_u64_u64_u64_u64_u64_u64_u64_128_128_64, "/docker_m/oort/build/bwd_kernel_dq-^fp16@16,^fp16@16,^fp16@16,fp32@16,^fp16@16,^fp16@16,^fp16@16,^fp32@16,^fp32@16,u64,u64,u64,u64,u64,u64,u64,u64,u64,u64,u64,u64,u64,u64,u64,128,128,64.hsaco");

namespace oort {

template<> hipError_t
bwd_kernel_dq<128 /* BLOCK_M */,
              128 /* BLOCK_DMODEL */,
              64 /* BLOCK_N */
                  >::operator()(dim3 grid, dim3 block, const void* Q,
                       const void* K,
                       const void* V,
                       float sm_scale,
                       const void* Out,
                       const void* DO,
                       const void* DQ,
                       const float* L,
                       const float* D,
                       uint64_t stride_qz,
                       uint64_t stride_qh,
                       uint64_t stride_qm,
                       uint64_t stride_qk,
                       uint64_t stride_kz,
                       uint64_t stride_kh,
                       uint64_t stride_kn,
                       uint64_t stride_kk,
                       uint64_t stride_vz,
                       uint64_t stride_vh,
                       uint64_t stride_vk,
                       uint64_t stride_vn,
                       uint64_t Z,
                       uint64_t H,
                       uint64_t N_CTX, hipStream_t stream) {
  static oort::OortKernel kernel("bwd_kernel_dq_0d1d2d3d4d5d6d7d8d91011121314151617181920212223",
                                 g_oort_kernel_for_shim_bwd_kernel_dq__Pfp16A16_Pfp16A16_Pfp16A16_fp32A16_Pfp16A16_Pfp16A16_Pfp16A16_Pfp32A16_Pfp32A16_u64_u64_u64_u64_u64_u64_u64_u64_u64_u64_u64_u64_u64_u64_u64_128_128_64_data,
                                 32768);
  std::vector<void*> args = { static_cast<void*>(&Q),
                              static_cast<void*>(&K),
                              static_cast<void*>(&V),
                              static_cast<void*>(&sm_scale),
                              static_cast<void*>(&Out),
                              static_cast<void*>(&DO),
                              static_cast<void*>(&DQ),
                              static_cast<void*>(&L),
                              static_cast<void*>(&D),
                              static_cast<void*>(&stride_qz),
                              static_cast<void*>(&stride_qh),
                              static_cast<void*>(&stride_qm),
                              static_cast<void*>(&stride_qk),
                              static_cast<void*>(&stride_kz),
                              static_cast<void*>(&stride_kh),
                              static_cast<void*>(&stride_kn),
                              static_cast<void*>(&stride_kk),
                              static_cast<void*>(&stride_vz),
                              static_cast<void*>(&stride_vh),
                              static_cast<void*>(&stride_vk),
                              static_cast<void*>(&stride_vn),
                              static_cast<void*>(&Z),
                              static_cast<void*>(&H),
                              static_cast<void*>(&N_CTX) };
  return kernel.invoke(grid, block, args, stream);
}

}

