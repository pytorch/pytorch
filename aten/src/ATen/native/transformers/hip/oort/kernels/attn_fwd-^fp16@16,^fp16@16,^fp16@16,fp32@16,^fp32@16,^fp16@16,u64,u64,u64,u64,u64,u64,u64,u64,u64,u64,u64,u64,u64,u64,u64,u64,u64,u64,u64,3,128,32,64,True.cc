#define INCBIN_PREFIX g_oort_kernel_for_shim_
#define INCBIN_STYLE INCBIN_STYLE_SNAKE
#include <incbin.h>
#include "oort_kernel.h"
#include "attn_fwd.h"

INCBIN(attn_fwd__Pfp16A16_Pfp16A16_Pfp16A16_fp32A16_Pfp32A16_Pfp16A16_u64_u64_u64_u64_u64_u64_u64_u64_u64_u64_u64_u64_u64_u64_u64_u64_u64_u64_u64_3_128_32_64_True, "/docker_m/oort/build/attn_fwd-^fp16@16,^fp16@16,^fp16@16,fp32@16,^fp32@16,^fp16@16,u64,u64,u64,u64,u64,u64,u64,u64,u64,u64,u64,u64,u64,u64,u64,u64,u64,u64,u64,3,128,32,64,True.hsaco");

namespace oort {

template<> hipError_t
attn_fwd<3 /* STAGE */,
         128 /* BLOCK_M */,
         32 /* BLOCK_DMODEL */,
         64 /* BLOCK_N */,
         true /* pre_load_v */
                  >::operator()(dim3 grid, dim3 block, const void* Q,
                       const void* K,
                       const void* V,
                       float sm_scale,
                       const float* M,
                       const void* Out,
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
                       uint64_t stride_oz,
                       uint64_t stride_oh,
                       uint64_t stride_om,
                       uint64_t stride_on,
                       uint64_t Z,
                       uint64_t H,
                       uint64_t N_CTX, hipStream_t stream) {
  static oort::OortKernel kernel("attn_fwd_0d1d2d3d4d5d6789101112131415161718192021222324",
                                 g_oort_kernel_for_shim_attn_fwd__Pfp16A16_Pfp16A16_Pfp16A16_fp32A16_Pfp32A16_Pfp16A16_u64_u64_u64_u64_u64_u64_u64_u64_u64_u64_u64_u64_u64_u64_u64_u64_u64_u64_u64_3_128_32_64_True_data,
                                 8256);
  std::vector<void*> args = { static_cast<void*>(&Q),
                              static_cast<void*>(&K),
                              static_cast<void*>(&V),
                              static_cast<void*>(&sm_scale),
                              static_cast<void*>(&M),
                              static_cast<void*>(&Out),
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
                              static_cast<void*>(&stride_oz),
                              static_cast<void*>(&stride_oh),
                              static_cast<void*>(&stride_om),
                              static_cast<void*>(&stride_on),
                              static_cast<void*>(&Z),
                              static_cast<void*>(&H),
                              static_cast<void*>(&N_CTX) };
  return kernel.invoke(grid, block, args, stream);
}

}

