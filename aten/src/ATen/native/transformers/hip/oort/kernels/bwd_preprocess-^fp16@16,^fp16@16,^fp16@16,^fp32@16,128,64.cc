#define INCBIN_PREFIX g_oort_kernel_for_shim_
#define INCBIN_STYLE INCBIN_STYLE_SNAKE
#include <incbin.h>
#include "oort_kernel.h"
#include "bwd_preprocess.h"

INCBIN(bwd_preprocess__Pfp16A16_Pfp16A16_Pfp16A16_Pfp32A16_128_64, "/docker_m/oort/build/bwd_preprocess-^fp16@16,^fp16@16,^fp16@16,^fp32@16,128,64.hsaco");

namespace oort {

template<> hipError_t
bwd_preprocess<128 /* BLOCK_M */,
               64 /* D_HEAD */
                  >::operator()(dim3 grid, dim3 block, const void* Out,
                       const void* DO,
                       const void* NewDO,
                       const float* Delta, hipStream_t stream) {
  static oort::OortKernel kernel("bwd_preprocess_0d1d2d3d",
                                 g_oort_kernel_for_shim_bwd_preprocess__Pfp16A16_Pfp16A16_Pfp16A16_Pfp32A16_128_64_data,
                                 512);
  std::vector<void*> args = { static_cast<void*>(&Out),
                              static_cast<void*>(&DO),
                              static_cast<void*>(&NewDO),
                              static_cast<void*>(&Delta) };
  return kernel.invoke(grid, block, args, stream);
}

}

