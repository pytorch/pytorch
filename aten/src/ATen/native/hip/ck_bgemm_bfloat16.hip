#undef __HIP_NO_HALF_CONVERSIONS__

#include <iostream>

#include <ATen/native/hip/ck_bgemm.h>
#include <ATen/native/hip/bgemm_kernels/bgemm_kernel_collection.h>

namespace at::native {

using BGEMMKernel_BFloat16 = std::function<void(CUDABLAS_BGEMM_ARGTYPES(at::BFloat16))>;


struct IntTupleHash {
  size_t operator()(const std::tuple<int, int, int>& t) const {
    auto hash1 = std::hash<int>{}(std::get<0>(t));
    auto hash2 = std::hash<int>{}(std::get<1>(t));
    auto hash3 = std::hash<int>{}(std::get<2>(t));
    return hash1 ^ hash2 ^ hash3;
  }
};

// Map for specific shape dispatching.
static const std::unordered_map<
    std::tuple<int, int, int>,
    BGEMMKernel_BFloat16,
    IntTupleHash>
    lookup_dispatch = {
        {{5120, 64, 8192},
         bgemm_kernel_bf16bf16bf16_256_128x128x64_32x32_2x2_8x32x1_8x32x1_1x16x1x16_4_Intrawave_v3}
    };


// This is the heuristic to choose a kernel based on inputs
BGEMMKernel_BFloat16 dispatch_bfloat16_bgemm(CUDABLAS_BGEMM_ARGTYPES(at::BFloat16)) {
  // Optional/future use: directly lookup shape tuples to map to instances
  /*
  auto it = lookup_dispatch.find({m, n, k});
  if (it != lookup_dispatch.end()) {
    return it->second;
  }
  */

  // B is A and A is B, so m<-->n
  // std::cout << "dispatch_bfloat16_bgemm: m=" << m << " n=" << n << " k=" << k << " num_batches=" << num_batches << " transa=" << transa << " transb=" << transb << std::endl;

  if (m <= 5120) {
    if (n <= 4) return bgemm_kernel_bf16bf16bf16_64_16x16x64_16x16_1x1_8x8x1_8x8x1_1x16x1x4_4_Intrawave_v1;
    else if (n <= 32) return bgemm_kernel_bf16bf16bf16_128_16x64x64_16x16_1x2_8x16x1_8x16x1_1x16x1x8_4_Intrawave_v2;
    else if (n <= 128) return bgemm_kernel_bf16bf16bf16_256_128x128x64_32x32_2x2_8x32x1_8x32x1_1x16x1x16_4_Intrawave_v3; // <512, <1024, <2048 missing
    else if (n <= 4096) return bgemm_kernel_bf16bf16bf16_256_224x256x64_16x16_7x8_8x32x1_8x32x1_1x16x1x16_4_Intrawave_v3;
  }
  else if (m <= 8192) {
    if (n <= 8) return bgemm_kernel_bf16bf16bf16_128_16x32x64_16x16_1x1_8x16x1_8x16x1_1x16x1x8_4_Intrawave_v1; // 3 options available, need to investigate
    if (n <= 32) return bgemm_kernel_bf16bf16bf16_128_16x64x64_16x16_1x2_8x16x1_8x16x1_1x16x1x8_4_Intrawave_v2;
    if (n <= 512) return bgemm_kernel_bf16bf16bf16_256_128x128x64_32x32_2x2_8x32x1_8x32x1_1x16x1x16_4_Intrawave_v3;
    if (n <= 4096) return bgemm_kernel_bf16bf16bf16_256_256x224x64_16x16_8x7_8x32x1_8x32x1_1x32x1x8_4_Intrawave_v3;
  }

  // Default instance
  return bgemm_kernel_bf16bf16bf16_256_256x224x64_16x16_8x7_8x32x1_8x32x1_1x32x1x8_4_Intrawave_v3;
}

template <>
void bgemm_internal_ck<at::BFloat16>(CUDABLAS_BGEMM_ARGTYPES(at::BFloat16)) {
  /*
  std::cerr << "bgemm_internal_ck: " << std::endl
   << "\t" << num_batches
   << "\t" << m
   << "\t" << n
   << "\t" << k << std::endl
   << "\t" << stridea
   << "\t" << strideb
   << "\t" << stridec
   << "\t" << lda
   << "\t" << ldb
   << "\t" << ldc << std::endl
   << "\t" << transa
   << "\t" << transb << std::endl
   << "\t" << alpha
   << "\t" << beta << std::endl;
  */

  BGEMMKernel_BFloat16 bgemm_impl = dispatch_bfloat16_bgemm(CUDABLAS_BGEMM_ARGS(at::BFloat16));
  bgemm_impl(CUDABLAS_BGEMM_ARGS(at::BFloat16));
}

} // namespace at::native
