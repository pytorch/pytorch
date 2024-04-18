// Pytorch also has an implementation of Philox RNG: https://github.com/pytorch/pytorch/blob/8ca3c881db3e3510fcb7725389f6a0633c9b992c/torch/csrc/jit/tensorexpr/cuda_random.h
#pragma once
// Philox CUDA.

#include <ATen/cuda/PhiloxUtils.cuh>

namespace pytorch_flash{

struct ull2 {
    unsigned long long x;
    unsigned long long y;
};

__forceinline__ __device__ uint2 mulhilo32(const unsigned int a, const unsigned int b) {
    uint2 *res;
    unsigned long long tmp;
    asm ("mul.wide.u32 %0, %1, %2;\n\t"
          : "=l"(tmp)
          : "r"(a), "r"(b));
    res = (uint2*)(&tmp);
    return *res;
}

__forceinline__ __device__ uint4 philox_single_round(const uint4 ctr, const uint2 key) {
    constexpr unsigned long kPhiloxSA = 0xD2511F53;
    constexpr unsigned long kPhiloxSB = 0xCD9E8D57;
    uint2 res0 = mulhilo32(kPhiloxSA, ctr.x);
    uint2 res1 = mulhilo32(kPhiloxSB, ctr.z);
    uint4 ret = {res1.y ^ ctr.y ^ key.x, res1.x, res0.y ^ ctr.w ^ key.y, res0.x};
    return ret;
}

__forceinline__ __device__ uint4 philox(unsigned long long seed,
                               unsigned long long subsequence,
                               unsigned long long offset) {
    constexpr unsigned long kPhilox10A = 0x9E3779B9;
    constexpr unsigned long kPhilox10B = 0xBB67AE85;
    uint2 key = reinterpret_cast<uint2&>(seed);
    uint4 counter;
    ull2 *tmp = reinterpret_cast<ull2*>(&counter);
    tmp->x = offset;
    tmp->y = subsequence;
    #pragma unroll
    for (int i = 0; i < 6; i++) {
        counter = philox_single_round(counter, key);
        key.x += (kPhilox10A);
        key.y += (kPhilox10B);
    }
    uint4 output = philox_single_round(counter, key);
    return output;
}

} // namespace flash
