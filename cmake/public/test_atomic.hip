#include <hip/hip_runtime.h>

static inline __device__ bool __hip_is_shared(const __attribute__((address_space(0))) void*) asm("llvm.amdgcn.is.shared");
static inline __device__ void atomicAddNoRet_impl(__attribute__((address_space(1))) float*, float) asm("llvm.amdgcn.global.atomic.fadd.p1f32.f32");
static inline __device__ void gpuAtomicAddNoReturn(float* address, float val) {
    using FP = __attribute__((address_space(0))) float*;
    using GP = __attribute__((address_space(1))) float*;
    using LP = __attribute__((address_space(3))) float*;
    if (!__hip_is_shared((FP)address))
        atomicAddNoRet_impl((GP)address, val);
    else
        __builtin_amdgcn_ds_faddf((LP)address, val, 0, 0, false);
}
static inline __device__ void atomicAddNoReturn(float *address, float val) { gpuAtomicAddNoReturn(address, val); }

__global__ void sum(float *a, int n)
{
    __shared__ float v[512];

    // Get our global thread ID
    int id = blockIdx.x*blockDim.x+threadIdx.x;
 
    // Make sure we do not go out of bounds
    if (id < n) atomicAddNoReturn(&v[threadIdx.x], a[id]);
}
