namespace torch {
namespace jit {
namespace fuser {
namespace cuda {

// IO data structure for kernel code;
static auto code_template_tensor_struct = R"(
typedef unsigned char uint8_t;
typedef signed char int8_t;
typedef short int  int16_t;
typedef long long int int64_t;

template<typename T, int N>
struct Tensor {
  T& operator[](int64_t ind) {
    return data[ind];
  };

  T* data;
  int64_t size[N];
  int64_t stride[N];
};
)";

// Code support for FP16 __half type and intrinsics
static auto code_fp16_support = R"(
#define __HALF_TO_US(var) *(reinterpret_cast<unsigned short *>(&(var)))
#define __HALF_TO_CUS(var) *(reinterpret_cast<const unsigned short *>(&(var)))
struct __align__(2) __half {
  __host__ __device__ __half() { }
protected:
  unsigned short __x;
};

/* Definitions of intrinsics */
__device__ __half __float2half(const float f) {
  __half val;
  asm("{  cvt.rn.f16.f32 %0, %1;}\n" : "=h"(__HALF_TO_US(val)) : "f"(f));
  return val;
}
__device__ float __half2float(const __half h) {
  float val;
  asm("{  cvt.f32.f16 %0, %1;}\n" : "=f"(val) : "h"(__HALF_TO_CUS(h)));
  return val;
}
)";

// struct and code for functions that need random number generation
static auto code_random_number_gen = R"(
class Philox {
public:
  __device__ inline Philox(unsigned long long seed,
                           unsigned long long subsequence,
                           unsigned long long offset) {
    key.x = (unsigned int)seed;
    key.y = (unsigned int)(seed >> 32);
    counter = make_uint4(0, 0, 0, 0);
    counter.z = (unsigned int)(subsequence);
    counter.w = (unsigned int)(subsequence >> 32);
    STATE = 0;
    incr_n(offset / 4);
  }
  __device__ inline unsigned long operator()() {
    if(STATE == 0) {
      uint4 counter_ = counter;
      uint2 key_ = key;
      for(int i = 0; i < 9; i++) {
        counter_ = single_round(counter_, key_);
        key_.x += (kPhilox10A); key_.y += (kPhilox10B);
      }
      output = single_round(counter_, key_);
      incr();
    }
    unsigned long ret;
    switch(STATE) {
      case 0: ret = output.x; break;
      case 1: ret = output.y; break;
      case 2: ret = output.z; break;
      case 3: ret = output.w; break;
    }
    STATE = (STATE + 1) % 4;
    return ret;
  }
private:
  uint4 counter;
  uint4 output;
  uint2 key;
  unsigned int STATE;
  __device__ inline void incr_n(unsigned long long n) {
    unsigned int nlo = (unsigned int)(n);
    unsigned int nhi = (unsigned int)(n >> 32);
    counter.x += nlo;
    if (counter.x < nlo)
      nhi++;
    counter.y += nhi;
    if (nhi <= counter.y)
      return;
    if (++counter.z)
      return;
    ++counter.w;
  }
  __device__ inline void incr() {
    if (++counter.x)
      return;
    if (++counter.y)
      return;
    if (++counter.z)
      return;
    ++counter.w;
  }
  __device__ unsigned int mulhilo32(unsigned int a, unsigned int b,
                                    unsigned int *result_high) {
    *result_high = __umulhi(a, b);
    return a*b;
  }
  __device__ inline uint4 single_round(uint4 ctr, uint2 key) {
    unsigned int hi0;
    unsigned int hi1;
    unsigned int lo0 = mulhilo32(kPhiloxSA, ctr.x, &hi0);
    unsigned int lo1 = mulhilo32(kPhiloxSB, ctr.z, &hi1);
    uint4 ret = {hi1 ^ ctr.y ^ key.x, lo1, hi0 ^ ctr.w ^ key.y, lo0};
    return ret;
  }
  static const unsigned long kPhilox10A = 0x9E3779B9;
  static const unsigned long kPhilox10B = 0xBB67AE85;
  static const unsigned long kPhiloxSA = 0xD2511F53;
  static const unsigned long kPhiloxSB = 0xCD9E8D57;
};
// Inverse of 2^32.
#define M_RAN_INVM32 2.3283064e-10f
__device__  __inline__ float uniform(unsigned int x) {
  return x * M_RAN_INVM32;
}
)";

// Helper functions for Operations
static auto code_helper_funcs = R"(
__device__ int ceilDiv(const int a, const int b) {
  return (a + b - 1) / b;
}
__device__ float clamp(const float x, const float minv, const float maxv) {
  return x < minv ? minv : (x > maxv ? maxv : x);
}
__device__ float frac(const float x) {
  return x - truncf(x);
}
__device__ float gelu(const float x) {
  return x * normcdf(x);
}
__device__ float reciprocal(const float x) {
  return 1.f / x;
}
__device__ float relu(const float x) {
  return x <= 0.f ? 0.f : x;
}
__device__ float remainder(const float a, const float b) {
  return a - b * floorf(a / b);
}
__device__ float sigmoid(const float x) {
  return 1.f / (1.f + expf(-x));
}
__device__ float threshold(const float x, const float t, const float v) {
  return x <= t ? v : x;
}
__device__ float where(const bool c, const float a, const float b) {
  return c ? a : b;
}
__device__ float randLike(Philox rnd) {
  return uniform(rnd());
};
)";

/*
 *  EXAMPLE USAGE:
 *  blockReduceSum<X_THREADS, Y_THREADS, Z_THREADS>
 *    (output[output_index], inputs[input_index], [] __device__ (T& a, const T
 * b) { a += b; } );
 */
static auto code_template_block_reduction = R"(
// [Z,Y,X]_THREADS is the number of participating threads in the z, y, x
// dimension of the block. If set to 0 it means that dimension doesn't
// participate, otherwise it is the number of threads. We could start with warp
// reductions, then reduce the warps, this could save some shared memory, but
// may actually be slower.
template<int X_THREADS, int Y_THREADS, int Z_THREADS, typename T, typename Func>
__inline__ __device__
void blockReduce(T& out, const T inp_val, Func reduction_op) {

  static constexpr int X_STRIDE = (X_THREADS > 0 ? X_THREADS: 1);
  static constexpr int Y_STRIDE = (Y_THREADS > 0 ? Y_THREADS: 1);
  static constexpr int Z_STRIDE = (Z_THREADS > 0 ? Z_THREADS: 1);

  static constexpr int numel = X_STRIDE * Y_STRIDE * Z_STRIDE;

  __shared__ T shared_mem[numel];

  unsigned int reduction_size = 1;
  unsigned int linear_tid = 0;

  if(X_THREADS > 0){
    linear_tid += threadIdx.x;
    reduction_size *= X_STRIDE;
  }
  if(Y_THREADS > 0){
    linear_tid += threadIdx.y * X_STRIDE;
    reduction_size *= Y_STRIDE;
  }
  if(Z_THREADS > 0){
    linear_tid += threadIdx.z * Y_STRIDE * X_STRIDE;
    reduction_size *= Z_STRIDE;
  }

  // how many threads in inner most contig reduction, i.e. if this is >32 we can
  // do warp shuffles. We could do some template magic to make this a constexpr
  // value.
  int contig_threads = X_STRIDE;
  if(Y_THREADS > 0){
    contig_threads*=Y_THREADS;
    if(Z_THREADS>0)
      contig_threads*=Z_THREADS;
  }

  // Round contig_threads down to nearest power of 2
  contig_threads = 1 << (31 - __clz(contig_threads));
  // If greater than a warp round down to a warp
  contig_threads = contig_threads > 32 ? 32 : contig_threads;

  shared_mem[linear_tid] = inp_val;
  __syncthreads();
  // Reduce down to nearest power of 2:
  int np2 =  1 << (31 - __clz(reduction_size));

  if( linear_tid < np2 ){
    if( linear_tid + np2 < reduction_size){
      reduction_op( shared_mem[linear_tid], shared_mem[linear_tid + np2] );
    }
  }
  __syncthreads();
  for (int factor = np2/2; factor >= contig_threads; factor>>=1) {
    if (linear_tid < factor) {
      reduction_op( shared_mem[linear_tid], shared_mem[linear_tid + factor] );
    }
    __syncthreads();
  }

  unsigned int mask = 0;
  mask = ~mask; // flip all bits to 1
  mask >>= (32 - contig_threads); // Move bits right

  T val = shared_mem[linear_tid];
  if( linear_tid < contig_threads / 2){
     reduction_op(val, shared_mem[linear_tid + contig_threads / 2] );
    for (int offset = contig_threads/2; offset > 0; offset /= 2){
      reduction_op(val, __shfl_down_sync(mask, val, offset));
    }
  }

  if(linear_tid == 0)
    out = val;
}

)";

} // namespace cuda
} // namespace fuser
} // namespace jit
} // namespace torch