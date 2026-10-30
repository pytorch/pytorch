#define TORCH_ASSERT_ONLY_METHOD_OPERATORS
#include <ATen/core/Tensor.h>
#include <ATen/AccumulateType.h>
#include <ATen/cuda/CUDAGeneratorImpl.h>
#include <ATen/Dispatch.h>
#include <ATen/Utils.h>
#include <ATen/cuda/detail/IndexUtils.cuh>
#include <ATen/cuda/detail/TensorInfo.cuh>
#include <ATen/cuda/CUDAGraphsUtils.cuh>
#include <c10/macros/Macros.h>
#include <curand_kernel.h>
#include <type_traits>
#include <utility>

#include <ATen/native/TensorIterator.h>
#include <ATen/native/cuda/Loops.cuh>
#include <ATen/native/cuda/MemoryAccess.cuh>

#ifndef AT_PER_OPERATOR_HEADERS
#include <ATen/Functions.h>
#include <ATen/NativeFunctions.h>
#else
#include <ATen/ops/_masked_scale_native.h>
#include <ATen/ops/empty_like.h>
#include <ATen/ops/native_dropout_backward_native.h>
#include <ATen/ops/ones_like.h>
#include <ATen/ops/zeros_like.h>
#endif

namespace at::native {

namespace {

// philox generates 128 bits of randomness at a time, returned as a float4 by curand_uniform4.
// Each iteration handles at least UNROLL elements so that all 4 values are used; the unvectorized
// kernel handles exactly UNROLL. Don't change!
constexpr int UNROLL = 4;

// Elements each thread handles per grid-stride iteration. Each output element needs one philox
// value, so this is also the number of philox values drawn per iteration.
C10_HOST_DEVICE constexpr int elems_per_iter(int vec) {
  return vec > UNROLL ? vec : UNROLL;
}

template <
    typename scalar_t,
    typename accscalar_t,
    typename IndexType,
    int VEC,
    typename mask_t>
C10_LAUNCH_BOUNDS_2(256, 4)
__global__ void
fused_dropout_kernel_vec(at::cuda::detail::TensorInfo<const scalar_t, IndexType> a,
                         at::cuda::detail::TensorInfo<scalar_t, IndexType> b,
                         at::cuda::detail::TensorInfo<mask_t, IndexType> c,
                         IndexType totalElements, accscalar_t p,
                         PhiloxCudaState philox_args) {
  auto [seed, offset] = at::cuda::philox::unpack(philox_args);
  IndexType idx = blockIdx.x * blockDim.x + threadIdx.x;
  curandStatePhilox4_32_10_t state;
  curand_init(seed, idx, offset, &state);
  accscalar_t scale = 1.0 / p;

  // Elements per thread per iteration, one philox value each.
  constexpr int ELEMS = elems_per_iter(VEC);
  static_assert(ELEMS % 4 == 0 && ELEMS % VEC == 0, "VEC must be a power of two");
  // Number of VEC-sized chunks per thread per iteration; > 1 only for VEC < 4.
  constexpr int CHUNKS = ELEMS / VEC;

  // An iteration's chunks are chunk_stride apart, so neighboring threads access neighboring memory.
  const IndexType chunk_stride = gridDim.x * blockDim.x * VEC;
  // Note: We explicitly check for is_contiguous() before launching the vectorized kernel
  // and use linear indices directly to allow vectorization of NHWC (or other) ordering.
  for (IndexType base = idx * VEC; base < totalElements; base += chunk_stride * CHUNKS) {
    // Draw this iteration's philox values.
    float rand[ELEMS];
    #pragma unroll
    for (int d = 0; d < ELEMS / 4; d++) {
      // curand_uniform_double was pure evil anyway, not doing what it promises, and there's
      // nothing for Halfs, so generate float for everything
      const float4 r4 = curand_uniform4(&state);
      rand[4 * d] = r4.x;
      rand[4 * d + 1] = r4.y;
      rand[4 * d + 2] = r4.z;
      rand[4 * d + 3] = r4.w;
    }

    // Apply dropout to each chunk and store its outputs and mask.
    #pragma unroll
    for (int ci = 0; ci < CHUNKS; ci++) {
      const IndexType chunk_start = base + ci * chunk_stride;
      if (chunk_start >= totalElements) {
        break;
      }
      using ChunkT = memory::aligned_vector<scalar_t, VEC>;
      using MaskChunkT = memory::aligned_vector<mask_t, VEC>;
      scalar_t src[VEC];
      scalar_t out[VEC];
      mask_t mask[VEC];
      *reinterpret_cast<ChunkT*>(&src[0]) = *reinterpret_cast<const ChunkT*>(&a.data[chunk_start]);
      #pragma unroll
      for (int ii = 0; ii < VEC; ii++) {
        const float keep = rand[ci * VEC + ii] < p;
        out[ii] = src[ii] * keep * scale;
        mask[ii] = (mask_t)keep;
      }
      *reinterpret_cast<ChunkT*>(&b.data[chunk_start]) = *reinterpret_cast<ChunkT*>(&out[0]);
      *reinterpret_cast<MaskChunkT*>(&c.data[chunk_start]) = *reinterpret_cast<MaskChunkT*>(&mask[0]);
      // Not needed for correctness; keeping warps in step measurably helps memory throughput.
      __syncthreads();
    }
  }
}

template <
    typename scalar_t,
    typename accscalar_t,
    typename IndexType,
    int ADims,
    int BDims = ADims,
    typename mask_t>
C10_LAUNCH_BOUNDS_2(256, 4)
__global__ void
fused_dropout_kernel(cuda::detail::TensorInfo<const scalar_t, IndexType> a,
                     cuda::detail::TensorInfo<scalar_t, IndexType> b,
                     cuda::detail::TensorInfo<mask_t, IndexType> c,
                     IndexType totalElements, accscalar_t p,
                     PhiloxCudaState philox_args) {
  auto [seed, offset] = at::cuda::philox::unpack(philox_args);
  IndexType idx = blockIdx.x * blockDim.x + threadIdx.x;
  curandStatePhilox4_32_10_t state;
  curand_init(seed, idx, offset, &state);
  accscalar_t scale = 1.0 / p;

  IndexType rounded_size = ((totalElements - 1)/(blockDim.x * gridDim.x * UNROLL)+1) *
        blockDim.x * gridDim.x * UNROLL;
  for (IndexType linearIndex = idx;
       linearIndex < rounded_size;
       linearIndex += gridDim.x * blockDim.x*UNROLL) {
//curand_uniform_double was pure evil anyway, not doing what it promises, and there's nothing for Halfs, so generate float for everything
       float4 rand = curand_uniform4(&state);
       scalar_t src[UNROLL];
       rand.x = rand.x < p;
       rand.y = rand.y < p;
       rand.z = rand.z < p;
       rand.w = rand.w < p;
       for (int ii = 0; ii < UNROLL; ii++) {
           IndexType li = linearIndex + blockDim.x * gridDim.x * ii;
           if (li < totalElements) {
    // Convert `linearIndex` into an offset of `a`
               const IndexType aOffset =
                   cuda::detail::IndexToOffset<const scalar_t, IndexType, ADims>::get(li, a);
               src[ii] = a.data[aOffset];
           }
       }
       for (int ii = 0; ii < UNROLL; ii++) {
           IndexType li = linearIndex + blockDim.x * gridDim.x * ii;
           if (li < totalElements) {
    // Convert `linearIndex` into an offset of `b`
               const IndexType bOffset =
                   cuda::detail::IndexToOffset<scalar_t, IndexType, BDims>::get(li, b);
               b.data[bOffset] = src[ii]*(&rand.x)[ii]*scale;
               c.data[bOffset] = (mask_t)(&rand.x)[ii];
           }
       }
       __syncthreads();
  }
}

template<typename mask_t, typename scalar_t, typename accscalar_t>
void masked_scale_kernel(at::Tensor& ret, const at::Tensor& src, const at::Tensor& mask, accscalar_t scale){
   auto iter = at::TensorIteratorConfig()
     .check_all_same_dtype(false)
     .add_output(ret)
     .add_const_input(src)
     .add_const_input(mask)
     .build();

   at::native::gpu_kernel(
       iter,
       [=]GPU_LAMBDA(const scalar_t src_val, const mask_t mask_val) -> scalar_t {
          return (float)mask_val * src_val * scale;
       });
}

template <typename scalar_t>
int get_vector_size(at::Tensor self, at::Tensor ret, at::Tensor mask) {
  int vec_size = 4;
  // get the vector size
  if (!self.is_non_overlapping_and_dense() || !ret.is_non_overlapping_and_dense() || !mask.is_non_overlapping_and_dense()) {
    vec_size = 1;
  } else {
    vec_size = memory::can_vectorize_up_to<scalar_t>((const char*)self.const_data_ptr());
#ifdef USE_ROCM
    // make sure we don't break assumption that we can't have > 16 elements / thread
    TORCH_INTERNAL_ASSERT(vec_size <= 16, "Value of VEC must be in [2, 4, 8, 16]");
#else
    const int optimal_vec_size = 16 / static_cast<int>(sizeof(scalar_t));
    vec_size = std::min<int>(optimal_vec_size, vec_size);

    TORCH_INTERNAL_ASSERT(vec_size <= 8, "Value of VEC must be in [2, 4, 8]");
#endif
  }

  // check that we'd have no remainders - prefer a smaller vector size with no remainders over a larger vector and remainder.
  bool can_vectorize = true;
  do {
    can_vectorize = self.numel() % vec_size == 0 && ret.numel() % vec_size == 0 && mask.numel() % vec_size == 0;
    if (!can_vectorize) vec_size /= 2;
  } while (vec_size > 1 && !can_vectorize);
  return can_vectorize ? vec_size : 1;
}

template <typename index_type, typename mask_t>
inline void launcher(
    const Tensor& self,
    Tensor& ret,
    Tensor& mask,
    double p,
    const int64_t nelem,
    CUDAGeneratorImpl* gen,
    dim3 grid,
    dim3 dim_block) {
  AT_DISPATCH_FLOATING_TYPES_AND2(
      at::ScalarType::Half,
      at::ScalarType::BFloat16,
      self.scalar_type(),
      "fused_dropout",
      [&] {
        using accscalar_t = acc_type<scalar_t, true>;
        accscalar_t pa = (accscalar_t)(p);
        auto self_info =
            cuda::detail::getTensorInfo<const scalar_t, index_type>(self);
        auto ret_info =
            cuda::detail::getTensorInfo<scalar_t, index_type>(ret);
        auto mask_info =
            cuda::detail::getTensorInfo<mask_t, index_type>(mask);
        self_info.collapseDims();
        ret_info.collapseDims();
        mask_info.collapseDims(); // ret and mask are collapsed to 1d
                                  // contiguous tensor

        int vec_size = get_vector_size<scalar_t>(self, ret, mask);

        // Philox values each thread draws: elems_per_iter per iteration, times its iteration count.
        const int64_t elems = elems_per_iter(vec_size);
        const int64_t total_threads = static_cast<int64_t>(dim_block.x) * grid.x;
        const int64_t counter_offset = ((nelem - 1) / (total_threads * elems) + 1) * elems;
        PhiloxCudaState rng_engine_inputs;
        {
          // See Note [Acquire lock when using random generators]
          std::lock_guard<std::mutex> lock(gen->mutex_);
          rng_engine_inputs = gen->philox_cuda_state(counter_offset);
        }

        if (vec_size > 1) {
          auto launch_vec = [&](auto vec) {
            fused_dropout_kernel_vec<scalar_t, accscalar_t, index_type, decltype(vec)::value>
                <<<grid, dim_block, 0, at::cuda::getCurrentCUDAStream()>>>(
                    self_info, ret_info, mask_info, nelem, pa, rng_engine_inputs);
            C10_CUDA_KERNEL_LAUNCH_CHECK();
          };
          switch (vec_size) {
            case 16:
              launch_vec(std::integral_constant<int, 16>{});
              break;
            case 8:
              launch_vec(std::integral_constant<int, 8>{});
              break;
            case 4:
              launch_vec(std::integral_constant<int, 4>{});
              break;
            case 2:
              launch_vec(std::integral_constant<int, 2>{});
              break;
            default:
              TORCH_INTERNAL_ASSERT(false, "Unexpected vectorization size");
          }
        } else {
          switch (self_info.dims) {
            case 1:
              fused_dropout_kernel<scalar_t, accscalar_t, index_type, 1>
                  <<<grid, dim_block, 0, at::cuda::getCurrentCUDAStream()>>>(
                      self_info,
                      ret_info,
                      mask_info,
                      nelem,
                      pa,
                      rng_engine_inputs);
              C10_CUDA_KERNEL_LAUNCH_CHECK();
              break;
            default:
              if (!self.is_contiguous() && ret.is_contiguous() &&
                  mask.is_contiguous()) {
                fused_dropout_kernel<scalar_t, accscalar_t, index_type, -1, 1>
                    <<<grid,
                        dim_block,
                        0,
                        at::cuda::getCurrentCUDAStream()>>>(
                        self_info,
                        ret_info,
                        mask_info,
                        nelem,
                        pa,
                        rng_engine_inputs);
                C10_CUDA_KERNEL_LAUNCH_CHECK();
              } else {
                fused_dropout_kernel<scalar_t, accscalar_t, index_type, -1>
                    <<<grid,
                        dim_block,
                        0,
                        at::cuda::getCurrentCUDAStream()>>>(
                        self_info,
                        ret_info,
                        mask_info,
                        nelem,
                        pa,
                        rng_engine_inputs);
                C10_CUDA_KERNEL_LAUNCH_CHECK();
              }
          }
        }
      });
}

} //anonymous namespace

template <typename mask_t>
std::tuple<Tensor,Tensor>
dropout_cuda(CUDAGeneratorImpl* gen, const Tensor& self, double p){
  Tensor mask = at::empty_like(self, self.options().dtype(c10::CppTypeToScalarType<mask_t>::value));
  const int64_t nelem = self.numel();
  // empty tensors should not get here, but just in case, avoid FPE
  // non-training shot-cut
  if (nelem==0) return std::tuple<Tensor,Tensor>(self.clone(), mask);

  Tensor ret = at::empty_like(self);
  const int64_t block_size = 256;
  unsigned int blocks_per_sm = at::cuda::getCurrentDeviceProperties()->maxThreadsPerMultiProcessor/block_size;
  dim3 dim_block(block_size);
  dim3 grid((nelem + block_size -1)/block_size);
  grid.x = std::min((unsigned int)at::cuda::getCurrentDeviceProperties()->multiProcessorCount * blocks_per_sm, grid.x);
  if (cuda::detail::canUse32BitIndexMath(self)){
    launcher<unsigned int, mask_t>(
        self, ret, mask, p, nelem, gen, grid, dim_block);
  } else {
    launcher<uint64_t, mask_t>(
        self, ret, mask, p, nelem, gen, grid, dim_block);
  }
  return std::tuple<Tensor,Tensor>(ret, mask);
}

std::tuple<Tensor,Tensor>
native_dropout_cuda(const Tensor& self, double p, std::optional<bool> train){
  // short-cut for train == false
  if (train.has_value() && !train.value()) {
    return std::make_tuple(self.clone(), at::ones_like(self, self.options().dtype(c10::CppTypeToScalarType<bool>::value)));
  }
  // short-cut
  if (p == 1) {
    // native_dropout_cuda is in derivatives.yaml, so we don't need to add data
    // dependency from output to input for autograd
    auto ret = at::zeros_like(self);
    auto mask = at::zeros_like(self, self.options().dtype(c10::CppTypeToScalarType<bool>::value));
    return std::tuple<Tensor,Tensor>(std::move(ret), std::move(mask));
  }

  auto gen = get_generator_or_default<CUDAGeneratorImpl>(std::nullopt, cuda::detail::getDefaultCUDAGenerator());
  double p1m = 1. - p;
  return dropout_cuda<bool>(gen, self, p1m);
}

// TODO: _fused_dropout_cuda is to be removed, see PR #63937
std::tuple<Tensor,Tensor>
fused_dropout_cuda(const Tensor& self, double p, std::optional<Generator> gen_){
  auto gen = get_generator_or_default<CUDAGeneratorImpl>(gen_, cuda::detail::getDefaultCUDAGenerator());
  return dropout_cuda<uint8_t>(gen, self, p);
}

template <typename mask_t>
Tensor dropout_backward_cuda(const Tensor& grad, const Tensor& mask, double scale){
   Tensor ret = at::empty_like(grad, grad.suggest_memory_format());
   AT_DISPATCH_FLOATING_TYPES_AND2(at::ScalarType::Half, at::ScalarType::BFloat16, ret.scalar_type(), "masked_scale", [&] {
      using accscalar_t = acc_type<scalar_t, true>;
      masked_scale_kernel<mask_t, scalar_t>(ret, grad, mask, (accscalar_t)scale);
  });
  return ret;
}

Tensor native_dropout_backward_cuda(const Tensor& grad, const Tensor& mask, double scale){
   TORCH_CHECK(mask.scalar_type() == at::ScalarType::Bool, "Mask should be Bool Scalar Type", mask.scalar_type());
  return dropout_backward_cuda<bool>(grad, mask, scale);
}

// TODO: masked_scale_cuda is to be removed, see PR #63937
Tensor masked_scale_cuda(const Tensor& self, const Tensor& mask, double scale){
  TORCH_CHECK(mask.scalar_type() == at::ScalarType::Byte, "mask should be torch.uint8 dtype");
  return dropout_backward_cuda<uint8_t>(self, mask, scale);
}

} // namespace at::native
