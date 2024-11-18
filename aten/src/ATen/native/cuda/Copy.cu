#define TORCH_ASSERT_ONLY_METHOD_OPERATORS
#include <ATen/core/Tensor.h>
#include <ATen/Context.h>
#include <ATen/Dispatch.h>
#include <ATen/Dispatch_v2.h>
#include <ATen/cuda/CachingHostAllocator.h>
#include <ATen/cuda/CUDAContext.h>
#include <ATen/cuda/CUDAEvent.h>
#include <ATen/cuda/PeerToPeerAccess.h>
#include <ATen/native/Copy.h>
#include <ATen/native/TensorIterator.h>
#include <ATen/native/cuda/Loops.cuh>

#ifndef AT_PER_OPERATOR_HEADERS
#include <ATen/Functions.h>
#else
#include <ATen/ops/empty_like.h>
#endif

#include <c10/cuda/CUDACachingAllocator.h>
#include <c10/cuda/CUDAStream.h>

// TODO(NS): Investigate why FP8 conversion intrinsics end up being slower
#ifdef AT_USE_NV_CVT_INTRINSICS
#include <cuda_fp8.h>
#endif

namespace at::native {

void neg_kernel_cuda(TensorIteratorBase &iter);
void conj_kernel_cuda(TensorIteratorBase &iter);

void float16_copy_kernel_cuda(TensorIteratorBase &iter) {
    gpu_kernel_nocast(iter, [] GPU_LAMBDA(float value) {
        return static_cast<at::Half>(value);
    });
}

void bfloat16_copy_kernel_cuda(TensorIteratorBase &iter) {
    gpu_kernel_nocast(iter, [] GPU_LAMBDA(float value) {
        return static_cast<at::BFloat16>(value);
    });
}

void float8_copy_kernel_cuda(TensorIteratorBase &iter) {
  ScalarType dtype = iter.dtype(0);
  ScalarType other_dtype = iter.dtype(1);
  if (dtype == kFloat8_e4m3fn) {
    switch (other_dtype) {
      case kFloat:
         gpu_kernel_nocast(iter, [] GPU_LAMBDA(float value) {
             return Float8_e4m3fn(value);
         });
         break;
      case kHalf:
         gpu_kernel_nocast(iter, [] GPU_LAMBDA(Half value) {
             return Float8_e4m3fn(value);
         });
         break;
      case kBFloat16:
         gpu_kernel_nocast(iter, [] GPU_LAMBDA(BFloat16 value) {
             return Float8_e4m3fn(value);
         });
         break;
      default:
        gpu_kernel(iter, [] GPU_LAMBDA(Float8_e4m3fn x) { return x; });
        break;
    }
  } else if (dtype == kFloat8_e5m2) {
    switch (other_dtype) {
      case kFloat:
         gpu_kernel_nocast(iter, [] GPU_LAMBDA(float value) {
#ifdef AT_USE_NV_CVT_INTRINSICS
             const auto x =  __nv_cvt_float_to_fp8(value, __NV_NOSAT, __NV_E5M2);
             return Float8_e5m2(x, Float8_e5m2::from_bits());
#else
             return Float8_e5m2(value);
#endif
         });
         break;
      case kHalf:
         gpu_kernel_nocast(iter, [] GPU_LAMBDA(Half value) {
#ifdef AT_USE_NV_CVT_INTRINSICS
             const auto x =  __nv_cvt_halfraw_to_fp8(static_cast<__half>(value), __NV_NOSAT, __NV_E5M2);
             return Float8_e5m2(x, Float8_e5m2::from_bits());
#else
             return Float8_e5m2(value);
#endif
         });
         break;
      case kBFloat16:
         gpu_kernel_nocast(iter, [] GPU_LAMBDA(BFloat16 value) {
#ifdef AT_USE_NV_CVT_INTRINSICS
             const auto x =  __nv_cvt_bfloat16raw_to_fp8(static_cast<__nv_bfloat16>(value), __NV_NOSAT, __NV_E5M2);
             return Float8_e5m2(x, Float8_e5m2::from_bits());
#else
             return Float8_e5m2(value);
#endif
         });
         break;
      default:
         gpu_kernel(iter, [] GPU_LAMBDA(Float8_e5m2 x) { return x; });
         break;
    }
  } else if (dtype == kFloat8_e4m3fnuz) {
    switch (other_dtype) {
      case kFloat:
         gpu_kernel_nocast(iter, [] GPU_LAMBDA(float value) {
             return Float8_e4m3fnuz(value);
         });
         break;
      case kHalf:
         gpu_kernel_nocast(iter, [] GPU_LAMBDA(Half value) {
             return Float8_e4m3fnuz(value);
         });
         break;
      case kBFloat16:
         gpu_kernel_nocast(iter, [] GPU_LAMBDA(BFloat16 value) {
             return Float8_e4m3fnuz(value);
         });
         break;
      default:
        gpu_kernel(iter, [] GPU_LAMBDA(Float8_e4m3fnuz x) { return x; });
        break;
    }
  } else if (dtype == kFloat8_e5m2fnuz) {
    switch (other_dtype) {
      case kFloat:
         gpu_kernel_nocast(iter, [] GPU_LAMBDA(float value) {
             return Float8_e5m2fnuz(value);
         });
         break;
      case kHalf:
         gpu_kernel_nocast(iter, [] GPU_LAMBDA(Half value) {
             return Float8_e5m2fnuz(value);
         });
         break;
      case kBFloat16:
         gpu_kernel_nocast(iter, [] GPU_LAMBDA(BFloat16 value) {
             return Float8_e5m2fnuz(value);
         });
         break;
      default:
         gpu_kernel(iter, [] GPU_LAMBDA(Float8_e5m2fnuz x) { return x; });
         break;
    }
  } else {
    TORCH_CHECK(false, "This supposed ot be called only for Float8 types");
  }
}

// This API is for detecting whether the permute parameter of a three-dimensional tensor
// in the Copy operation from src to dst is from [0, 1, 2] to [0, 2, 1].
bool is_permute_021(TensorIteratorBase &iter) {
  const auto& input = iter.tensor(1);
  const auto& output = iter.tensor(0);

  bool is_permute = false;
  if (input.dim() == 3) {
    is_permute = true;
    is_permute &= input.dim() == output.dim();
    is_permute &= input.stride(0) == input.size(1) * input.size(2);
    is_permute &= input.stride(1) == 1;
    is_permute &= input.stride(2) == input.size(1);
    is_permute &= output.is_contiguous();
  }
  return is_permute;
}

template<class _T, int _WG>
__global__ void transpose_tile_big_kernel(const void* __restrict a, void* __restrict c, const int N, const int K)
{
    constexpr uint32_t BIG_TILE_SIZE = 64;
    // pad LDS row by dword
    constexpr uint32_t LDS_PAD = (4 / sizeof(_T));
    constexpr uint32_t element_size = sizeof(_T);  // in bytes
    constexpr uint32_t elements_in_16B = 16 / element_size;

    union BLOCK_16B
    {
        _T e[elements_in_16B];
        __uint128_t ow;
    };
    // Round up processing to next full tile
    const uint32_t n_tiles = (N + BIG_TILE_SIZE - 1) / BIG_TILE_SIZE;
    const uint32_t k_tiles = (K + BIG_TILE_SIZE - 1) / BIG_TILE_SIZE;
    const uint32_t nk_tiles = n_tiles * k_tiles;
    const uint32_t m = blockIdx.x / nk_tiles;
    const uint64_t stride_n = N * sizeof(_T);
    const uint64_t stride_k = K * sizeof(_T);
    const uint64_t stride_nk = N * K * sizeof(_T);

    // Walk destination tiles continuously for cache coherency
    constexpr uint32_t XCD = 8;
    constexpr uint32_t SEQ = 8;
    constexpr uint32_t sblk = XCD * SEQ;
    const uint32_t max_swizzle = (nk_tiles / sblk) * sblk;
    uint32_t tIdx = blockIdx.x % nk_tiles;
    tIdx = tIdx > max_swizzle ? tIdx :
        (tIdx / sblk) * sblk + (tIdx % sblk) / SEQ + (tIdx % SEQ) * XCD;
    uint32_t ti = tIdx / k_tiles;
    uint32_t tj = tIdx % k_tiles;

    __shared__ _T sa[BIG_TILE_SIZE][BIG_TILE_SIZE + LDS_PAD];

    // Detect partial tiles
    uint32_t max_part_n = (ti == (n_tiles - 1) && (N % BIG_TILE_SIZE) != 0) ? (N % BIG_TILE_SIZE) : BIG_TILE_SIZE;
    uint32_t max_part_k = (tj == (k_tiles - 1) && (K % BIG_TILE_SIZE) != 0) ? (K % BIG_TILE_SIZE) : BIG_TILE_SIZE;

    if (max_part_n == BIG_TILE_SIZE && max_part_k == BIG_TILE_SIZE)
    {
        // Copy full tile with large loads
        constexpr uint32_t row_bytes = BIG_TILE_SIZE * sizeof(_T);
        constexpr uint32_t vmem_per_row = row_bytes / sizeof(__uint128_t);
        constexpr uint32_t rows_per_wg = _WG / vmem_per_row;
        constexpr uint32_t vmem_per_thread = BIG_TILE_SIZE / rows_per_wg;
        // Make sure WG isn't too large
        static_assert(vmem_per_thread >= 1);

        const uint8_t* pat = (const uint8_t*)a + tj * BIG_TILE_SIZE * stride_n + ti * row_bytes + m * stride_nk;
        #pragma unroll
        for (uint32_t t = 0; t < vmem_per_thread; t++)
        {
            uint32_t col = threadIdx.x % vmem_per_row;
            uint32_t row = threadIdx.x / vmem_per_row + t * rows_per_wg;
            uint64_t offset = row * stride_n + col * sizeof(__uint128_t);
            const __uint128_t* pfa = (const __uint128_t*)(pat + offset);
            BLOCK_16B d;
            d.ow = *pfa;
            #pragma unroll
            for (uint32_t i = 0; i < elements_in_16B; i++)
            {
                sa[row][col * elements_in_16B + i] = d.e[i];
            }
        }
        __syncthreads();

        const uint8_t* pc = (const uint8_t*)c + ti * BIG_TILE_SIZE * stride_k + tj * row_bytes + m * stride_nk;
        #pragma unroll
        for (uint32_t t = 0; t < vmem_per_thread; t++)
        {
            uint32_t col = threadIdx.x % vmem_per_row;
            uint32_t row = threadIdx.x / vmem_per_row + t * rows_per_wg;
            uint64_t offset = row * stride_k + col * sizeof(__uint128_t);
            BLOCK_16B d;
            // Transpose tile on read from LDS
            #pragma unroll
            for (uint32_t i = 0; i < elements_in_16B; i++)
            {
                d.e[i] = sa[col * elements_in_16B + i][row];
            }
            __uint128_t* pfc = (__uint128_t*)(pc + offset);
            *pfc = d.ow;
        }
    }
    else
    {
        // Copy partial tiles with element accesses
        constexpr uint32_t row_bytes = BIG_TILE_SIZE * sizeof(_T);
        constexpr uint32_t vmem_per_row = BIG_TILE_SIZE;
        constexpr uint32_t rows_per_wg = _WG / vmem_per_row;
        constexpr uint32_t vmem_per_thread = BIG_TILE_SIZE / rows_per_wg;
        // Make sure WG isn't too large
        static_assert(vmem_per_thread >= 1);

        const uint8_t* pat = (const uint8_t*)a + tj * BIG_TILE_SIZE * stride_n + ti * row_bytes + m * stride_nk;
        #pragma unroll
        for (uint32_t t = 0; t < vmem_per_thread; t++)
        {
            uint32_t col = threadIdx.x % vmem_per_row;
            uint32_t row = threadIdx.x / vmem_per_row + t * rows_per_wg;
            uint64_t offset = (col < max_part_n && row < max_part_k) ? row * stride_n + col * 2 : 0;
            const uint16_t* pfa = (const uint16_t*)(pat + offset);
            sa[row][col] = *pfa;
        }
        __syncthreads();

        const uint8_t* pc = (const uint8_t*)c + ti * BIG_TILE_SIZE * stride_k + tj * row_bytes + m * stride_nk;
        #pragma unroll
        for (uint32_t t = 0; t < vmem_per_thread; t++)
        {
            uint32_t col = threadIdx.x % vmem_per_row;
            uint32_t row = threadIdx.x / vmem_per_row + t * rows_per_wg;
            if (col < max_part_k && row < max_part_n)
            {
                uint64_t offset = row * stride_k + col * 2;
                uint16_t* pfc = (uint16_t*)(pc + offset);
                *pfc = sa[col][row];
            }
        }
    }
}

void transpose_last2dim(TensorIteratorBase &iter) {
  void* dst = iter.data_ptr(0);
  void* src = iter.data_ptr(1);
  const auto& input = iter.tensor(1);

  int M = input.size(0);
  int N = input.size(1);
  int K = input.size(2);

  auto stream = c10::cuda::getCurrentCUDAStream();
  constexpr uint32_t BIG_TILE_SIZE = 64;
  int big_tile_wg = M * ((N + BIG_TILE_SIZE - 1) / BIG_TILE_SIZE) * ((K + BIG_TILE_SIZE - 1) / BIG_TILE_SIZE);
  const dim3 grid_dim(big_tile_wg, 1, 1);
  const dim3 block_dim(256, 1, 1);
  transpose_tile_big_kernel<uint16_t, 256><<<grid_dim, block_dim, 0, stream>>>(src, dst, N, K);
}

// TODO: We probably can use the opaque type trick to avoid creating duplicate
// kernels for equivalent bit lengths
void direct_copy_kernel_cuda(TensorIteratorBase &iter) {
  ScalarType dtype = iter.dtype(0);

  if (isQIntType(dtype)) {
    AT_DISPATCH_QINT_TYPES(dtype, "copy_", [&] {
      gpu_kernel(iter, [] GPU_LAMBDA(scalar_t x) { return x; });
    });
  } else if (dtype == kFloat8_e5m2 || dtype == kFloat8_e4m3fn || dtype == kFloat8_e5m2fnuz || dtype == kFloat8_e4m3fnuz) {
     float8_copy_kernel_cuda(iter);
  } else if (iter.dtype(1) == kFloat && (dtype == kBFloat16 || dtype == kHalf)) {
     if (dtype == kBFloat16) {
       bfloat16_copy_kernel_cuda(iter);
     } else {
       float16_copy_kernel_cuda(iter);
     }
  } else if (isBitsType(dtype)) {
    TORCH_CHECK(dtype == iter.dtype(1), "copy_() does not support casting "
      "bits types to different bits types. Source dtype is ", iter.dtype(1), "target dtype is ", dtype);
    AT_DISPATCH_BIT_TYPES(dtype, "copy_", [&] {
      gpu_kernel_nocast(iter, [] GPU_LAMBDA(scalar_t x) { return x; });
    });
  } else if (is_permute_021(iter) && (dtype == kBFloat16 || dtype == kHalf)) {
    transpose_last2dim(iter);
  } else {
    AT_DISPATCH_V2(
        dtype, "copy_", AT_WRAP([&] {
          gpu_kernel(iter, [] GPU_LAMBDA(scalar_t x) { return x; });
    }), AT_EXPAND(AT_ALL_TYPES_AND_COMPLEX), kHalf, kBool, kBFloat16, kComplexHalf, AT_EXPAND(AT_BAREBONES_UNSIGNED_TYPES));
  }
}

void neg_conj_kernel_cuda(TensorIteratorBase &iter) {
  AT_DISPATCH_COMPLEX_TYPES(iter.common_dtype(), "neg_conj_cuda", [&] {
    gpu_kernel(iter, [] GPU_LAMBDA(scalar_t x) { return -std::conj(x); });
  });
}

using namespace at::cuda;

// device-to-device copy, does type conversion
void copy_device_to_device(TensorIterator& iter,
                           bool non_blocking,
                           bool p2p_enabled) {
  int64_t numel = iter.numel();

  // We can memcpy the memory if both tensors have the same type AND both
  // tensors are contiguous after dimension coalescing and reordering.
  bool same_type = iter.dtype(0) == iter.dtype(1);
  bool same_conj = iter.tensor(0).is_conj() == iter.tensor(1).is_conj();
  bool same_neg = iter.tensor(0).is_neg() == iter.tensor(1).is_neg();
  bool memcpy_eligible = same_type && same_conj && same_neg && iter.is_contiguous();

  Device dst_device = iter.device(0);
  Device src_device = iter.device(1);

  CUDAGuard device_guard(src_device);

  // We always perform the copy on the source device, using the current stream
  // on the source device, and we fully synchronize on both src and dst's
  // current streams for completion of the copy. We have to explicitly do this
  // for non-contig copies. This mimics the behavior of cross-device
  // cudaMemcpyAsync on the default stream.
  CUDAStream copy_stream = getCurrentCUDAStream(src_device.index());
  if (src_device != dst_device) {
    // This is a cross-device copy on the src current stream and dst current
    // stream. We perform a two-way barrier between both devices' streams
    // before the copy. This ensures that any write-after-write and
    // write-after-read dependencies on the destination side are handled, so
    // that no one is operating on the dst memory when we perform the copy.
    // src waits on dst barrier (src already waits on src)
    CUDAEvent dst_ready;
    device_guard.set_device(dst_device);
    dst_ready.record(getCurrentCUDAStream(dst_device.index()));

    device_guard.set_device(src_device);
    dst_ready.block(copy_stream);
  }

  if (memcpy_eligible) {
    void *dst = iter.data_ptr(0);
    void *src = iter.data_ptr(1);
    size_t size = numel * iter.element_size(0);
    if (src != dst || src_device != dst_device) {
      // Due to bizarre cuda driver intricacies, copies of
      // cudaMallocAsynced memory between devices that aren't
      // peer-to-peer-capable need "cudaMemcpyPeerAsync".
      // So we let the allocator implement the correct call
      // (either cudaMemcpyAsync or cudaMemcpyPeerAsync)
      AT_CUDA_CHECK(CUDACachingAllocator::memcpyAsync(
        dst, dst_device.index(),
        src, src_device.index(),
        size, copy_stream, p2p_enabled));
    }
  } else {
    if (same_neg) {
      if (!same_conj) {
        conj_kernel_cuda(iter);
      } else {
        direct_copy_kernel_cuda(iter);
      }
    } else {
      if (!same_conj) {
        neg_conj_kernel_cuda(iter);
      } else {
        neg_kernel_cuda(iter);
      }
    }
  }

  if (src_device != dst_device) {
    // dst waits on src barrier (dst already waits on dst). We cannot
    // operate on dst's copy until the copy is complete.

    // Still on src_device, record stream event
    CUDAEvent src_ready;
    src_ready.record(copy_stream);

    device_guard.set_device(dst_device);
    src_ready.block(getCurrentCUDAStream(dst_device.index()));
  }

  AT_CUDA_CHECK(cudaGetLastError());
}

static bool copy_requires_temporaries(TensorIterator& iter, bool p2p_enabled) {
  Device dst_device = iter.device(0);
  Device src_device = iter.device(1);

  if (dst_device == src_device) {
    // We never require temporaries for copies on the same GPU.
    TORCH_INTERNAL_ASSERT(dst_device.is_cuda() && src_device.is_cuda());
    return false;
  }

  bool same_dtype = iter.dtype(0) == iter.dtype(1);
  if (same_dtype && iter.is_contiguous()) {
    // Contiguous same-dtype copies can always use cudaMemcpyAsync
    return false;
  } else if (dst_device.is_cuda() && src_device.is_cuda()) {
    // Copies between GPUs can use the copy kernel if P2P is supported
    return !p2p_enabled;
  } else {
    // The remaining cases require temporaries. For example, this includes
    // non-contiguous copies between CPU and GPU.
    return true;
  }
}

static bool maybe_enable_p2p_access(Device dst_device, Device src_device) {
  if (dst_device.is_cpu() || src_device.is_cpu()) {
    return false;
  }
  return at::cuda::get_p2p_access(src_device.index(), dst_device.index());
}

static void copy_kernel_cuda(TensorIterator& iter, bool non_blocking) {
  TORCH_CHECK(iter.ntensors() == 2);

  Device dst_device = iter.device(0);
  Device src_device = iter.device(1);

  // Enable p2p access between devices. (No-op if it involves the CPU)
  bool p2p_enabled = maybe_enable_p2p_access(dst_device, src_device);

  if (copy_requires_temporaries(iter, p2p_enabled)) {
    // NB: this involves recursive calls to copy. Be careful that those copies
    // don't require temporaries or you will cause an infinite recursion!
    auto& dst = iter.tensor(0);
    Tensor dst_contig;
    Tensor src_contig;

    // If non_blocking is true - type conversions are performed on the GPU
    // For blocking transfers conversions are performed on CPU to avoid allocating
    // extra GPU memory
    // for GPU-GPU transfers conversions are performed on the source device
    auto conversion_device = non_blocking ? kCUDA : kCPU;
    if (iter.device_type(1) == conversion_device) {
      dst_contig = dst.is_contiguous() ? dst : at::empty_like(dst, LEGACY_CONTIGUOUS_MEMORY_FORMAT);
      src_contig = iter.tensor(1).to(iter.dtype(0)).expand_as(dst).contiguous();
    } else {
      bool same_type = iter.dtype(0) == iter.dtype(1);
      dst_contig = (dst.is_contiguous() && same_type) ? dst : at::empty_like(dst, iter.dtype(1), LEGACY_CONTIGUOUS_MEMORY_FORMAT);
      src_contig = iter.tensor(1).expand_as(dst).contiguous();
    }

    // propagate the correct conjugate bit
    dst_contig._set_conj(dst.is_conj());
    src_contig._set_conj(iter.tensor(1).is_conj());

    dst_contig._set_neg(dst.is_neg());
    src_contig._set_neg(iter.tensor(1).is_neg());

    // perform a same-dtype copy on contiguous tensors
    TORCH_INTERNAL_ASSERT(dst_contig.sizes().equals(src_contig.sizes()));
    TORCH_INTERNAL_ASSERT(dst_contig.scalar_type() == src_contig.scalar_type());
    dst_contig.copy_(src_contig, non_blocking);

    // if necessary, copy back into dst
    if (!dst_contig.is_same(dst)) {
      TORCH_INTERNAL_ASSERT(dst_contig.device() == dst.device());
      dst.copy_(dst_contig, non_blocking);
    }
    return;
  }

  // Copy on GPU (or between GPUs)
  if (dst_device.is_cuda() && src_device.is_cuda()) {
    copy_device_to_device(iter, non_blocking, p2p_enabled);
    return;
  }

  // Copy between CPU and GPU
  cuda::OptionalCUDAGuard device_guard;
  cudaMemcpyKind kind;
  if (dst_device.is_cuda() && src_device.is_cpu()) {
    device_guard.set_device(dst_device);
    kind = cudaMemcpyHostToDevice;
  } else if (dst_device.is_cpu() && src_device.is_cuda()) {
    device_guard.set_device(src_device);
    kind = cudaMemcpyDeviceToHost;
  } else {
    TORCH_INTERNAL_ASSERT(false, "unsupported devices in GPU copy_()");
  }

  void* dst = iter.data_ptr(0);
  void* src = iter.data_ptr(1);
  int64_t nbytes = iter.numel() * iter.element_size(0);
  CUDAStream stream = getCurrentCUDAStream();

  if (non_blocking) {
    AT_CUDA_CHECK(cudaMemcpyAsync(dst, src, nbytes, kind, stream));
    // we use both the storage context and the tensor data pointer as the key
    // for the caching host allocator. This allows us to better attribute the
    // events to the original tensor allocation correctly. The cases we seek to
    // handle are:

    // 1: a user can pass a pinned memory tensor with an alternative
    // context, for example if allocating memory directly from the pinned memory
    // allocator and constructing a tensor with torch::from_blob.

    // 2: a user can pass a tensor with a different base pointer to the original
    // allocation (via slicing).
    const auto& dst_tensor = iter.tensor(0);
    const auto& src_tensor = iter.tensor(1);
    const auto& host_tensor = (dst_device == kCPU ? dst_tensor : src_tensor);
    auto* ptr = (dst_device == kCPU ? dst : src);
    auto* ctx = host_tensor.storage().data_ptr().get_context();
    // TODO: warn on the return value.
    CachingHostAllocator_recordEvent(ptr, ctx, stream);

  } else {
    at::cuda::memcpy_and_sync(dst, src, nbytes, kind, stream);
  }

  if (iter.tensor(0).is_conj() != iter.tensor(1).is_conj()) {
     iter.tensor(0).conj_physical_();
  }
  if (iter.tensor(0).is_neg() != iter.tensor(1).is_neg()) {
     iter.tensor(0).neg_();
  }
}

REGISTER_DISPATCH(copy_stub, &copy_kernel_cuda)

} // namespace at::native
