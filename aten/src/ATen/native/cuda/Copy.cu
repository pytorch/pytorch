#define TORCH_ASSERT_ONLY_METHOD_OPERATORS
#include <ATen/Context.h>
#include <ATen/Dispatch.h>
#include <ATen/Dispatch_v2.h>
#include <ATen/core/Tensor.h>
#include <ATen/cuda/CUDAContext.h>
#include <ATen/cuda/CUDAEvent.h>
#include <ATen/cuda/CachingHostAllocator.h>
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
#include <ATen/cuda/CUDAGraphsUtils.cuh>

#if defined(CUDA_VERSION) && CUDA_VERSION >= 13000
#include <cuda_fp8.h>
#endif

namespace at::native {

namespace {

// Initial pool size for CUDA events per device.
constexpr size_t kInitialEventPoolSize = 8;

at::cuda::CUDAEventPool::Event getEventFromPool(const at::DeviceIndex device_idx) {
  // Pre-populate the pool with events to avoid stalls in creating events
  static auto* event_pool = new at::cuda::CUDAEventPool(kInitialEventPoolSize);
  return event_pool->get(device_idx);
}

} // namespace

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

void bfloat16tofloat32_copy_kernel_cuda(TensorIteratorBase &iter) {
    gpu_kernel_nocast(iter, [] GPU_LAMBDA(at::BFloat16 value) {
        return static_cast<float>(value);
    });
}
void float16tofloat32_copy_kernel_cuda(TensorIteratorBase &iter) {
    gpu_kernel_nocast(iter, [] GPU_LAMBDA(at::Half value) {
        return static_cast<float>(value);
    });
}

template <typename SrcT>
struct ConvertToFloat8E4M3fnOp {
  __device__ __forceinline__ Float8_e4m3fn operator()(SrcT value) const {
#if defined(CUDA_VERSION) && CUDA_VERSION >= 13000 && defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 890
    __nv_fp8_storage_t x;
    if constexpr (std::is_same_v<SrcT, float>) {
      x = __nv_cvt_float_to_fp8(value, __NV_SATFINITE, __NV_E4M3);
    } else if constexpr (std::is_same_v<SrcT, Half>) {
      x = __nv_cvt_halfraw_to_fp8(static_cast<__half>(value), __NV_SATFINITE, __NV_E4M3);
    } else if constexpr (std::is_same_v<SrcT, BFloat16>) {
      x = __nv_cvt_bfloat16raw_to_fp8(static_cast<__nv_bfloat16>(value), __NV_SATFINITE, __NV_E4M3);
    } else {
      x = __nv_cvt_float_to_fp8(static_cast<float>(value), __NV_SATFINITE, __NV_E4M3);
    }
    return Float8_e4m3fn(x, Float8_e4m3fn::from_bits());
#else
    return Float8_e4m3fn(value);
#endif
  }
};

// e5m2 intrinsics are correct but slower; only used for float on Blackwell
// to work around the ptxas subnormal codegen bug.
struct ConvertFloatToFloat8E5M2Op {
  __device__ __forceinline__ Float8_e5m2 operator()(float value) const {
#if defined(CUDA_VERSION) && CUDA_VERSION >= 13020 && defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 1000
    auto x = __nv_cvt_float_to_fp8(value, __NV_NOSAT, __NV_E5M2);
    return Float8_e5m2(x, Float8_e5m2::from_bits());
#else
    return Float8_e5m2(value);
#endif
  }
};

void float8_copy_kernel_cuda(TensorIteratorBase &iter) {
  ScalarType dtype = iter.dtype(0);
  ScalarType other_dtype = iter.dtype(1);
  if (dtype == kFloat8_e4m3fn) {
    switch (other_dtype) {
      case kFloat:
         gpu_kernel_nocast(iter, ConvertToFloat8E4M3fnOp<float>{});
         break;
      case kHalf:
         gpu_kernel_nocast(iter, ConvertToFloat8E4M3fnOp<Half>{});
         break;
      case kBFloat16:
         gpu_kernel_nocast(iter, ConvertToFloat8E4M3fnOp<BFloat16>{});
         break;
      default:
        gpu_kernel(iter, [] GPU_LAMBDA(Float8_e4m3fn x) { return x; });
        break;
    }
  } else if (dtype == kFloat8_e5m2) {
    switch (other_dtype) {
      case kFloat:
         gpu_kernel_nocast(iter, ConvertFloatToFloat8E5M2Op{});
         break;
      case kHalf:
         gpu_kernel_nocast(iter, [] GPU_LAMBDA(Half value) {
             return Float8_e5m2(value);
         });
         break;
      case kBFloat16:
         gpu_kernel_nocast(iter, [] GPU_LAMBDA(BFloat16 value) {
             return Float8_e5m2(value);
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
  } else if (dtype == kFloat8_e8m0fnu) {
    // TODO(#146647): clean this up, too much copy-pasta
    switch (other_dtype) {
      case kFloat:
         gpu_kernel_nocast(iter, [] GPU_LAMBDA(float value) {
             return Float8_e8m0fnu(value);
         });
         break;
      case kHalf:
         gpu_kernel_nocast(iter, [] GPU_LAMBDA(Half value) {
             return Float8_e8m0fnu(value);
         });
         break;
      case kBFloat16:
         gpu_kernel_nocast(iter, [] GPU_LAMBDA(BFloat16 value) {
             return Float8_e8m0fnu(value);
         });
         break;
      default:
         gpu_kernel(iter, [] GPU_LAMBDA(Float8_e8m0fnu x) { return x; });
         break;
    }
  } else {
    TORCH_CHECK(false, "This supposed to be called only for Float8 types");
  }
}

// TODO: We probably can use the opaque type trick to avoid creating duplicate
// kernels for equivalent bit lengths
void direct_copy_kernel_cuda(TensorIteratorBase &iter) {
  ScalarType dtype = iter.dtype(0);
  if (isQIntType(dtype)) {
    AT_DISPATCH_QINT_TYPES(dtype, "copy_", [&] {
      gpu_kernel(iter, [] GPU_LAMBDA(scalar_t x) { return x; });
    });
  } else if (isFloat8Type(dtype)) {
     float8_copy_kernel_cuda(iter);
  } else if (iter.dtype(1) == kFloat && (dtype == kBFloat16 || dtype == kHalf)) {
     if (dtype == kBFloat16) {
       bfloat16_copy_kernel_cuda(iter);
     } else {
       float16_copy_kernel_cuda(iter);
     }
  }
  else if ((iter.dtype(1) == kBFloat16 || iter.dtype(1) == kHalf) && dtype == kFloat) {
    if (iter.dtype(1) == kBFloat16) {
      bfloat16tofloat32_copy_kernel_cuda(iter);
    } else {
      float16tofloat32_copy_kernel_cuda(iter);
    }
  }
  else if (isBitsType(dtype)) {
    TORCH_CHECK(dtype == iter.dtype(1), "copy_() does not support casting "
      "bits types to different bits types. Source dtype is ", iter.dtype(1), "target dtype is ", dtype);
    AT_DISPATCH_BIT_TYPES(dtype, "copy_", [&] {
      gpu_kernel_nocast(iter, [] GPU_LAMBDA(scalar_t x) { return x; });
    });
  } else if (dtype == ScalarType::Float4_e2m1fn_x2) {
    TORCH_CHECK(dtype == iter.dtype(1), "copy_() does not support casting "
      "Float4_e2m1fn_x2 to different types. Source dtype is ", iter.dtype(1), "target dtype is ", dtype);
    gpu_kernel_nocast(iter, [] GPU_LAMBDA(Float4_e2m1fn_x2 x) { return x; });
  } else {
    AT_DISPATCH_V2(
        dtype, "copy_", AT_WRAP([&] {
          gpu_kernel(iter, [] GPU_LAMBDA(scalar_t x) { return x; });
    }), AT_EXPAND(AT_ALL_TYPES_AND_COMPLEX), kHalf, kBool, kBFloat16, kComplexHalf, kBComplex32, AT_EXPAND(AT_BAREBONES_UNSIGNED_TYPES));
  }
}

void neg_conj_kernel_cuda(TensorIteratorBase &iter) {
  AT_DISPATCH_COMPLEX_TYPES(iter.common_dtype(), "neg_conj_cuda", [&] {
    gpu_kernel(iter, [] GPU_LAMBDA(scalar_t x) { return -std::conj(x); });
  });
}

using namespace at::cuda;

namespace {

constexpr int kTransposeTile = 32;
constexpr int kTransposeRows = 8;

// Shared-memory banks are 4 bytes wide, so what must be coprime with 32 is
// the tile row stride measured in 32-bit words, not in elements. Padding by
// one element only achieves that for 4-byte types; 1- and 2-byte types need
// a wider pad. For 8-byte types a warp's access splits into two 16-lane
// phases that already cover all 32 banks.
template <typename T>
struct TransposeTilePad {
  static constexpr int value = sizeof(T) == 1 ? 4    // 36 B = 9 words
                             : sizeof(T) == 2 ? 2    // 68 B = 17 words
                                              : 1;   // 4 B: 33 words
};

template <typename T, bool kGridStride>
__global__ void transpose_copy_tiled_kernel(
    const T* __restrict__ src, T* __restrict__ dst,
    int64_t width, int64_t height,
    int64_t src_pitch, int64_t dst_pitch) {
  __shared__ T tile[kTransposeTile][kTransposeTile + TransposeTilePad<T>::value];

  // gridDim.y is capped at 65535 on every compute capability, so walk the
  // tile rows with a grid-stride loop instead of mapping them 1:1 to blocks.
  const int64_t tiles_y = (height + kTransposeTile - 1) / kTransposeTile;

  int64_t by = blockIdx.y;
  do {
    int64_t x = static_cast<int64_t>(blockIdx.x) * kTransposeTile + threadIdx.x;
    int64_t y = by * kTransposeTile + threadIdx.y;

    for (int j = 0; j < kTransposeTile; j += kTransposeRows) {
      if (x < width && (y + j) < height) {
        tile[threadIdx.y + j][threadIdx.x] = src[(y + j) * src_pitch + x];
      }
    }
    __syncthreads();

    x = by * kTransposeTile + threadIdx.x;
    y = static_cast<int64_t>(blockIdx.x) * kTransposeTile + threadIdx.y;

    for (int j = 0; j < kTransposeTile; j += kTransposeRows) {
      if (x < height && (y + j) < width) {
        dst[(y + j) * dst_pitch + x] = tile[threadIdx.x][threadIdx.y + j];
      }
    }
    // With a single pass there is no next iteration to guard
    if (!kGridStride) break;
    // Required before the next iteration overwrites the tile.
    __syncthreads();
    by += gridDim.y;
  } while (by < tiles_y);
}

template <typename T>
void launch_tiled_transpose(bool needs_stride, dim3 grid, dim3 block,
                            cudaStream_t stream, const void* sp, void* dp,
                            int64_t w, int64_t h,
                int64_t src_pitch, int64_t dst_pitch) {
  if (needs_stride) {
    transpose_copy_tiled_kernel<T, true><<<grid, block, 0, stream>>>(
        reinterpret_cast<const T*>(sp), reinterpret_cast<T*>(dp), w, h, src_pitch, dst_pitch);
  } else {
    transpose_copy_tiled_kernel<T, false><<<grid, block, 0, stream>>>(
        reinterpret_cast<const T*>(sp), reinterpret_cast<T*>(dp), w, h, src_pitch, dst_pitch);
  }
}

bool maybe_tiled_transpose_copy(TensorIterator& iter) {
  if (iter.ndim() != 2) return false;
  const int64_t es = iter.element_size(0);

  auto shape = iter.shape();
  auto os = iter.strides(0);
  auto is = iter.strides(1);
  const int64_t h = shape[0];
  const int64_t w = shape[1];

  if (os[0] != es || is[1] != es) return false;
  if (os[1] < es * h || is[0] < es * w) return false;
  const int64_t dst_pitch = os[1] / es;   // elements between dst rows
  const int64_t src_pitch = is[0] / es;   // elements between src rows
  if (os[1] % es != 0 || is[0] % es != 0) return false;

  // Below ~4 MB the tiled path and the generic strided kernel are
  // indistinguishable above kernel-launch overhead (measured on sm_89 with
  // L2 flushing and 3 repeats; every smaller size showed >4% run-to-run
  // variance). Above it the tiled path was faster in every measured case
  // except fp64 with a very short output row, where the strided kernel
  // already reaches peak bandwidth and tiling costs ~4%.
  if (h * w * es < (int64_t(4) << 20)) return false;

  const int64_t kMaxGridY = at::cuda::getCurrentDeviceProperties()->maxGridSize[1];
  const int64_t tiles_x = (w + kTransposeTile - 1) / kTransposeTile;
  const int64_t tiles_y = (h + kTransposeTile - 1) / kTransposeTile;
  dim3 block(kTransposeTile, kTransposeRows);
  const bool needs_stride = tiles_y > kMaxGridY;
  dim3 grid((unsigned)tiles_x,
            (unsigned)(needs_stride ? kMaxGridY : tiles_y));
  auto stream = at::cuda::getCurrentCUDAStream();
  const void* sp = iter.tensor(1).const_data_ptr();
  void* dp = iter.tensor(0).mutable_data_ptr();

  switch (es) {
    case 1: launch_tiled_transpose<uint8_t>(needs_stride, grid, block, stream, sp, dp, w, h, src_pitch, dst_pitch); break;
    case 2: launch_tiled_transpose<uint16_t>(needs_stride, grid, block, stream, sp, dp, w, h, src_pitch, dst_pitch); break;
    case 4: launch_tiled_transpose<uint32_t>(needs_stride, grid, block, stream, sp, dp, w, h, src_pitch, dst_pitch); break;
    case 8: launch_tiled_transpose<uint64_t>(needs_stride, grid, block, stream, sp, dp, w, h, src_pitch, dst_pitch); break;
    default: return false;
  }

  C10_CUDA_KERNEL_LAUNCH_CHECK();
  return true;
}

} // namespace

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

    // Use event pool for better performance instead of creating new events
    auto dst_ready = getEventFromPool(dst_device.index());
    device_guard.set_device(dst_device);
    dst_ready->record(getCurrentCUDAStream(dst_device.index()));

    device_guard.set_device(src_device);
    dst_ready->block(copy_stream);
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
    if (same_type && same_neg && same_conj && maybe_tiled_transpose_copy(iter)) {
      // handled by the tiled transpose kernel
    } else if (same_neg) {
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
    auto src_ready = getEventFromPool(src_device.index());
    src_ready->record(copy_stream);

    device_guard.set_device(dst_device);
    src_ready->block(getCurrentCUDAStream(dst_device.index()));
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
  const Tensor* host_tensor = nullptr;
  if (dst_device.is_cuda() && src_device.is_cpu()) {
    device_guard.set_device(dst_device);
    kind = cudaMemcpyHostToDevice;
    host_tensor = &iter.tensor(1);
  } else if (dst_device.is_cpu() && src_device.is_cuda()) {
    device_guard.set_device(src_device);
    kind = cudaMemcpyDeviceToHost;
    host_tensor = &iter.tensor(0);
  } else {
    TORCH_INTERNAL_ASSERT(false, "unsupported devices in GPU copy_()");
  }

  // Check for unpinned CPU memory during CUDA graph capture
  if (at::cuda::currentStreamCaptureStatus() != at::cuda::CaptureStatus::None) {
    TORCH_CHECK(
        host_tensor->is_pinned(),
        "Cannot copy between CPU and CUDA tensors during CUDA graph capture ",
        "unless the CPU tensor is pinned. Please use tensor.pin_memory() or ",
        "allocate the tensor with pin_memory=True.");
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
    at::getHostAllocator(at::kCUDA)->record_event(ptr, ctx, stream.unwrap());
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
