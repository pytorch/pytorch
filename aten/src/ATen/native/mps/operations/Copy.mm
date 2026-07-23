//  Copyright © 2022 Apple Inc.
#define TORCH_ASSERT_ONLY_METHOD_OPERATORS
#include <ATen/TensorIterator.h>
#include <ATen/mps/MPSAllocatorInterface.h>
#include <ATen/mps/MPSProfiler.h>
#include <ATen/native/mps/Copy.h>
#include <ATen/native/mps/OperationUtils.h>
#include <ATen/ops/_copy_from_and_resize_native.h>
#include <ATen/ops/_copy_from_native.h>
#include <ATen/ops/imag.h>
#include <ATen/ops/neg.h>
#include <ATen/ops/real.h>
#include <ATen/ops/view_as_real.h>
#include <ATen/ops/zeros_like.h>
#include <fmt/format.h>

namespace at::native {
#ifndef PYTORCH_JIT_COMPILE_SHADERS
static auto& lib = mps::MetalShaderLibrary::getBundledLibrary();
#else
#include <ATen/native/mps/Copy_metallib.h>
#endif

namespace mps {

// MPS->MPS dtype-and/or-conj/neg-bit copy via the unary-kernel cast path. One kernel per output dtype handles every
// input dtype the runtime switch covers; the (needs_conj, needs_neg) pair selects the functor. Conj+neg uses a fused
// copy_conj_neg functor for complex (the only case where it differs from neg); real types route through copy_neg.
// Strided iterators get the `_strided_castout_<in>` template; contiguous get `_dense_castout_*`.
static void copy_cast_kernel_mps(at::Tensor& dst, const at::Tensor& src) {
  const bool needs_conj = src.is_conj() != dst.is_conj();
  const bool needs_neg = src.is_neg() != dst.is_neg();

  // Strip conj/neg bits via aliases so TensorIterator reads/writes raw storage; the functor materializes the requested
  // bit flips. alias() gives a fresh TensorImpl over the same storage so we don't mutate the caller's tensors.
  Tensor src_view = src.alias();
  src_view._set_conj(false);
  src_view._set_neg(false);
  Tensor dst_view = dst.alias();
  dst_view._set_conj(false);
  dst_view._set_neg(false);

  auto build_iter = [&](at::Tensor& out) {
    return at::TensorIteratorConfig()
        .check_all_same_dtype(false)
        .set_check_mem_overlap(false)
        .resize_outputs(false)
        .add_output(out)
        .add_input(src_view)
        .build();
  };

  // conj is identity on real types, so conj+neg degenerates to plain neg; only complex needs the fused functor.
  const bool fused_conj_neg = needs_conj && needs_neg && c10::isComplexType(src_view.scalar_type());
  const std::string_view name = fused_conj_neg ? "copy_conj_neg"
      : needs_neg                              ? "copy_neg"
      : needs_conj                             ? "copy_conj"
                                               : "copy_identity";
  auto iter = build_iter(dst_view);
  // ILP castout only wins past ~128K elements; smaller copies underfill the GPU and
  // run faster through the plain (non-ILP) castout kernel.
  lib.exec_unary_kernel(iter, std::string(name), std::nullopt, std::nullopt, /*ilp_threshold=*/131072u);
}

// Byte-erased compute copy, not a blit: faster at small sizes and avoids the encoder switch.
// One dispatch per <=2GB chunk keeps chunk_bytes in uint. Callers must pass contiguous src and
// dst with equal nbytes (both are treated as flat byte runs).
static void contiguous_copy_kernel_mps(at::Tensor& dst, const at::Tensor& src, bool non_blocking) {
  uint64_t profile_id = getMPSProfiler().beginProfileCopy(
      getMTLBufferStorage(src), getMTLBufferStorage(dst), src, dst, src.nbytes(), non_blocking, /*usesBlitter=*/false);
  auto* kernel = lib.getCachedKernelFunctionPtr("contiguous_byte_copy");
  constexpr size_t max_chunk = 0x80000000; // 2GB
  const size_t total = src.nbytes();
  kernel->runCommandBlock([&] {
    kernel->startEncoding();
    kernel->setArg(0, dst);
    kernel->setArg(1, src);
    for (size_t base = 0; base < total;) {
      const uint32_t chunk = static_cast<uint32_t>(std::min(max_chunk, total - base));
      kernel->setArg(2, chunk);
      kernel->setArg(3, static_cast<uint64_t>(base));
      kernel->dispatch((chunk + 15) / 16);
      base += chunk;
    }
  });
  if (profile_id) {
    getMPSProfiler().endProfileCopy(profile_id, SyncType::NONE);
  }
}

static void* pageAlignedBlockPtr(const void* ptr, NSUInteger size, NSUInteger* alignedBlockSize) {
  uintptr_t address = (uintptr_t)ptr;
  uintptr_t alignedAddress = address & ~(PAGE_SIZE - 1);
  uintptr_t alignedEnd = ((address + size) + PAGE_SIZE - 1) & ~(PAGE_SIZE - 1);
  uint64_t alignedLength = alignedEnd - alignedAddress;

  assert(address >= alignedAddress);
  assert(address + size <= alignedAddress + alignedLength);

  *alignedBlockSize = alignedLength;
  return (void*)alignedAddress;
}

// Returns an MTLBuffer and byte offset for the host side of a CPU<->MPS copy that
// the caller must not release: a pinned tensor's own shared MTLBuffer is returned
// borrowed (kept alive by the tensor), otherwise the host pages are wrapped in an
// autoreleased newBufferWithBytesNoCopy (valid for the caller's @autoreleasepool),
// retaining the storage across an async copy so the pages outlive the in-flight blit.
static std::pair<id<MTLBuffer>, NSUInteger> buffer_with_offset_from_tensor(const at::Tensor& cpu_tensor,
                                                                           size_t nbytes,
                                                                           bool non_blocking) {
  const auto byte_offset = cpu_tensor.storage_offset() * cpu_tensor.itemsize();
  // Blit directly from/to the pinned tensor's own shared MTLBuffer, avoiding the
  // newBufferWithBytesNoCopy wrapper. Metal blit offsets must be 4-byte aligned.
  if (void* pinned = at::mps::getMPSPinnedMTLBuffer(cpu_tensor.storage().data()); pinned && byte_offset % 4 == 0) {
    // Mark the buffer so that if it is freed while this copy's blit is still in
    // flight, the allocator defers recycling it instead of handing it to a new
    // allocation that could CPU-overwrite it before the GPU is done.
    at::mps::getIMPSAllocator()->recordEvents({pinned});
    return {__builtin_bit_cast(id<MTLBuffer>, pinned), static_cast<NSUInteger>(byte_offset)};
  }
  id<MTLDevice> device = MPSDevice::getInstance()->device();
  const void* host = static_cast<const char*>(cpu_tensor.storage().data()) + byte_offset;
  NSUInteger alignedLength = 0;
  void* alignedPtr = pageAlignedBlockPtr(host, (NSUInteger)nbytes, &alignedLength);
  // Only capture on non_blocking - capturing across waitUntilCompleted would
  // deadlock Metal's completion thread on the GIL.
  auto* storage = non_blocking ? new c10::Storage(cpu_tensor.storage()) : nullptr;
  MTLResourceOptions options = MTLResourceCPUCacheModeDefaultCache | MTLResourceStorageModeShared;
  id<MTLBuffer> buffer = [[device newBufferWithBytesNoCopy:alignedPtr
                                                    length:alignedLength
                                                   options:options
                                               deallocator:^(void*, NSUInteger) {
                                                 delete storage;
                                               }] autorelease];
  return {buffer, static_cast<NSUInteger>(uintptr_t(host) - uintptr_t(alignedPtr))};
}

static at::Tensor& copy_from_mps_(at::Tensor& dst_, const at::Tensor& src_, bool non_blocking) {
  auto sameMemFormat =
      src_.is_contiguous(dst_.suggest_memory_format()) && dst_.is_contiguous(dst_.suggest_memory_format());

  MPSStream* stream = getCurrentMPSStream();
  Tensor dst = dst_;
  Tensor src = src_;

  // Equal strides alone don't make the flat blit/castout below valid: both
  // sides must also map a contiguous storage segment. Views like x[::2] share
  // strides yet have holes, so a flat copy reads/writes the wrong bytes and
  // clobbers out-of-view storage (the CPU-to-MPS direction already guards
  // this with is_dense_in_storage). Gather/scatter through contiguous
  // temporaries in that case.
  const bool direct_copy = dst_.strides() == src_.strides() && is_dense_in_storage(src_);
  if (!direct_copy) {
    dst = at::empty_like(dst_, LEGACY_CONTIGUOUS_MEMORY_FORMAT);
  }

  auto storage_byte_offset = src_.storage_offset() * src_.itemsize();
  if (!direct_copy) {
    Tensor emptyShell = Tensor();
    src = gatherViewTensor(src_, emptyShell);
    if (src.has_storage()) {
      storage_byte_offset = 0;
    } else {
      src = src_.expand_as(dst).contiguous();
      storage_byte_offset = src.storage_offset() * src.itemsize();
    }
  }

  id<MTLBuffer> sourceBuffer = getMTLBufferStorage(src);
  size_t dst_tensor_nbytes = dst.nbytes();

  @autoreleasepool {
    auto [destBuffer, destOffset] = buffer_with_offset_from_tensor(dst, dst_tensor_nbytes, non_blocking);
    // 4 bytes alignment required on macos for blits.
    TORCH_INTERNAL_ASSERT(destOffset % 4 == 0, "Unaligned blit request");

    id<MTLBuffer> blitSourceBuffer = sourceBuffer;
    Tensor blitSource = src;
    NSUInteger blitSourceOffset = storage_byte_offset;
    bool needsBlit = true;
    if (src_.dtype() != dst.dtype()) {
      // Unified memory: cast straight from the MPS source into the CPU-wrapped
      // destination buffer at the requested offsets. This avoids the temporary
      // that used to alias the live source buffer and blitting from it (see
      // #189563). src and dst are dense with identical strides here, so a linear
      // castout of numel elements from the source offset to the dest offset is a
      // faithful conversion.
      needsBlit = false;
      const bool needs_conj = src.is_conj() != dst.is_conj();
      const bool needs_neg = src.is_neg() != dst.is_neg();
      const bool fused_conj_neg = needs_conj && needs_neg && c10::isComplexType(src.scalar_type());
      const std::string_view name = fused_conj_neg ? "copy_conj_neg"
          : needs_neg                              ? "copy_neg"
          : needs_conj                             ? "copy_conj"
                                                   : "copy_identity";
      lib.exec_unary_kernel_raw(name,
                                sourceBuffer,
                                static_cast<uint32_t>(storage_byte_offset),
                                src.scalar_type(),
                                destBuffer,
                                static_cast<uint32_t>(destOffset),
                                dst.scalar_type(),
                                static_cast<uint32_t>(src.numel()),
                                /*ilp_threshold=*/0u);
      if (!non_blocking) {
        stream->synchronize(SyncType::COMMIT_AND_WAIT);
      }
    }

    if (needsBlit) {
      const size_t size_to_copy = (src.nbytes() / src.element_size()) * dst.element_size();

      // If there's anything wrong with source, we shouldn't return dst_ silently and must error out.
      TORCH_INTERNAL_ASSERT(blitSourceBuffer && dst_tensor_nbytes > 0);
      uint64_t profile_id =
          getMPSProfiler().beginProfileCopy(blitSourceBuffer, destBuffer, blitSource, dst, size_to_copy, non_blocking);

      stream->copy_and_sync(
          blitSourceBuffer, destBuffer, size_to_copy, blitSourceOffset, destOffset, non_blocking, profile_id);
    }
  }
  if (!dst.is_same(dst_)) {
    dst_.copy_(dst, non_blocking);
  }

  return dst_;
}

// Copies tensor from cpu to mps backed by identical strided-contiguous data
static void copy_to_mps_stride_contig(at::Tensor& dst, const at::Tensor& src, bool non_blocking) {
  MPSStream* stream = getCurrentMPSStream();
  auto dst_byte_offset = dst.storage_offset() * dst.itemsize();
  id<MTLBuffer> destBuffer = getMTLBufferStorage(dst);
  const size_t size_to_copy = src.nbytes();

  TORCH_INTERNAL_ASSERT(src.dtype() == dst.dtype() && src.strides() == dst.strides() && is_dense_in_storage(src));

  @autoreleasepool {
    auto [sourceBuffer, sourceOffset] = buffer_with_offset_from_tensor(src, size_to_copy, non_blocking);
    uint64_t profile_id =
        getMPSProfiler().beginProfileCopy(sourceBuffer, destBuffer, src, dst, size_to_copy, non_blocking);

    stream->copy_and_sync(
        sourceBuffer, destBuffer, size_to_copy, sourceOffset, dst_byte_offset, non_blocking, profile_id);
  }
}

static at::Tensor& copy_to_mps_(at::Tensor& dst_, const at::Tensor& src_, bool non_blocking) {
  // Typecast to dst_ if needed and expand, which is a no-op
  Tensor src = (src_.dtype() != dst_.dtype() ? src_.to(dst_.dtype()) : src_).expand_as(dst_);

  // If src is not densely mapped in storage it must be cloned
  // It does not mean that tensor is contiguous, but rather
  // that it could be represented as 1d view
  if (!is_dense_in_storage(src)) {
    src = src.clone();
    TORCH_INTERNAL_ASSERT(is_dense_in_storage(src));
  }
  Tensor dst = dst_;
  bool needs_copy = false;
  // If src and dst_ strides do not match, it means that
  // either dst_ is not representable as 1d view or its stride order is different
  // in that case create an empty storage like src, copy it to device and then do
  // reshaping on the device
  if (src.strides() != dst_.strides()) {
    needs_copy = true;
    dst = at::empty_like(src, at::device(at::kMPS));
  }
  copy_to_mps_stride_contig(dst, src, non_blocking && !needs_copy);
  return needs_copy ? dst_.copy_(dst) : dst_;
}

void copy_blit_mps(void* dst, const void* src, size_t size) {
  // we don't have tensors info for profiling here
  uint64_t profile_id =
      getMPSProfiler().beginProfileCopy(src, dst, at::OptionalTensorRef(), at::OptionalTensorRef(), size, false);

  MPSStream* stream = getCurrentMPSStream();
  stream->copy_and_sync((id<MTLBuffer>)(src), (id<MTLBuffer>)(dst), size, 0, 0, true, profile_id);
}

static at::Tensor& copy_kernel_mps(at::Tensor& dst_, const at::Tensor& src_, bool non_blocking) {
  auto dst_byte_offset = dst_.storage_offset() * dst_.itemsize();

  // If dst is contiguous and there is no byte offset, we can save directly the result of
  // gather into dst. This reduces the overhead of doing an additional copy for most cases.
  bool returnGatherOutput = dst_.is_contiguous();
  Tensor src;
  auto sameMemFormat =
      src_.is_contiguous(dst_.suggest_memory_format()) && dst_.is_contiguous(dst_.suggest_memory_format());
  const bool sameDataType =
      src_.dtype() == dst_.dtype() && src_.is_conj() == dst_.is_conj() && src_.is_neg() == dst_.is_neg();

  if ((!src_.is_contiguous(MemoryFormat::Contiguous) && !sameMemFormat) ||
      // the copy_cast path requires storage_offset to be applied before casting
      (src_.storage_offset() && !sameDataType)) {
    Tensor emptyShell = Tensor();
    src = gatherViewTensor(src_, returnGatherOutput ? dst_ : emptyShell);

    if (src.has_storage()) {
      if (returnGatherOutput) {
        return dst_;
      }
    } else {
      src = src_.expand_as(dst_).contiguous();
    }
  } else {
    src = src_;
  }
  id<MTLBuffer> destBuffer = getMTLBufferStorage(dst_);

  // Strided dst can't be written as one contiguous run: route it through the scatter kernel.
  if (!dst_.is_contiguous(MemoryFormat::Contiguous) && !sameMemFormat) {
    return scatterViewTensor(src, dst_);
  }
  src._set_conj(src_.is_conj());
  src._set_neg(src_.is_neg());

  MPSStream* stream = getCurrentMPSStream();
  if (sameDataType) {
    contiguous_copy_kernel_mps(dst_, src, non_blocking);
  } else {
    if (dst_byte_offset) {
      auto maybeCastedSource =
          at::empty(dst_.sizes(), dst_.scalar_type(), std::nullopt, kMPS, std::nullopt, std::nullopt);
      auto maybeCastedSourceBuffer = getMTLBufferStorage(maybeCastedSource);
      copy_cast_kernel_mps(maybeCastedSource, src);

      uint64_t profile_id = getMPSProfiler().beginProfileCopy(
          maybeCastedSourceBuffer, destBuffer, maybeCastedSource, dst_, dst_.nbytes(), true);
      stream->copy(maybeCastedSourceBuffer, destBuffer, dst_.nbytes(), 0, dst_byte_offset, profile_id);
    } else {
      copy_cast_kernel_mps(dst_, src);
    }
  }
  return dst_;
}

at::Tensor& mps_copy_(at::Tensor& dst, const at::Tensor& src, bool non_blocking) {
  TORCH_CHECK(dst.defined(), "dst is undefined");
  TORCH_CHECK(src.defined(), "src is undefined");

  bool needs_broadcasting = false;

  if (src.numel() == 0 || dst.is_same(src)) {
    return dst;
  }
  if (dst.numel() == 0) {
    dst.resize_as_(src);
  }

  TORCH_CHECK(
      dst.dim() >= src.dim(), "Destination ", dst.sym_sizes(), " doesn't match the broadcast shape ", src.sym_sizes());
  if (dst.dim() > src.dim()) {
    needs_broadcasting = true;
  } else {
    const IntArrayRef src_sizes = src.sizes();
    const IntArrayRef dst_sizes = dst.sizes();
    for (const auto j : c10::irange(src.dim())) {
      if (src_sizes[j] == 1 && dst_sizes[j] != 1) {
        needs_broadcasting = true;
        break;
      }
    }
  }

  if (src.device().type() == at::kMPS && dst.device().type() == at::kCPU) {
    return copy_from_mps_(dst, needs_broadcasting ? src.expand_as(dst) : src, non_blocking);
  }
  if (src.device().type() == at::kCPU && dst.device().type() == at::kMPS) {
    return copy_to_mps_(dst, needs_broadcasting ? src.expand_as(dst) : src, non_blocking);
  }

  if (src.device().type() == at::kMPS && dst.device().type() == at::kMPS) {
    return copy_kernel_mps(dst, needs_broadcasting ? src.expand_as(dst) : src, non_blocking);
  }
  TORCH_INTERNAL_ASSERT(src.device().type() == DeviceType::MPS, "mps_copy_ is implemented only for *->MPS; MPS->*");
  return dst;
}

// Materialize a strided view into a contiguous tensor (or into the provided dst). conj/neg bits on src
// and dst are honored via copy_cast_kernel_mps's functor dispatch. Replaces the JIT scatter/gather
// shader machinery that previously lived in View.mm; the strided castout templates handle the same
// cross-dtype and bit-flip combinations.
Tensor gatherViewTensor(const at::Tensor& src, at::Tensor& dst) {
  Tensor output = dst.has_storage()
      ? dst
      : at::empty(src.sizes(), src.scalar_type(), std::nullopt, kMPS, std::nullopt, std::nullopt);
  if (src.numel() == 0 || output.numel() == 0) {
    return dst.has_storage() ? dst : output;
  }
  copy_cast_kernel_mps(output, src);
  return dst.has_storage() ? dst : output;
}

// Scatter a contiguous tensor into a strided view. Symmetrical to gatherViewTensor; the strided side is
// the destination here so the iterator's strided dimensions live on the output.
Tensor& scatterViewTensor(const at::Tensor& src, at::Tensor& output) {
  if (src.numel() == 0 || output.numel() == 0) {
    return output;
  }
  copy_cast_kernel_mps(output, src);
  return output;
}

} // namespace mps

Tensor _copy_from_and_resize_mps(const at::Tensor& self, const at::Tensor& dst) {
  const_cast<Tensor&>(dst).resize_as_(self);
  return mps::mps_copy_(const_cast<Tensor&>(dst), self, false);
}

Tensor _copy_from_mps(const at::Tensor& self, const at::Tensor& dst, bool non_blocking) {
  return mps::mps_copy_(const_cast<Tensor&>(dst), self, non_blocking);
}

} // namespace at::native
