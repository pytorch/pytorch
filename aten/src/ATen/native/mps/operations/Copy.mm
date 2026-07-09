//  Copyright © 2022 Apple Inc.
#define TORCH_ASSERT_ONLY_METHOD_OPERATORS
#include <ATen/TensorIterator.h>
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

// Returns an autoreleased MTLBuffer and byte offset for the host side of a
// CPU<->MPS copy; it stays valid for the caller's enclosing @autoreleasepool. The
// host pages are wrapped with newBufferWithBytesNoCopy, retaining the storage
// across an async copy so the pages outlive the in-flight blit.
static std::pair<id<MTLBuffer>, NSUInteger> buffer_with_offset_from_tensor(const at::Tensor& cpu_tensor,
                                                                           size_t nbytes,
                                                                           bool non_blocking) {
  const auto byte_offset = cpu_tensor.storage_offset() * cpu_tensor.itemsize();
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

  if (dst_.strides() != src_.strides()) {
    dst = at::empty_like(dst_, LEGACY_CONTIGUOUS_MEMORY_FORMAT);
  }

  auto storage_byte_offset = src_.storage_offset() * src_.itemsize();
  if (dst_.strides() != src_.strides()) {
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

    id<MTLBuffer> maybeCastedSourceBuffer = sourceBuffer;
    Tensor maybeCastedSource;
    bool needsBlit = true;
    if (src_.dtype() != dst.dtype()) {
      uint32_t dst_offs_for_cast = 0;
      if (destOffset == 0 && storage_byte_offset == 0) {
        // Return the casted tensor directly if there's no destination offset
        needsBlit = false;
        maybeCastedSourceBuffer = destBuffer;
      } else if (src.element_size() < dst.element_size()) {
        maybeCastedSource = at::empty(dst.sizes(), dst.scalar_type(), std::nullopt, kMPS, std::nullopt, std::nullopt);
        maybeCastedSourceBuffer = getMTLBufferStorage(maybeCastedSource);
      }

      // In case of dtype change, first convert src inplace via a Metal castout kernel writing into
      // maybeCastedSourceBuffer (which may be the wrapped CPU destBuffer, a fresh MPS temp, or
      // sourceBuffer itself when src.element_size() >= dst.element_size()).
      const bool needs_conj = src.is_conj() != dst.is_conj();
      const bool needs_neg = src.is_neg() != dst.is_neg();
      const bool fused_conj_neg = needs_conj && needs_neg && c10::isComplexType(src.scalar_type());
      const std::string_view name = fused_conj_neg ? "copy_conj_neg"
          : needs_neg                              ? "copy_neg"
          : needs_conj                             ? "copy_conj"
                                                   : "copy_identity";
      lib.exec_unary_kernel_raw(std::string(name),
                                sourceBuffer,
                                static_cast<uint32_t>(storage_byte_offset),
                                src.scalar_type(),
                                maybeCastedSourceBuffer,
                                dst_offs_for_cast,
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
      TORCH_INTERNAL_ASSERT(sourceBuffer && dst_tensor_nbytes > 0);
      uint64_t profile_id =
          getMPSProfiler().beginProfileCopy(sourceBuffer, destBuffer, src, dst, size_to_copy, non_blocking);

      stream->copy_and_sync(
          maybeCastedSourceBuffer, destBuffer, size_to_copy, storage_byte_offset, destOffset, non_blocking, profile_id);
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
  auto src_byte_offset = src_.storage_offset() * src_.itemsize();
  auto dst_byte_offset = dst_.storage_offset() * dst_.itemsize();

  // If dst is contiguous and there is no byte offset, we can save directly the result of
  // gather into dst. This reduces the overhead of doing an additional blit for most cases.
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

      src_byte_offset = 0;
    } else {
      src = src_.expand_as(dst_).contiguous();
      src_byte_offset = src.storage_offset() * src.itemsize();
    }
  } else {
    src = src_;
  }
  id<MTLBuffer> destBuffer = getMTLBufferStorage(dst_);
  id<MTLBuffer> sourceBuffer = getMTLBufferStorage(src);

  // Scatter to `dst` if the memory is not contiguous
  // If the memory is not contiguous, it means that the tensor has strides and we would not be
  // able to do the copy using a single blit
  if (!dst_.is_contiguous(MemoryFormat::Contiguous) && !sameMemFormat) {
    return scatterViewTensor(src, dst_);
  }
  src._set_conj(src_.is_conj());
  src._set_neg(src_.is_neg());

  MPSStream* stream = getCurrentMPSStream();
  if (sameDataType) {
    uint64_t profile_id = getMPSProfiler().beginProfileCopy(sourceBuffer, destBuffer, src, dst_, src.nbytes(), true);
    // for GPU to GPU copies we only encode to stream's command buffer (no flushing)
    stream->copy(sourceBuffer, destBuffer, src.nbytes(), src_byte_offset, dst_byte_offset, profile_id);
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
