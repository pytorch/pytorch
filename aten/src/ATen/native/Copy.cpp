#define TORCH_ASSERT_ONLY_METHOD_OPERATORS
#include <ATen/native/Copy.h>

#include <ATen/core/Tensor.h>
#include <ATen/Dispatch.h>
#include <ATen/FunctionalTensorWrapper.h>
#include <ATen/TensorIterator.h>
#include <ATen/native/quantized/Copy.h>
#include <ATen/native/mps/Copy.h>
#include <ATen/native/vulkan/ops/Copy.h>
#include <ATen/native/TensorShape.h>
#include <ATen/quantized/Quantizer.h>
#include <ATen/vulkan/Context.h>
#include <ATen/metal/Context.h>
#include <ATen/NamedTensorUtils.h>
#include <ATen/Parallel.h>
#include <c10/util/irange.h>

#ifndef AT_PER_OPERATOR_HEADERS
#include <ATen/Functions.h>
#include <ATen/NativeFunctions.h>
#else
#include <ATen/ops/_copy_from.h>
#include <ATen/ops/_propagate_xla_data.h>
#include <ATen/ops/_propagate_xla_data_native.h>
#include <ATen/ops/copy_native.h>
#include <ATen/ops/empty.h>
#include <ATen/ops/expand_copy.h>
#endif

#ifdef USE_FBGEMM
#include <fbgemm/Fbgemm.h>
#include <fbgemm/FbgemmConvert.h>
#endif

namespace {

using namespace at;

bool copy_transpose_valid(const Tensor& self, const Tensor& src) {
  const int MIN_SZ = 60 * 60;
  return self.is_contiguous() && src.numel() != 0 && src.dim() == 2 &&
      src.stride(0) == 1 && src.stride(1) == src.size(0) &&
      self.scalar_type() == src.scalar_type() &&
      self.sizes().equals(src.sizes()) &&
      self.is_neg() == src.is_neg() &&
      self.is_conj() == src.is_conj() &&
      self.numel() >= MIN_SZ;
}

// special case copy where tensor is contiguous and src is a transposed matrix
// This can be generalized to most copies, but it's trickier
void copy_same_type_transpose_(Tensor& self, const Tensor& src) {
  // NOLINTNEXTLINE(cppcoreguidelines-init-variables)
  int64_t BLOCK_SZ;
  if (self.scalar_type() == kByte) {
    // NOLINTNEXTLINE(cppcoreguidelines-avoid-magic-numbers)
    BLOCK_SZ = 120;
  } else {
    // NOLINTNEXTLINE(cppcoreguidelines-avoid-magic-numbers)
    BLOCK_SZ = 60;
  }
  Tensor buf = empty({BLOCK_SZ, BLOCK_SZ}, self.options());

  // The code below is implemented with the assumption that sizes are equal
  TORCH_INTERNAL_ASSERT_DEBUG_ONLY(self.sizes().equals(src.sizes()));

  AT_DISPATCH_ALL_TYPES_AND_COMPLEX_AND4(kHalf, kBool, kBFloat16, kComplexHalf, self.scalar_type(), "copy_", [&] {
    scalar_t* sp = src.data_ptr<scalar_t>();
    scalar_t* rp = self.data_ptr<scalar_t>();
    scalar_t* bp = buf.data_ptr<scalar_t>();

    int64_t NR = src.size(0);
    int64_t NC = src.size(1);
    for (int64_t R = 0; R < NR; R += BLOCK_SZ) {
      for (int64_t C = 0; C < NC; C += BLOCK_SZ) {
        scalar_t* spo = sp + R + C * NR;
        scalar_t* rpo = rp + C + R * NC;

        int nr = std::min(NR - R, BLOCK_SZ);
        int nc = std::min(NC - C, BLOCK_SZ);

        // 1. copy columns from src to buf
        for (const auto c : c10::irange(nc)) {
          memcpy(bp + c * BLOCK_SZ, spo + c * NR, nr * sizeof(scalar_t));
        }

        // 2. transpose buf in place
        int rc_max = std::max(nr, nc);
        int rc_min = std::min(nr, nc);
        for (const auto r : c10::irange(rc_max)) {
          int end = std::min(r, rc_min);
          for (const auto c : c10::irange(end)) {
            scalar_t tmp = bp[r + BLOCK_SZ * c];
            bp[r + BLOCK_SZ * c] = bp[r * BLOCK_SZ + c];
            bp[r * BLOCK_SZ + c] = tmp;
          }
        }

        // 3. copy rows from buf to dst
        for (const auto r : c10::irange(nr)) {
          memcpy(rpo + r * NC, bp + r * BLOCK_SZ, nc * sizeof(scalar_t));
        }
      }
    }
  });
}

// Devices directly supported by this copy implementation. Other device types
// (e.g. XLA) may be supported by overriding copy_ and _copy_from.
bool is_supported_device(Device device) {
  DeviceType device_type = device.type();
  return device_type == kCPU || device_type == kCUDA || device_type == kHIP || device_type == kVulkan || device_type == kMetal || device_type == kMPS;
}

} // namespace

namespace at {
namespace native {

static Tensor & copy_impl(Tensor & self, const Tensor & src, bool non_blocking) {
  // TODO: this should be handled during dispatch, but that's missing...
  TORCH_CHECK(self.defined(), "self is undefined");
  TORCH_CHECK(src.defined(), "src is undefined");

  // FBGeMM kernel support exists only for the following case,
  // 1. Memory Format for source and destination tensors is contiguous.
  // 2. Device for both the source and destination tensor is CPU.
  // 3. dtype conversion between FP32->FP16 and FP16->FP32.
  // This checks that self.sizes() == src.sizes() because this code path doesn't
  // support broadcasting. This also guards against out of bounds memory access
  // when copying, see fbgemm::Float16ToFloat_ref.
  // https://github.com/pytorch/pytorch/issues/88543
  #ifdef USE_FBGEMM
    if (((self.dtype() == at::kFloat && src.dtype() == at::kHalf) ||
         (self.dtype() == at::kHalf && src.dtype() == at::kFloat)) &&
        (self.device().is_cpu() && src.device().is_cpu()) &&
        ((self.is_contiguous() && src.is_contiguous()) ||
         (self.is_non_overlapping_and_dense() && self.strides() == src.strides())) &&
        (self.sizes() == src.sizes())) {
      if (src.dtype() == at::kFloat && self.dtype() == at::kHalf) {
        auto* output_ptr =
            reinterpret_cast<fbgemm::float16*>(self.data_ptr<at::Half>());
        if (self.numel() < at::internal::GRAIN_SIZE) {
          fbgemm::FloatToFloat16_simd(src.data_ptr<float>(), output_ptr, self.numel());
        } else {
          at::parallel_for(
              0,
              self.numel(),
              at::internal::GRAIN_SIZE,
              [&](int64_t begin, int64_t end) {
                fbgemm::FloatToFloat16_simd(
                    src.data_ptr<float>() + begin,
                    output_ptr + begin,
                  end - begin);
              });
        }
      } else {
        auto in_data = reinterpret_cast<fbgemm::float16*>(
            src.data_ptr<at::Half>());
        auto* output_ptr = self.data_ptr<float>();
        if (self.numel() < at::internal::GRAIN_SIZE) {
          fbgemm::Float16ToFloat_simd(in_data, output_ptr, self.numel());
        } else {
          at::parallel_for(
              0,
              self.numel(),
              at::internal::GRAIN_SIZE,
              [&](int64_t begin, int64_t end) {
                fbgemm::Float16ToFloat_simd(
                    in_data + begin, output_ptr + begin, end - begin);
              });
        }
      }
      return self;
    }
  #endif

  if (self.is_same(src)) {
    return self;
  }

  // Copies into meta self are OK and just ignored (similar to inplace)
  if (self.is_meta()) {
    // TODO: need to see if there is extra error checking needed
    return self;
  }

  if (src.is_meta()) {
    TORCH_CHECK_NOT_IMPLEMENTED(false, "Cannot copy out of meta tensor; no data!")
  }

  // Re-dispatch copies when either src or self device not implemented here (e.g. XLA).
  // _copy_from has a proper device dispatch setup.
  // This includes:
  //   cpu_tensor.copy_(xla_tensor) => xla_tensor._copy_from(cpu_tensor)
  //   xla_tensor.copy_(cpu_tensor) => cpu_tensor._copy_from(xla_tensor)
  // Both the _copy_from calls above will be dispatched to XLA's _copy_from kernels.
  if (!is_supported_device(src.device()) || !is_supported_device(self.device())) {
    at::_copy_from(src, self, non_blocking);
    return self;
  }

  if (self.is_quantized() && !src.is_quantized()) {
    return quantized_copy_from_float_(self, src);
  }

  if (self.is_quantized() && src.is_quantized()) {
    TORCH_CHECK(self.qscheme() == src.qscheme(),
                "Quantized Copy only works with same qscheme");
    TORCH_CHECK(self.scalar_type() == src.scalar_type());
    set_quantizer_(self, src.quantizer());
  }

  if (!self.is_quantized() && src.is_quantized()) {
    TORCH_CHECK(false, "Copying from quantized Tensor to non-quantized Tensor is not allowed, please use dequantize to get a float Tensor from a quantized Tensor");
  }

  if (self.device().type() == at::kVulkan || src.device().type() == at::kVulkan) {
  #ifdef USE_VULKAN_API
    return vulkan::ops::copy_(self, src);
  #else
    return at::vulkan::vulkan_copy_(self, src);
  #endif
  }

  if (self.device().type() == at::kMetal || src.device().type() == at::kMetal) {
    return at::metal::metal_copy_(self, src);
  }

  // Exit early if self and src are views of the same data
  const bool is_same_data = (
      self.is_alias_of(src) &&
      self.storage_offset() == src.storage_offset() &&
      self.strides().equals(src.strides()) &&
      self.sizes().equals(src.sizes()) &&
      self.scalar_type() == src.scalar_type()
    );
  if (is_same_data) {
    return self;
  }


  auto iter = TensorIteratorConfig()
    .add_output(self)
    .add_input(src)
    .resize_outputs(false)
    .check_all_same_dtype(false)
    .check_all_same_device(false)
    .build();

  if (iter.numel() == 0) {
    return self;
  }

  DeviceType device_type = iter.device_type(0);
  if (iter.device_type(1) == kCUDA) {
    device_type = kCUDA;
  } else if (iter.device_type(1) == kHIP) {
    device_type = kHIP;
  } else if (iter.device_type(1) == kMPS) {
    device_type = kMPS;
  }

  // TODO: if we need to, we can also enable this path for quantized tensor
  if (device_type == kCPU && copy_transpose_valid(self, src) && !self.is_quantized()) {
    copy_same_type_transpose_(self, src);
    return self;
  }

#ifdef USE_MPS
  if (self.device().type() == at::kMPS || src.device().type() == at::kMPS) {
    return at::native::mps::mps_copy_(self, src, non_blocking);
  }
#endif

  if(!self.is_complex() && src.is_complex()) {
    TORCH_WARN_ONCE("Casting complex values to real discards the imaginary part");
  }
  copy_stub(device_type, iter, non_blocking);
  return self;
}

Tensor copy(const Tensor& self, const Tensor& src, bool non_blocking) {
  // copy() is the "functional" form of copy_(). It exists so we can properly functionalize copy_(), but:
  // (1) It isn't exposed to the frontend (no python bindings)
  // (2) It isn't exposed to the backend (it's a composite, that decomposes into to() and expand_as() calls.
  auto r = clone_preserve_strides(self);
  r.copy_(src, non_blocking);
  return r;
}

Tensor& copy_(Tensor& self, const Tensor& src, bool non_blocking) {
  auto maybe_outnames = namedinference::compute_broadcast_outnames(self, src);
  {
    NoNamesGuard guard;
    if (self._is_zerotensor()) {
     TORCH_CHECK(false, "ZeroTensors are immutable. Please materialize the tensor using `.clone()`, if you want a mutable zero tensor.");
    }
    if (src._is_zerotensor()) {
      return self.zero_();
    }
    copy_impl(self, src, non_blocking);
  }
  namedinference::propagate_names_if_nonempty(self, maybe_outnames);
  return self;
}

void copy_ignoring_overlaps(const TensorBase &dst, const TensorBase &src) {
  // Called when we are copying into an overlapping index `dst`, but we don't
  // care which writer wins. Hacky but it works. This is only used by
  // CUDA_tensor_apply2 in case that there are write overlaps.
  // FIXME: really, overlapping writes should be illegal/an error in Torch
  auto iter = TensorIteratorConfig()
      .add_output(dst)
      .add_input(src)
      .resize_outputs(false)
      .set_check_mem_overlap(false)
      .check_all_same_dtype(true)
      .check_all_same_device(true)
      .build();
  copy_stub(iter.device_type(), iter, /*non_blocking=*/false);
}

void _propagate_xla_data(const Tensor& input, const Tensor& output) {
  TORCH_INTERNAL_ASSERT(input.device().type() == kXLA, "This op should only be called by XLA")
}

DEFINE_DISPATCH(copy_stub);

} // namespace native
} // namespace at
