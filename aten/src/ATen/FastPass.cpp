// FastPass
#ifdef _MSC_VER
#ifndef _USE_MATH_DEFINES
#define _USE_MATH_DEFINES
#endif
#include <math.h>
#endif

#include <ATen/ATen.h>
#include <ATen/native/TensorFactories.h>
#include <ATen/native/Fill.h>

// Fast pass functions can be directly referenced from outside of ATen/native
// without dispatching. They are majorly for performance improvement. Keep them
// in this file to be always built in low-level target (eg. aten_cpu), so that
// other higher-level components (native, jit, autograd, etc.) can directly
// reference the symbols, but not the other way.

namespace at {
namespace native {

// ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ empty ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
Tensor empty_cpu(IntArrayRef size, const TensorOptions& options_, c10::optional<c10::MemoryFormat> optional_memory_format) {

  TORCH_CHECK(
      !(options_.has_memory_format() && optional_memory_format.has_value()),
      "Cannot set memory_format both in TensorOptions and explicit argument; please delete "
      "the redundant setter.");
  TensorOptions options = options_.merge_in(TensorOptions().memory_format(optional_memory_format));

  AT_ASSERT(options.device().type() == DeviceType::CPU);
  TORCH_INTERNAL_ASSERT(impl::variable_excluded_from_dispatch());
  check_size_nonnegative(size);

  c10::Allocator* allocator;
  if (options.pinned_memory()) {
    allocator = detail::getCUDAHooks().getPinnedMemoryAllocator();
  } else {
    allocator = at::getCPUAllocator();
  }

  int64_t nelements = prod_intlist(size);
  auto dtype = options.dtype();
  int64_t size_bytes = nelements * dtype.itemsize();
  auto storage_impl = c10::make_intrusive<StorageImpl>(
      c10::StorageImpl::use_byte_size_t(),
      size_bytes,
      allocator->allocate(size_bytes),
      allocator,
      /*resizeable=*/true);

  auto tensor = detail::make_tensor<TensorImpl>(
      std::move(storage_impl), at::DispatchKey::CPU, dtype);
  // Default TensorImpl has size [0]
  if (size.size() != 1 || size[0] != 0) {
    tensor.unsafeGetTensorImpl()->set_sizes_contiguous(size);
  }

  auto memory_format = options.memory_format_opt().value_or(MemoryFormat::Contiguous);
  tensor.unsafeGetTensorImpl()->empty_tensor_restride(memory_format);

  return tensor;
}

namespace {
template <typename scalar_t>
inline void fill_fast(Tensor& self, Scalar value_scalar) {
  auto value = value_scalar.to<scalar_t>();
  scalar_t * dptr = static_cast<scalar_t *>(self.data_ptr());
  *dptr = value;
}
} // namspace
// ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ fill ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

DEFINE_DISPATCH(fill_stub);

Tensor& fill_out(Tensor& self, Scalar value) {
  if (self.is_quantized()) {
    at::Tensor out = at::ones(self.sizes()).to(kFloat) * value;
    out = out.to(self.device());
    // Trust the `copy_` to handle the quantization and the boundary chacks.
    self.copy_(out);
    return self;
  }
  // When filling a number to 1-element CPU tensor, we want to skip
  // everything but manipulate data ptr directly.
  // Ideally this fast pass should be implemented in TensorIterator,
  // but we also want to skip compute_types which in not avoidable
  // in TensorIterator for now.
  if (self.device() == at::kCPU && self.numel() == 1 && !self.is_complex() && !value.isComplex()) {
    AT_DISPATCH_ALL_TYPES_AND3(kHalf, kBool, kBFloat16, self.scalar_type(), "fill_out", [&]() {
      fill_fast<scalar_t>(self, value);});
    return self;
  }
  auto iter = TensorIteratorConfig()
      .set_check_mem_overlap(false)  // Fill is idempotent, so overlap is okay
      .check_all_same_dtype(false)
      .add_output(self)
      .resize_outputs(false)
      .build();
  fill_stub(iter.device_type(), iter, value);
  return self;
}

Tensor& fill_(Tensor& self, const Tensor& value) {
  TORCH_CHECK(value.dim() == 0, "fill_ only supports 0-dimension value tensor but got tensor with ", value.dim(), " dimensions.");
  return fill_out(self, value.item());
}

Tensor& fill_(Tensor& self, Scalar value) {
  return fill_out(self, value);
}

// ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ scalar_tensor ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Tensor scalar_tensor(Scalar s, const TensorOptions& options) {
  if (options.device() == at::kCPU) {
    // This is a fast track to skip device dispatch for making scalar tensor on CPU.
    // See https://github.com/pytorch/pytorch/pull/29915 for more detailed perf
    // difference.
    // In the future when we remove the overhead of device dispatch, we'll happily
    // revert this to following:
    //   auto result = at::empty({}, options);
    at::tracer::impl::NoTracerDispatchMode tracer_guard;
    at::AutoNonVariableTypeMode non_var_type_mode(true);
    auto result = empty_cpu({}, options);
    at::native::fill_(result, s);
    return result;
  }
  return at::empty({}, options).fill_(s);
}


} // namespace native
} // namespace at
