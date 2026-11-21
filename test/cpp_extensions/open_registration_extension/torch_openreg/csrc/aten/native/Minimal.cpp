#include "Minimal.h"
#include "runtime/OpenRegGenerator.h"

#include <ATen/core/DistributionsHelper.h>
#include <c10/core/ScalarType.h>
#include <c10/core/TensorOptions.h>
#include <torch/headeronly/core/DeviceType.h>

#include <optional>
#include <unordered_set>

namespace at::native::openreg {

// LITERALINCLUDE START: EMPTY.MEMORY_FORMAT IMPL
at::Tensor empty_memory_format(
    c10::IntArrayRef size,
    std::optional<c10::ScalarType> dtype_opt,
    std::optional<c10::Layout> layout_opt,
    std::optional<c10::Device> device_opt,
    std::optional<bool> pin_memory_opt,
    std::optional<c10::MemoryFormat> memory_format_opt) {
  const auto device = c10::device_or_default(device_opt);
  const auto dtype = c10::dtype_or_default(dtype_opt);
  TORCH_CHECK(device.is_privateuseone());
  TORCH_CHECK(
      c10::layout_or_default(layout_opt) == c10::Layout::Strided,
      "Non strided layout not supported");
  TORCH_CHECK(
      !c10::pinned_memory_or_default(pin_memory_opt),
      "Pin memory can only be on CPU");
  const c10::DeviceGuard device_guard(device);
  constexpr c10::DispatchKeySet pu1_dks(c10::DispatchKey::PrivateUse1);
  auto allocator = at::GetAllocator(at::kPrivateUse1);
  return at::detail::empty_generic(
      size, allocator, pu1_dks, dtype, memory_format_opt);
}
// LITERALINCLUDE END: EMPTY.MEMORY_FORMAT IMPL

at::Tensor empty_strided(
    c10::IntArrayRef size,
    c10::IntArrayRef stride,
    std::optional<c10::ScalarType> dtype_opt,
    std::optional<c10::Layout> layout_opt,
    std::optional<c10::Device> device_opt,
    std::optional<bool> pin_memory_opt) {
  const auto device = c10::device_or_default(device_opt);
  const auto dtype = c10::dtype_or_default(dtype_opt);
  TORCH_CHECK(device.is_privateuseone());
  TORCH_CHECK(
      c10::layout_or_default(layout_opt) == c10::Layout::Strided,
      "Non strided layout not supported");
  TORCH_CHECK(
      !c10::pinned_memory_or_default(pin_memory_opt),
      "Pin memory can only be on CPU");
  const c10::DeviceGuard device_guard(device);
  constexpr c10::DispatchKeySet pu1_dks(c10::DispatchKey::PrivateUse1);
  auto allocator = at::GetAllocator(at::kPrivateUse1);
  return at::detail::empty_strided_generic(
      size, stride, allocator, pu1_dks, dtype);
}

at::Tensor as_strided(
    const at::Tensor& self,
    c10::SymIntArrayRef size,
    c10::SymIntArrayRef stride,
    std::optional<c10::SymInt> storage_offset) {
  MemoryGuard guard(self);

  return at::cpu::as_strided_symint(self, size, stride, storage_offset);
}

const at::Tensor& resize_(
    const at::Tensor& self,
    c10::SymIntArrayRef size,
    ::std::optional<at::MemoryFormat> memory_format) {
  return at::native::resize_(
      self, C10_AS_INTARRAYREF_SLOW(size), memory_format);
}

at::Tensor _reshape_alias(
    const at::Tensor& self,
    c10::SymIntArrayRef size,
    c10::SymIntArrayRef stride) {
  return at::native::_reshape_alias(
      self, C10_AS_INTARRAYREF_SLOW(size), C10_AS_INTARRAYREF_SLOW(stride));
}

at::Tensor _copy_from(
    const at::Tensor& self,
    const at::Tensor& dst,
    bool non_blocking) {
  TORCH_CHECK(self.defined(), "Source tensor (self) is not defined.");
  TORCH_CHECK(dst.defined(), "Destination tensor (dst) is not defined.");

  MemoryGuard guard(self, dst);

  if (self.device() == dst.device()) {
    at::Tensor dst_as_cpu = at::from_blob(
        dst.data_ptr(),
        dst.sizes(),
        dst.strides(),
        dst.options().device(at::kCPU));
    const at::Tensor self_as_cpu = at::from_blob(
        self.data_ptr(),
        self.sizes(),
        self.strides(),
        self.options().device(at::kCPU));

    at::native::copy_(
        const_cast<at::Tensor&>(dst_as_cpu), self_as_cpu, non_blocking);

  } else {
    if (self.is_cpu()) {
      at::Tensor dst_as_cpu = at::from_blob(
          dst.data_ptr(),
          dst.sizes(),
          dst.strides(),
          dst.options().device(at::kCPU));

      at::native::copy_(
          const_cast<at::Tensor&>(dst_as_cpu), self, non_blocking);

    } else {
      at::Tensor self_as_cpu = at::from_blob(
          self.data_ptr(),
          self.sizes(),
          self.strides(),
          self.options().device(at::kCPU));

      at::native::copy_(
          const_cast<at::Tensor&>(dst), self_as_cpu, non_blocking);
    }
  }

  return dst;
}

at::Tensor _copy_from_and_resize(
    const at::Tensor& self,
    const at::Tensor& dst) {
  at::native::resize_(dst, self.sizes(), std::nullopt);
  return at::native::copy_(const_cast<at::Tensor&>(dst), self, false);
}

at::Scalar _local_scalar_dense(const at::Tensor& self) {
  MemoryGuard guard(self);
  return at::native::_local_scalar_dense_cpu(self);
}

at::Tensor& set_source_Tensor_(at::Tensor& self, const at::Tensor& source) {
  return at::native::set_tensor_(self, source);
}

at::Tensor& set_source_Storage_(at::Tensor& self, at::Storage source) {
  return at::native::set_(self, source);
}

at::Tensor& set_source_Storage_storage_offset_(
    at::Tensor& result,
    at::Storage storage,
    int64_t storage_offset,
    c10::IntArrayRef size,
    c10::IntArrayRef stride) {
  return at::cpu::set_(result, storage, storage_offset, size, stride);
}

at::Tensor view(const at::Tensor& self, c10::SymIntArrayRef size) {
  MemoryGuard guard(self);
  return at::native::view(self, C10_AS_INTARRAYREF_SLOW(size));
}

// LITERALINCLUDE START: FALLBACK IMPL
void cpu_fallback(const c10::OperatorHandle& op, torch::jit::Stack* stack) {
  static const std::unordered_set<c10::OperatorName> cpu_fallback_blocklist = {
      c10::OperatorName("aten::abs", ""),
      c10::OperatorName("aten::abs", "out"),
  };

  const auto& op_name = op.schema().operator_name();
  if (cpu_fallback_blocklist.count(op_name)) {
    TORCH_CHECK(
        false,
        "Operator '",
        op_name,
        "' is not implemented for device openreg.");
  } else {
    at::native::cpu_fallback(op, stack);
  }
}
// LITERALINCLUDE END: FALLBACK IMPL

namespace {
// Helper to fetch a usable OpenReg generator for a device index
inline c10::openreg::OpenRegGeneratorImpl* get_openreg_gen(
    const std::optional<at::Generator>& gen_opt,
    c10::DeviceIndex device_index) {
  if (gen_opt.has_value()) {
    auto* gen = at::check_generator<c10::openreg::OpenRegGeneratorImpl>(
        gen_opt.value());
    TORCH_CHECK(
        gen->device().index() == device_index,
        "Generator device index (",
        (int)gen->device().index(),
        ") does not match target tensor device index (",
        (int)device_index,
        ") for openreg backend");
    return gen;
  }
  const auto& gen = c10::openreg::getDefaultOpenRegGenerator(device_index);
  return at::check_generator<c10::openreg::OpenRegGeneratorImpl>(gen);
}

template <typename T>
inline void fill_uniform(
    T* data,
    int64_t numel,
    c10::openreg::OpenRegGeneratorImpl* gen) {
  at::uniform_real_distribution<T> dist(
      static_cast<T>(0.0), static_cast<T>(1.0));
  for (int64_t i = 0; i < numel; ++i) {
    data[i] = dist(gen);
  }
}

template <typename T>
inline void fill_normal(
    T* data,
    int64_t numel,
    c10::openreg::OpenRegGeneratorImpl* gen) {
  at::normal_distribution<T> dist(static_cast<T>(0.0), static_cast<T>(1.0));
  for (int64_t i = 0; i < numel; ++i) {
    data[i] = dist(gen);
  }
}

template <typename T>
inline void fill_uniform_int(
    T* data,
    int64_t numel,
    T low,
    T high,
    c10::openreg::OpenRegGeneratorImpl* gen) {
  at::uniform_int_from_to_distribution<T> dist(
      static_cast<uint64_t>(high), static_cast<int64_t>(low));
  for (int64_t i = 0; i < numel; ++i) {
    data[i] = dist(gen);
  }
}

} // anonymous namespace
/*
 * Note that there is lots of overloads of rand and here we only implement one
 * of them for code clarity.
 */
at::Tensor rand(
    c10::IntArrayRef size,
    std::optional<at::Generator> generator,
    std::optional<c10::ScalarType> dtype_opt,
    std::optional<c10::Layout> layout_opt,
    std::optional<c10::Device> device_opt,
    std::optional<bool> pin_memory_opt) {
  const auto device = c10::device_or_default(device_opt);
  const auto dtype = c10::dtype_or_default(dtype_opt);
  const auto layout = c10::layout_or_default(layout_opt);
  const auto pin_memory = c10::pinned_memory_or_default(pin_memory_opt);

  auto result = empty_memory_format(
      size, dtype, layout, device, pin_memory, std::nullopt);

  if (result.numel() == 0) {
    return result;
  }

  MemoryGuard guard(result);
  auto* gen = get_openreg_gen(generator, device.index());
  switch (dtype) {
    case at::kFloat: {
      fill_uniform<float>(
          result.mutable_data_ptr<float>(), result.numel(), gen);
      break;
    }
    case at::kDouble: {
      fill_uniform<double>(
          result.mutable_data_ptr<double>(), result.numel(), gen);
      break;
    }
    default: {
      TORCH_CHECK(
          false,
          "Unsupported dtype for rand on openreg: ",
          c10::toString(dtype));
    }
  }
  return result;
}
/*
 * Note that there is lots of overloads of randn and here we only implement one
 * of them for code clarity.
 */
at::Tensor randn(
    c10::IntArrayRef size,
    std::optional<at::Generator> generator,
    std::optional<c10::ScalarType> dtype_opt,
    std::optional<c10::Layout> layout_opt,
    std::optional<c10::Device> device_opt,
    std::optional<bool> pin_memory_opt) {
  const auto device = c10::device_or_default(device_opt);
  const auto dtype = c10::dtype_or_default(dtype_opt);
  const auto layout = c10::layout_or_default(layout_opt);
  const auto pin_memory = c10::pinned_memory_or_default(pin_memory_opt);

  auto result = empty_memory_format(
      size, dtype, layout, device, pin_memory, std::nullopt);

  if (result.numel() == 0) {
    return result;
  }

  MemoryGuard guard(result);
  auto* gen = get_openreg_gen(generator, device.index());
  switch (dtype) {
    case at::kFloat: {
      fill_normal<float>(result.mutable_data_ptr<float>(), result.numel(), gen);
      break;
    }
    case at::kDouble: {
      fill_normal<double>(
          result.mutable_data_ptr<double>(), result.numel(), gen);
      break;
    }
    default: {
      TORCH_CHECK(
          false,
          "Unsupported dtype for randn on openreg: ",
          c10::toString(dtype));
    }
  }

  return result;
}
/*
 * Note that there is lots of overloads of randint and here we only implement
 * one of them for code clarity.
 */
at::Tensor randint(
    int64_t low,
    int64_t high,
    c10::IntArrayRef size,
    std::optional<at::Generator> generator,
    std::optional<c10::ScalarType> dtype_opt,
    std::optional<c10::Layout> layout_opt,
    std::optional<c10::Device> device_opt,
    std::optional<bool> pin_memory_opt) {
  const auto device = c10::device_or_default(device_opt);
  const auto dtype = c10::dtype_or_default(dtype_opt);
  const auto layout = c10::layout_or_default(layout_opt);
  const auto pin_memory = c10::pinned_memory_or_default(pin_memory_opt);

  auto result = empty_memory_format(
      size, dtype, layout, device, pin_memory, std::nullopt);

  if (result.numel() == 0) {
    return result;
  }

  MemoryGuard guard(result);
  auto* gen = get_openreg_gen(generator, device.index());
  switch (dtype) {
    case at::kInt: {
      fill_uniform_int<int32_t>(
          result.mutable_data_ptr<int32_t>(), result.numel(), low, high, gen);
      break;
    }
    case at::kLong: {
      fill_uniform_int<int64_t>(
          result.mutable_data_ptr<int64_t>(), result.numel(), low, high, gen);
      break;
    }
    case at::kShort: {
      fill_uniform_int<int16_t>(
          result.mutable_data_ptr<int16_t>(), result.numel(), low, high, gen);
      break;
    }
    default: {
      TORCH_CHECK(
          false,
          "Unsupported dtype for randint on openreg: ",
          c10::toString(dtype));
    }
  }

  return result;
}

} // namespace at::native::openreg
