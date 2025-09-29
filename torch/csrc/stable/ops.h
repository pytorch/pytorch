#pragma once

#include <torch/csrc/stable/TensorAccessor.h>
#include <torch/csrc/stable/stableivalue_conversions.h>
#include <array>
#include <cstdint>
#include <optional>
#include <string>
#include <vector>

#include <torch/csrc/inductor/aoti_torch/generated/c_shim_aten.h>
#include <torch/headeronly/core/ScalarType.h>

namespace torch::stable {

// We expect this to be the stable version of the empty_like op that takes in
// no kwargs (device, dtype, layout, memory_format). We will add kwargs
// support in the future.
inline torch::stable::Tensor empty_like(const torch::stable::Tensor& self) {
  const auto num_args = 6;
  std::array<StableIValue, num_args> stack{
      from(self),
      from(std::nullopt),
      from(std::nullopt),
      from(std::nullopt),
      from(std::nullopt),
      from(std::nullopt)};
  TORCH_ERROR_CODE_CHECK(
      aoti_torch_call_dispatcher("aten::empty_like", "", stack.data()));
  return to<torch::stable::Tensor>(stack[0]);
}

// We expect this to be the stable version of the fill_.Scalar op
// with identical semantics to the existing fill_.Scalar op.
// A subtle nuance is that `value` is typed as a double, but it is
// actually a Scalar. This is because Scalar.h is currently not
// header-only.
inline torch::stable::Tensor fill_(
    const torch::stable::Tensor& self,
    double value) {
  TORCH_ERROR_CODE_CHECK(aoti_torch_aten_fill__Scalar(self.get(), value));
  return self;
}

// We expect this to be the stable version of the narrow.default op.
// narrow takes in a SymInt for start and length, but these are typed as
// int64_t as SymInt is not yet header-only.
inline torch::stable::Tensor narrow(
    torch::stable::Tensor& self,
    int64_t dim,
    int64_t start,
    int64_t length) {
  AtenTensorHandle ret0 = nullptr;

  TORCH_ERROR_CODE_CHECK(
      aoti_torch_aten_narrow(self.get(), dim, start, length, &ret0));
  return torch::stable::Tensor(ret0);
}

// We expect this to be a stable version of the new_empty op that takes in
// only dtype information.
inline torch::stable::Tensor new_empty(
    const torch::stable::Tensor& self,
    std::vector<int64_t> size,
    std::optional<c10::ScalarType> dtype = std::nullopt,
    std::optional<int32_t> device_type = std::nullopt,
    std::optional<DeviceIndex> device_index = std::nullopt) {
  int32_t target_dtype;
  if (dtype.has_value()) {
    target_dtype = to<int32_t>(from(dtype.value()));
  } else {
    TORCH_ERROR_CODE_CHECK(aoti_torch_get_dtype(self.get(), &target_dtype));
  }

  int32_t self_device_type;
  TORCH_ERROR_CODE_CHECK(
      aoti_torch_get_device_type(self.get(), &self_device_type));

  int32_t device_type_;
  if (device_type.has_value()) {
    device_type_ = to<int32_t>(from(device_type.value()));
  } else {
    device_type_ = self_device_type;
  }

  int32_t device_index_;
  if (device_index.has_value()) {
    device_index_ = to<int32_t>(from(device_index.value()));
  } else {
    if (device_type_ == self_device_type) {
      TORCH_ERROR_CODE_CHECK(
          aoti_torch_get_device_index(self.get(), &device_index_));
    } else {
      // It is unmeaningul to use self device index when self device
      // type is different from target device type.
      device_index_ = 0;
    }
  }

  int32_t layout;
  TORCH_ERROR_CODE_CHECK(aoti_torch_get_layout(self.get(), &layout));

  AtenTensorHandle ret0;
  TORCH_ERROR_CODE_CHECK(aoti_torch_aten_new_empty(
      self.get(),
      size.data(),
      static_cast<int64_t>(size.size()),
      &target_dtype,
      &layout,
      &device_type_,
      device_index_,
      nullptr, // pin_memory (nullptr for default)
      &ret0));

  return torch::stable::Tensor(ret0);
}

// We expect this to be a stable version of the new_zeros op that takes in
// only dtype information.
inline torch::stable::Tensor new_zeros(
    const torch::stable::Tensor& self,
    std::vector<int64_t> size,
    std::optional<c10::ScalarType> dtype = std::nullopt) {
  int32_t device_type;
  TORCH_ERROR_CODE_CHECK(aoti_torch_get_device_type(self.get(), &device_type));

  int32_t device_index;
  TORCH_ERROR_CODE_CHECK(
      aoti_torch_get_device_index(self.get(), &device_index));

  int32_t target_dtype;
  if (dtype.has_value()) {
    target_dtype = to<int32_t>(from(dtype.value()));
  } else {
    TORCH_ERROR_CODE_CHECK(aoti_torch_get_dtype(self.get(), &target_dtype));
  }

  int32_t layout;
  TORCH_ERROR_CODE_CHECK(aoti_torch_get_layout(self.get(), &layout));

  AtenTensorHandle ath;
  TORCH_ERROR_CODE_CHECK(aoti_torch_aten_new_zeros(
      self.get(),
      size.data(),
      static_cast<int64_t>(size.size()),
      &target_dtype,
      &layout,
      &device_type,
      device_index,
      nullptr, // pin_memory (nullptr for default)
      &ath));

  return torch::stable::Tensor(ath);
}

// We expect this to be the stable version of the pad.default op.
// pad.default takes in a SymInt[] as the pad argument however pad is typed as
// use std::vector<int64_t> because
// (1) IntArrayRef is not yet header-only
// (2) SymInt is not yet header-only
inline torch::stable::Tensor pad(
    const torch::stable::Tensor& self,
    std::vector<int64_t> pad,
    const std::string& mode = "constant",
    double value = 0.0) {
  AtenTensorHandle ret0 = nullptr;

  TORCH_ERROR_CODE_CHECK(aoti_torch_aten_pad(
      self.get(), pad.data(), pad.size(), mode.c_str(), &value, &ret0));
  return torch::stable::Tensor(ret0);
}

// We expect the following two functions to be stable versions of the
// amax.default op with identical semantics to the existing amax.default op. If
// `keepdim` is true, the result will have the same number of dimensions as
// `self`, with the specified dimension having size 1. Otherwise, the result
// will have one fewer dimension than `self`, with the specified dimension
// removed.

// This function is an overload to compute the maximum value along each slice of
// `self` along a single dimension `dim`.
inline torch::stable::Tensor amax(
    const torch::stable::Tensor& self,
    int64_t dim,
    bool keepdim = false) {
  AtenTensorHandle ret = nullptr;
  TORCH_ERROR_CODE_CHECK(
      aoti_torch_aten_amax(self.get(), &dim, 1, keepdim, &ret));
  return torch::stable::Tensor(ret);
}

// This function is an overload to compute the maximum value along each slice of
// `self` reducing over all the dimensions in the vector `dims`. The
// amax.default op takes in a SymInt[] as the dims argument, however dims is
// typed as use std::vector<int64_t> here because (1) IntArrayRef is not yet
// header-only (2) SymInt is not yet header-only
inline torch::stable::Tensor amax(
    const torch::stable::Tensor& self,
    std::vector<int64_t> dims,
    bool keepdim = false) {
  AtenTensorHandle ret = nullptr;
  TORCH_ERROR_CODE_CHECK(aoti_torch_aten_amax(
      self.get(),
      dims.data(),
      static_cast<int64_t>(dims.size()),
      keepdim,
      &ret));
  return torch::stable::Tensor(ret);
}

// We expect this to be the stable version of the transpose op with identical
// semantics to the existing transpose.int op.
inline torch::stable::Tensor transpose(
    const torch::stable::Tensor& self,
    int64_t dim0,
    int64_t dim1) {
  const auto num_args = 3;
  std::array<StableIValue, num_args> stack{from(self), from(dim0), from(dim1)};
  TORCH_ERROR_CODE_CHECK(
      aoti_torch_call_dispatcher("aten::transpose", "int", stack.data()));
  return to<torch::stable::Tensor>(stack[0]);
}

// We expect this to be the stable version of the zero_ op with identical
// semantics to the existing zero_ op (except that it will not be called as
// a tensor method but only as a function i.e. zero_(t) not t.zero_()).
inline torch::stable::Tensor zero_(torch::stable::Tensor& self) {
  const auto num_args = 1;
  std::array<StableIValue, num_args> stack{from(self)};
  TORCH_ERROR_CODE_CHECK(
      aoti_torch_call_dispatcher("aten::zero_", "", stack.data()));
  return to<torch::stable::Tensor>(stack[0]);
}

// We expect this to be the stable version of the copy_ op with
// identical semantics to the existing copy_ op.
inline torch::stable::Tensor copy_(
    torch::stable::Tensor& self,
    const torch::stable::Tensor& src,
    std::optional<bool> non_blocking = std::nullopt) {
  const auto num_args = 3;
  std::array<StableIValue, num_args> stack{
      from(self), from(src), from(non_blocking.value_or(false))};
  TORCH_ERROR_CODE_CHECK(
      aoti_torch_call_dispatcher("aten::copy_", "", stack.data()));
  return to<torch::stable::Tensor>(stack[0]);
}

// We expect this to be the stable version of the clone op. We will
// add optional memory_format kwarg support in the future.
inline torch::stable::Tensor clone(const torch::stable::Tensor& self) {
  const auto num_args = 2;
  std::array<StableIValue, num_args> stack{from(self), from(std::nullopt)};
  TORCH_ERROR_CODE_CHECK(
      aoti_torch_call_dispatcher("aten::clone", "", stack.data()));
  return to<torch::stable::Tensor>(stack[0]);
}

inline std::vector<int64_t> sizes(const Tensor& t) {
  int64_t* ptr;
  TORCH_ERROR_CODE_CHECK(aoti_torch_get_sizes(t.get(), &ptr));
  std::vector<int64_t> r(ptr, ptr + t.dim());
  return r;
}

inline std::vector<int64_t> strides(const Tensor& t) {
  int64_t* ptr;
  TORCH_ERROR_CODE_CHECK(aoti_torch_get_strides(t.get(), &ptr));
  std::vector<int64_t> r(ptr, ptr + t.dim());
  return r;
}

template <typename T, size_t N>
torch::stable::TensorAccessor<T, N> accessor(const Tensor& t) {
  static_assert(
      N > 0,
      "accessor is used for indexing tensor, for scalars use *data_ptr<T>()");
  STD_TORCH_CHECK(
      t.dim() == N,
      "TensorAccessor expected ",
      N,
      " dims but tensor has ",
      t.dim());
  T* ptr = nullptr;
  if constexpr (std::is_const_v<T>) {
    ptr = t.const_data_ptr<T>();
  } else {
    ptr = t.mutable_data_ptr<T>();
  }
  auto sizes_ = sizes(t);
  auto strides_ = strides(t);
  return torch::stable::TensorAccessor<T, N>(
      ptr, sizes_.data(), strides_.data());
}

template <
    typename T,
    size_t N,
    template <typename U> class PtrTraits = torch::stable::DefaultPtrTraits,
    typename index_t = int64_t>
torch::stable::GenericPackedTensorAccessor<T, N, PtrTraits, index_t>
generic_packed_accessor(const Tensor& t) {
  static_assert(
      N > 0,
      "accessor is used for indexing tensor, for scalars use *data_ptr<T>()");
  STD_TORCH_CHECK(
      t.dim() == N,
      "TensorAccessor expected ",
      N,
      " dims but tensor has ",
      t.dim());
  T* ptr = nullptr;
  if constexpr (std::is_const_v<T>) {
    ptr = t.const_data_ptr<T>();
  } else {
    ptr = t.mutable_data_ptr<T>();
  }
  auto sizes_ = sizes(t);
  auto strides_ = strides(t);
  return torch::stable::GenericPackedTensorAccessor<T, N, PtrTraits, index_t>(
      static_cast<typename PtrTraits<T>::PtrType>(ptr),
      sizes_.data(),
      strides_.data());
}

template <
    typename T,
    size_t N,
    template <typename U> class PtrTraits = torch::stable::DefaultPtrTraits>
torch::stable::PackedTensorAccessor32<T, N, PtrTraits> packed_accessor32(
    const Tensor& t) {
  STD_TORCH_CHECK(
      t.numel() <= static_cast<int64_t>(std::numeric_limits<int32_t>::max()),
      "numel needs to be smaller than int32_t max; otherwise, please use packed_accessor64");
  return generic_packed_accessor<T, N, PtrTraits, int32_t>(t);
}

template <
    typename T,
    size_t N,
    template <typename U> class PtrTraits = torch::stable::DefaultPtrTraits>
torch::stable::PackedTensorAccessor64<T, N, PtrTraits> packed_accessor64(
    const Tensor& t) {
  return generic_packed_accessor<T, N, PtrTraits, int64_t>();
}

} // namespace torch::stable
