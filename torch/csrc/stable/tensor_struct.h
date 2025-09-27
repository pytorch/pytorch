#pragma once

#include <ATen/core/TensorAccessor.h>
#include <torch/csrc/inductor/aoti_runtime/mini_array_ref.h>
#include <torch/csrc/inductor/aoti_torch/c/shim.h>
#include <torch/headeronly/core/ScalarType.h>
#include <torch/headeronly/util/Exception.h>
#include <torch/headeronly/util/shim_utils.h>
#include <climits>
#include <memory>

#include <torch/csrc/stable/accelerator.h>

namespace torch::stable {

using accelerator::DeviceIndex;
using torch::headeronly::ScalarType;

// The torch::stable::Tensor class is a highlevel C++ wrapper around
// the C shim Tensor APIs. We've modeled this class after TensorBase, as custom
// op kernels only really need to interact with Tensor metadata (think sizes,
// strides, device, dtype). Other functions on Tensor (like empty_like) should
// live like the ATen op that they are and exist outside of this struct.
//
// There are several goals of this class over AtenTensorHandle and
// RAIIAtenTensorHandle:
// 1. torch::stable::Tensor is a nicer UX much closer to torch::Tensor than the
//    C APIs with AtenTensorHandle. Under the hood we still call to these C shim
//    APIs to preserve stability.
// 2. RAIIAtenTensorHandle boils down to a uniq_ptr that forces the user to pass
//    around ownership. This makes it difficult to pass one input into 2
//    different functions, e.g., doing something like c = a(t) + b(t) for
//    stable::Tensor t. Thus, we use a shared_ptr here.
class Tensor {
 private:
  std::shared_ptr<AtenTensorOpaque> ath_;

 public:
  // Construct a stable::Tensor with an uninitialized AtenTensorHandle (ATH)
  // Steals ownership from the ATH
  Tensor() {
    AtenTensorHandle ret;
    TORCH_ERROR_CODE_CHECK(aoti_torch_new_uninitialized_tensor(&ret));
    ath_ = std::shared_ptr<AtenTensorOpaque>(ret, [](AtenTensorHandle ath) {
      TORCH_ERROR_CODE_CHECK(aoti_torch_delete_tensor_object(ath));
    });
  }

  // Construct a stable::Tensor from an AtenTensorHandle (ATH)
  // Steals ownership from the ATH
  explicit Tensor(AtenTensorHandle ath)
      : ath_(ath, [](AtenTensorHandle ath) {
          TORCH_ERROR_CODE_CHECK(aoti_torch_delete_tensor_object(ath));
        }) {}

  // Copy and move constructors can be default cuz the underlying handle is a
  // shared_ptr
  Tensor(const Tensor& other) = default;
  Tensor(Tensor&& other) noexcept = default;

  // Copy and move assignment operators can be default cuz the underlying handle
  // is a shared_ptr
  Tensor& operator=(const Tensor& other) = default;
  Tensor& operator=(Tensor&& other) noexcept = default;

  // Destructor can be default: shared ptr has custom deletion logic
  ~Tensor() = default;

  // Returns a borrowed reference to the AtenTensorHandle
  AtenTensorHandle get() const {
    return ath_.get();
  }

  // =============================================================================
  // C-shimified TensorBase APIs: the below APIs have the same signatures and
  // semantics as their counterparts in TensorBase.h.
  // =============================================================================

  // Do not add new uses of data_ptr(), use const_data_ptr() if
  // possible, mutable_data_ptr() otherwise.
  void* data_ptr() const {
    void* data_ptr;
    TORCH_ERROR_CODE_CHECK(aoti_torch_get_data_ptr(ath_.get(), &data_ptr));
    return data_ptr;
  }

  void* mutable_data_ptr() const {
    void* data_ptr{};
    TORCH_ERROR_CODE_CHECK(
        aoti_torch_get_mutable_data_ptr(ath_.get(), &data_ptr));
    return data_ptr;
  }

  const void* const_data_ptr() const {
    const void* data_ptr{};
    TORCH_ERROR_CODE_CHECK(
        aoti_torch_get_const_data_ptr(ath_.get(), &data_ptr));
    return data_ptr;
  }

  template <typename T>
  T* mutable_data_ptr() const;

  template <typename T>
  const T* const_data_ptr() const;

  int64_t dim() const {
    int64_t dim;
    TORCH_ERROR_CODE_CHECK(aoti_torch_get_dim(ath_.get(), &dim));
    return dim;
  }

  int64_t numel() const {
    int64_t numel;
    TORCH_ERROR_CODE_CHECK(aoti_torch_get_numel(ath_.get(), &numel));
    return numel;
  }

  // note: this is a subset of the original TensorBase API. It takes no
  // arguments whereas the original API takes in a kwarg of memory format.
  // Here, we assume the default contiguous memory format.
  bool is_contiguous() const {
    bool is_contiguous;
    TORCH_ERROR_CODE_CHECK(
        aoti_torch_is_contiguous(ath_.get(), &is_contiguous));
    return is_contiguous;
  }

  int64_t stride(int64_t dim) const {
    int64_t stride;
    TORCH_ERROR_CODE_CHECK(aoti_torch_get_stride(ath_.get(), dim, &stride));
    return stride;
  }

  // This is almost the same API as the one in TensorBase.h, except
  // we add a check that the returned device_index is within the
  // range of int8_t.
  int8_t get_device() const {
    int32_t device_index;
    TORCH_ERROR_CODE_CHECK(
        aoti_torch_get_device_index(ath_.get(), &device_index));
    STD_TORCH_CHECK(
        device_index >= std::numeric_limits<int8_t>::min() &&
            device_index <= std::numeric_limits<int8_t>::max(),
        "Device index is out of range of return type int8_t, please use get_device_index() instead.");
    return static_cast<int8_t>(device_index);
  }

  // The same as get_device but with two differences:
  // 1. it has a more suiting name
  // 2. it returns a DeviceIndex, which is int32_t in this world
  //    that should be more stable than the likely shifting
  //    DeviceIndex in libtorch (it is int8_t that might become int16_t)
  DeviceIndex get_device_index() const {
    int32_t device_index;
    TORCH_ERROR_CODE_CHECK(
        aoti_torch_get_device_index(ath_.get(), &device_index));
    return device_index;
  }

  bool is_cuda() const {
    int32_t device_type;
    TORCH_ERROR_CODE_CHECK(
        aoti_torch_get_device_type(ath_.get(), &device_type));
    return device_type == aoti_torch_device_type_cuda();
  }

  bool is_cpu() const {
    int32_t device_type;
    TORCH_ERROR_CODE_CHECK(
        aoti_torch_get_device_type(ath_.get(), &device_type));
    return device_type == aoti_torch_device_type_cpu();
  }

  int64_t size(int64_t dim) const {
    int64_t size;
    TORCH_ERROR_CODE_CHECK(aoti_torch_get_size(ath_.get(), dim, &size));
    return size;
  }

  auto sizes() const {
    int64_t* ptr;
    TORCH_ERROR_CODE_CHECK(aoti_torch_get_sizes(ath_.get(), &ptr));
    return torch::aot_inductor::MiniArrayRef<int64_t>(ptr, dim());
  }

  auto strides() const {
    int64_t* ptr;
    TORCH_ERROR_CODE_CHECK(aoti_torch_get_strides(ath_.get(), &ptr));
    return torch::aot_inductor::MiniArrayRef<int64_t>(ptr, dim());
  }

  template <typename T, size_t N>
  at::TensorAccessor<T, N> accessor() const& {
    static_assert(
        N > 0,
        "accessor is used for indexing tensor, for scalars use *data_ptr<T>()");
    STD_TORCH_CHECK(
        dim() == N,
        "TensorAccessor expected ",
        N,
        " dims but tensor has ",
        dim());
    T* ptr = nullptr;
    if constexpr (std::is_const_v<T>) {
      ptr = const_data_ptr<T>();
    } else {
      ptr = mutable_data_ptr<T>();
    }
    return at::TensorAccessor<T, N>(ptr, sizes().data(), strides().data());
  }

  template <
      typename T,
      size_t N,
      template <typename U> class PtrTraits = at::DefaultPtrTraits,
      typename index_t = int64_t>
  at::GenericPackedTensorAccessor<T, N, PtrTraits, index_t>
  generic_packed_accessor() const& {
    static_assert(
        N > 0,
        "accessor is used for indexing tensor, for scalars use *data_ptr<T>()");
    TORCH_CHECK(
        dim() == N,
        "TensorAccessor expected ",
        N,
        " dims but tensor has ",
        dim());
    T* ptr = nullptr;
    if constexpr (std::is_const_v<T>) {
      ptr = const_data_ptr<T>();
    } else {
      ptr = mutable_data_ptr<T>();
    }
    return at::GenericPackedTensorAccessor<T, N, PtrTraits, index_t>(
        static_cast<typename PtrTraits<T>::PtrType>(ptr),
        sizes().data(),
        strides().data());
  }

  template <
      typename T,
      size_t N,
      template <typename U> class PtrTraits = at::DefaultPtrTraits>
  at::PackedTensorAccessor32<T, N, PtrTraits> packed_accessor32() const& {
    STD_TORCH_CHECK(
        numel() <= static_cast<int64_t>(std::numeric_limits<int32_t>::max()),
        "numel needs to be smaller than int32_t max; otherwise, please use packed_accessor64");
    return generic_packed_accessor<T, N, PtrTraits, int32_t>();
  }

  bool defined() const {
    bool defined;
    TORCH_ERROR_CODE_CHECK(aoti_torch_is_defined(ath_.get(), &defined));
    return defined;
  }

  // defined in tensor-inl.h to avoid circular dependencies
  ScalarType scalar_type() const;

  // =============================================================================
  // END of C-shimified TensorBase APIs
  // =============================================================================
};

} // namespace torch::stable
