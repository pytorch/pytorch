#pragma once

#include <torch/csrc/inductor/aoti_torch/c/shim.h>
#include <torch/headeronly/core/ScalarType.h>
#include <torch/headeronly/macros/Macros.h>
#include <torch/headeronly/util/Exception.h>
#include <torch/headeronly/util/HeaderOnlyArrayRef.h>
#include <torch/headeronly/util/shim_utils.h>
#include <climits>
#include <memory>

#include <torch/csrc/stable/accelerator.h>
#include <torch/csrc/stable/device_struct.h>

HIDDEN_NAMESPACE_BEGIN(torch, stable)

using accelerator::DeviceIndex;
using torch::headeronly::IntHeaderOnlyArrayRef;
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

/**
 * @brief An ABI stable wrapper around PyTorch tensors.
 *
 * This class is modeled after TensorBase, as custom
 * op kernels primarily need to interact with Tensor metadata (sizes,
 * strides, device, dtype). Other tensor operations (like ``empty_like``) exist
 * as standalone functions outside of this struct.
 *
 * Minimum compatible version: PyTorch 2.9.
 */
class Tensor {
 private:
  std::shared_ptr<AtenTensorOpaque> ath_;

 public:
  /**
   * @brief Constructs a Tensor with an uninitialized AtenTensorHandle.
   *
   * Creates a new stable::Tensor by allocating an uninitialized tensor handle.
   * The ownership of the handle is managed internally via shared_ptr.
   *
   * Minimum compatible version: PyTorch 2.9.
   */
  Tensor() {
    AtenTensorHandle ret;
    TORCH_ERROR_CODE_CHECK(aoti_torch_new_uninitialized_tensor(&ret));
    ath_ = std::shared_ptr<AtenTensorOpaque>(ret, [](AtenTensorHandle ath) {
      TORCH_ERROR_CODE_CHECK(aoti_torch_delete_tensor_object(ath));
    });
  }

  /**
   * @brief Constructs a Tensor from an existing AtenTensorHandle.
   *
   * Steals ownership of the provided AtenTensorHandle.
   *
   * @param ath The AtenTensorHandle to wrap. Ownership is transferred to this
   *            Tensor.
   *
   * Minimum compatible version: PyTorch 2.9.
   */
  explicit Tensor(AtenTensorHandle ath)
      : ath_(ath, [](AtenTensorHandle ath) {
          TORCH_ERROR_CODE_CHECK(aoti_torch_delete_tensor_object(ath));
        }) {}

  // Copy and move constructors can be default cuz the underlying handle is a
  // shared_ptr
  /// \private
  Tensor(const Tensor& other) = default;
  /// \private
  Tensor(Tensor&& other) noexcept = default;

  // Copy and move assignment operators can be default cuz the underlying handle
  // is a shared_ptr
  /// \private
  Tensor& operator=(const Tensor& other) = default;
  /// \private
  Tensor& operator=(Tensor&& other) noexcept = default;

  // Destructor can be default: shared ptr has custom deletion logic
  /// \private
  ~Tensor() = default;

  /**
   * @brief Returns a borrowed reference to the underlying AtenTensorHandle.
   *
   * @return The underlying AtenTensorHandle.
   *
   * Minimum compatible version: PyTorch 2.9.
   */
  AtenTensorHandle get() const {
    return ath_.get();
  }

  // =============================================================================
  // C-shimified TensorBase APIs: the below APIs have the same signatures and
  // semantics as their counterparts in TensorBase.h.
  // =============================================================================

  /**
   * @brief Returns a pointer to the tensor's data.
   *
   * @return A void pointer to the tensor's data storage.
   *
   * Minimum compatible version: PyTorch 2.9.
   */
  void* data_ptr() const {
    void* data_ptr;
    TORCH_ERROR_CODE_CHECK(aoti_torch_get_data_ptr(ath_.get(), &data_ptr));
    return data_ptr;
  }

#if TORCH_FEATURE_VERSION >= TORCH_VERSION_2_10_0
  /**
   * @brief Returns a mutable pointer to the tensor's data.
   *
   * @return A mutable void pointer to the tensor's data storage.
   *
   * Minimum compatible version: PyTorch 2.10.
   */
  void* mutable_data_ptr() const {
    void* data_ptr{};
    TORCH_ERROR_CODE_CHECK(torch_get_mutable_data_ptr(ath_.get(), &data_ptr));
    return data_ptr;
  }

  /**
   * @brief Returns a const pointer to the tensor's data.
   *
   * @return A const void pointer to the tensor's data storage.
   *
   * Minimum compatible version: PyTorch 2.10.
   */
  const void* const_data_ptr() const {
    const void* data_ptr{};
    TORCH_ERROR_CODE_CHECK(torch_get_const_data_ptr(ath_.get(), &data_ptr));
    return data_ptr;
  }

  /**
   * @brief Returns a typed mutable pointer to the tensor's data.
   *
   * @tparam T The type to cast the data pointer to.
   * @return A mutable pointer to the tensor's data cast to type T*.
   *
   * Minimum compatible version: PyTorch 2.10.
   */
  template <typename T>
  T* mutable_data_ptr() const;

  /**
   * @brief Returns a typed const pointer to the tensor's data.
   *
   * @tparam T The type to cast the data pointer to. Must not be
   * const-qualified.
   * @return A const pointer to the tensor's data cast to type const T*.
   *
   * Minimum compatible version: PyTorch 2.10.
   */
  template <typename T, std::enable_if_t<!std::is_const_v<T>, int> = 0>
  const T* const_data_ptr() const;

  /**
   * @brief Sets whether this tensor requires gradient computation.
   *
   * @param requires_grad If true, gradients will be computed for this tensor
   *                      during backpropagation.
   * @return A reference to this Tensor.
   *
   * Minimum compatible version: PyTorch 2.10.
   */
  const Tensor& set_requires_grad(bool requires_grad) const {
    TORCH_ERROR_CODE_CHECK(torch_set_requires_grad(ath_.get(), requires_grad));
    return *this;
  }
#endif // TORCH_FEATURE_VERSION >= TORCH_VERSION_2_10_0

  /**
   * @brief Returns the number of dimensions of the tensor.
   *
   * @return The number of dimensions (rank) of the tensor.
   *
   * Minimum compatible version: PyTorch 2.9.
   */
  int64_t dim() const {
    int64_t dim;
    TORCH_ERROR_CODE_CHECK(aoti_torch_get_dim(ath_.get(), &dim));
    return dim;
  }

  /**
   * @brief Returns the total number of elements in the tensor.
   *
   * @return The total number of elements across all dimensions.
   *
   * Minimum compatible version: PyTorch 2.9.
   */
  int64_t numel() const {
    int64_t numel;
    TORCH_ERROR_CODE_CHECK(aoti_torch_get_numel(ath_.get(), &numel));
    return numel;
  }

  // note: sizes and strides, for all intents and purposes, the same as in
  // TensorBase.h: it returns a borrowed reference of the dimension sizes of
  // a Tensor.
  //
  // The only difference is that it returns a header-only IntHeaderOnlyArrayRef,
  // which has slightly less functionality than a regular IntArrayRef. See
  // [HeaderOnlyArrayRef vs ArrayRef note] for more details.
  /**
   * @brief Returns the sizes (shape) of the tensor.
   *
   * Returns a borrowed reference of the dimension sizes of the tensor.
   *
   * @return An IntHeaderOnlyArrayRef containing the size of each dimension.
   *
   * Minimum compatible version: PyTorch 2.9.
   */
  IntHeaderOnlyArrayRef sizes() const {
    int64_t* sizes;
    TORCH_ERROR_CODE_CHECK(aoti_torch_get_sizes(ath_.get(), &sizes));
    return IntHeaderOnlyArrayRef(sizes, dim());
  }

  /**
   * @brief Returns the strides of the tensor.
   *
   * Returns a borrowed reference of the strides of the tensor.
   *
   *
   * @return An IntHeaderOnlyArrayRef containing the stride of each dimension.
   *
   * Minimum compatible version: PyTorch 2.9.
   */
  IntHeaderOnlyArrayRef strides() const {
    int64_t* strides;
    TORCH_ERROR_CODE_CHECK(aoti_torch_get_strides(ath_.get(), &strides));
    return IntHeaderOnlyArrayRef(strides, dim());
  }

  /**
   * @brief Checks if the tensor is contiguous in memory.
   *
   * @note This is a subset of the original TensorBase API. It takes no
   *       arguments whereas the original API takes a memory format argument.
   *       Here, we assume the default contiguous memory format.
   *
   * @return true if the tensor is contiguous, false otherwise.
   *
   * Minimum compatible version: PyTorch 2.9.
   */
  bool is_contiguous() const {
    bool is_contiguous;
    TORCH_ERROR_CODE_CHECK(
        aoti_torch_is_contiguous(ath_.get(), &is_contiguous));
    return is_contiguous;
  }

  /**
   * @brief Returns the stride of a specific dimension.
   *
   * @param dim The dimension index to query.
   * @return The stride of the specified dimension.
   *
   * Minimum compatible version: PyTorch 2.9.
   */
  int64_t stride(int64_t dim) const {
    int64_t stride;
    TORCH_ERROR_CODE_CHECK(aoti_torch_get_stride(ath_.get(), dim, &stride));
    return stride;
  }

  // This is almost the same API as the one in TensorBase.h, except
  // we add a check that the returned device_index is within the
  // range of int8_t.
  /// \private
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
  /**
   * @brief Returns the device index of the tensor.
   *
   * @return The device index as DeviceIndex (int32_t).
   *
   * Minimum compatible version: PyTorch 2.9.
   */
  DeviceIndex get_device_index() const {
    int32_t device_index;
    TORCH_ERROR_CODE_CHECK(
        aoti_torch_get_device_index(ath_.get(), &device_index));
    return device_index;
  }

  /**
   * @brief Checks if the tensor is on a CUDA device.
   *
   * @return true if the tensor is on a CUDA device, false otherwise.
   *
   * Minimum compatible version: PyTorch 2.9.
   */
  bool is_cuda() const {
    int32_t device_type;
    TORCH_ERROR_CODE_CHECK(
        aoti_torch_get_device_type(ath_.get(), &device_type));
    return device_type == aoti_torch_device_type_cuda();
  }

  /**
   * @brief Checks if the tensor is on the CPU.
   *
   * @return true if the tensor is on the CPU, false otherwise.
   *
   * Minimum compatible version: PyTorch 2.9.
   */
  bool is_cpu() const {
    int32_t device_type;
    TORCH_ERROR_CODE_CHECK(
        aoti_torch_get_device_type(ath_.get(), &device_type));
    return device_type == aoti_torch_device_type_cpu();
  }

  /**
   * @brief Returns the size of a specific dimension.
   *
   * @param dim The dimension index to query.
   * @return The size of the specified dimension.
   *
   * Minimum compatible version: PyTorch 2.9.
   */
  int64_t size(int64_t dim) const {
    int64_t size;
    TORCH_ERROR_CODE_CHECK(aoti_torch_get_size(ath_.get(), dim, &size));
    return size;
  }

  /**
   * @brief Checks if the tensor is defined (not null).
   *
   * @return true if the tensor is defined, false otherwise.
   *
   * Minimum compatible version: PyTorch 2.9.
   */
  bool defined() const {
    bool defined;
    TORCH_ERROR_CODE_CHECK(aoti_torch_is_defined(ath_.get(), &defined));
    return defined;
  }

  /**
   * @brief Returns the storage offset of the tensor.
   *
   * The storage offset is the number of elements from the beginning of the
   * underlying storage to the first element of the tensor.
   *
   * @return The storage offset in number of elements.
   *
   * Minimum compatible version: PyTorch 2.9.
   */
  int64_t storage_offset() const {
    int64_t storage_offset;
    TORCH_ERROR_CODE_CHECK(
        aoti_torch_get_storage_offset(ath_.get(), &storage_offset));
    return storage_offset;
  }

  /**
   * @brief Returns the size in bytes of each element in the tensor.
   *
   * @return The element size in bytes.
   *
   * Minimum compatible version: PyTorch 2.9.
   */
  size_t element_size() const {
    int32_t dtype;
    TORCH_ERROR_CODE_CHECK(aoti_torch_get_dtype(ath_.get(), &dtype));
    return aoti_torch_dtype_element_size(dtype);
  }

  // defined in tensor-inl.h to avoid circular dependencies
  /**
   * @brief Returns the scalar type (dtype) of the tensor.
   *
   * @return The ScalarType of the tensor.
   *
   * Minimum compatible version: PyTorch 2.9.
   */
  ScalarType scalar_type() const;

  // defined in tensor-inl.h to avoid circular dependencies
  /**
   * @brief Returns the device of the tensor.
   *
   * @return The Device on which the tensor resides.
   *
   * Minimum compatible version: PyTorch 2.9.
   */
  Device device() const;

  // =============================================================================
  // END of C-shimified TensorBase APIs
  // =============================================================================
};

HIDDEN_NAMESPACE_END(torch, stable)
