#pragma once

#include <ATen/core/DimVector.h>
#include <ATen/core/TensorImpl.h>
#include <ATen/core/context_base.h>
#include <ATen/core/context_base.h>

#include "caffe2/core/allocator.h"
#include "caffe2/core/common.h"
#include "caffe2/core/flags.h"
#include "caffe2/core/logging.h"

// A global boolean variable to control whether we free memory when a Tensor
// is shrinked to a smaller size. As a result, a Tensor is always going to
// keep the memory allocated for its maximum capacity reshaped to so far.
CAFFE2_DECLARE_bool(caffe2_keep_on_shrink);

// Since we can have high variance in blob memory allocated across different
// inputs in the same run, we will shrink the blob only if the memory gain
// is larger than this flag in bytes.
CAFFE2_DECLARE_int64(caffe2_max_keep_on_shrink_memory);

namespace caffe2 {

// Defined by protobuf
class DeviceOption;

/**
 * A utility function to convert vector<int> to vector<TIndex>.
 */
inline std::vector<TIndex> ToVectorTIndex(const std::vector<int>& src) {
  return std::vector<TIndex>(src.begin(), src.end());
}

/**
 * Return product of all dimensions starting from k
 */
inline TIndex size_from_dim_(int k, const std::vector<TIndex>& dims) {
  TIndex r = 1;
  for (size_t i = k; i < dims.size(); ++i) {
    r *= dims[i];
  }
  return r;
}

// Product of all dims up to k (not including dims[k])
inline TIndex size_to_dim_(int k, const std::vector<TIndex>& dims) {
  CAFFE_ENFORCE((unsigned)k <= dims.size());
  TIndex r = 1;
  for (int i = 0; i < k; ++i) {
    r *= dims[i];
  }
  return r;
}

// Product of all dims between k and l (not including dims[k] and dims[l])
inline TIndex size_between_dim_(int k, int l, const std::vector<TIndex>& dims) {
  CAFFE_ENFORCE((unsigned)l < dims.size());
  TIndex r = 1;
  if (k < l) {
    for (int i = k + 1; i < l; ++i) {
      r *= dims[i];
    }
  } else {
    for (int i = l + 1; i < k; ++i) {
      r *= dims[i];
    }
  }
  return r;
}

// Wrap around axis_index if it is negative, s.t., -1 is the last dim
inline int canonical_axis_index_(int axis_index, int ndims) {
  CAFFE_ENFORCE_GE(axis_index, -ndims);
  CAFFE_ENFORCE_LT(axis_index, ndims);
  if (axis_index < 0) {
    return axis_index + ndims;
  }
  return axis_index;
}

/**
 * @brief TensorImpl is the implementation of a tensor and the basic class
 * in Caffe2 that stores a contiguous memory with its shape information.
 *
 * The TensorImpl class is essentially a wrapper around a device-specific memory
 * (the device is specified by the Context template argument), and deals with
 * the allocation and de-allocation of such memory. We make a simplified
 * assumption that the memory is always contiguous.
 */
class CAFFE2_API TensorImpl : public c10::intrusive_ptr_target {
 public:
  TensorImpl() = delete;

  explicit TensorImpl(at::Storage storage) : storage_(std::move(storage)), storage_offset_(0) {
    data_type_ = storage_ ? storage_.dtype() : TypeMeta{};
  }

  TensorImpl(const TensorImpl&) = default;
  TensorImpl& operator=(const TensorImpl&) = default;
  TensorImpl(TensorImpl&&) = default;
  TensorImpl& operator=(TensorImpl&&) = default;

  virtual ~TensorImpl() noexcept {}

  /*
   * Since we removed template from tensor, we now store a static
   * context pointer in tensor, which indicates the type of the tensor.
   */
  at::BaseStaticContext* GetStaticContext() const {
    auto device_type = GetDeviceType();
    return get_static_context(device_type);
  }

  /* @brief
   * Create a context that has the same device_type
   * as the tensor.
   * Note that this doesn't support passing in argument
   * TODO(jerryzh): move this to a global registry
   * that can create context for us
   */
  std::unique_ptr<at::BaseContext> CreateContext() const {
    return GetStaticContext()->CreateContext();
  }

  at::DeviceType GetDeviceType() const {
    return storage_.device_type();
  }

  /**
   * @brief Copies the data from a source tensor, with a contex provided to
   * carry out the underlying memcpy operation.
   */
  void CopyFrom(const TensorImpl& src, at::BaseContext* context = nullptr) {
    if ((void*)&src == (void*)this) {
      return;
    }
    if (data_type_ != src.meta()) {
      CAFFE_ENFORCE_WITH_CALLER(
          src.is_contiguous(),
          "Right now only copy of contiguous source Tensor is supported.");
      storage_ = at::Storage(GetDeviceType(), src.meta());
      data_type_ = src.meta();
    }
    if (src.size() == -1) {
      dims_.clear();
      numel_ = -1;
      strides_.clear();
      is_contiguous_ = true;
      storage_.reset();
      data_type_ = TypeMeta();
      return;
    }
    Resize(src.dims());
    if (size() > 0) {
      if (data_type_.copy()) {
        CAFFE_ENFORCE(
            GetDeviceType() == ::at::DeviceType::CPU,
            "In CopyFrom source and dest tensors must both be CPU for meta copy");
        CAFFE_ENFORCE(
            src.GetDeviceType() == ::at::DeviceType::CPU,
            "In CopyFrom source and dest tensors must both be CPU for meta copy");
        data_type_.copy()(src.raw_data(), raw_mutable_data(), size());
      } else {
        // We'll need to use a non-CPU context to perform the copy if
        // one of the context is not CPU since only non-CPU context
        // knows how to copy between CPU and that context
        if (src.GetDeviceType() != ::at::DeviceType::CPU || GetDeviceType() == ::at::DeviceType::CPU) {
          if (!context) {
            src.CreateContext()->CopyBytesToDevice(
                nbytes(), src.raw_data(), raw_mutable_data(), GetDeviceType());
          } else {
            CAFFE_ENFORCE(
                context->device_type() == src.GetDeviceType(),
                "Type for provided context does not match the type of source");
            context->CopyBytesToDevice(
                nbytes(), src.raw_data(), raw_mutable_data(), GetDeviceType());
          }
        } else {
          // In case source context is CPU, and target context is non-CPU
          // We'll have to create a Context from target and perform the
          // copy using that context
          CreateContext()->CopyBytesFromCPU(
              nbytes(), src.raw_data(), raw_mutable_data());
        }
      }
    }
  }

  /**
   * @brief Extend the outer-most dimension of this tensor
   *        to dimension of `num`.
   */
  void ExtendTo(TIndex num, float growthPct, at::BaseContext* context) {
    CAFFE_ENFORCE_GE_WITH_CALLER(dims_.size(), 1);
    CAFFE_ENFORCE_GE_WITH_CALLER(growthPct, 0);
    CAFFE_ENFORCE(context != nullptr, "Context must be provided.");
    Extend(num - dims_[0], growthPct, context);
  }

  /**
   * @brief Extends the outer-most dimension of this tensor by num elements,
   * preserving the existing data.
   *
   * The underlying data may be reallocated in order to accommodate the new
   * elements, in which case this tensors' capacity is grown at a factor of
   * growthPct. This ensures that Extend runs on an amortized O(1) time
   * complexity.
   */
  void Extend(TIndex num, float growthPct, at::BaseContext* context) {
    CAFFE_ENFORCE_GE_WITH_CALLER(dims_.size(), 1);
    CAFFE_ENFORCE_GE_WITH_CALLER(
        num, 0, "`num` must be non-negative for Extend");
    CAFFE_ENFORCE_WITH_CALLER(
        is_contiguous_,
        "Right now Extend is only supported for contiguous Tensor.");
    auto newDims = dims_;
    newDims[0] += num;
    if (!storage_.data()) {
      Resize(newDims);
      return;
    }
    auto newNumel = std::accumulate(
        newDims.begin(),
        newDims.end(),
        static_cast<TIndex>(1),
        std::multiplies<TIndex>());
    if (newNumel * storage_.itemsize() <= storage_.capacity()) {
      dims_ = newDims;
      numel_ = newNumel;
      return;
    }
    auto newCapacity = dims_;
    newCapacity[0] = std::max<size_t>(
        newDims[0], std::ceil(dims_[0] * (growthPct + 100) / 100));
    auto oldData = std::move(storage_.data_ptr());
    auto oldSize = numel_;
    auto oldDims = dims_;
    Resize(newCapacity);
    auto* newData = raw_mutable_data(data_type_);
    CAFFE_ENFORCE(
        context != nullptr, "Context must be provided to Extend the tensor");
    context->CopyItemsSameDevice(
        data_type_, oldSize, oldData.get(), newData);
    reserved_ = true;
    dims_ = newDims;
    numel_ = newNumel;
  }

  /**
   * @brief Shrinks the outer-most dimension to given size, keeping the data.
   *
   * This method guarantees that no re-allocations are carried out, which means
   * that the extra capacity after the end of the shurnk tensor is maintained.
   */
  void ShrinkTo(TIndex outer_dim) {
    CAFFE_ENFORCE_WITH_CALLER(
        is_contiguous_,
        "Right now ShrinkTo is only supported on contiguous Tensor.");
    CAFFE_ENFORCE_WITH_CALLER(dims_.size() >= 1, "Tensor must be at least 1D");
    CAFFE_ENFORCE_WITH_CALLER(
        outer_dim <= dims_[0],
        "New outer dimension must be smaller than current.");
    CAFFE_ENFORCE(
        storage_.unique(),
        "Can't call ShrinkTo on shared storage, please call Resize instead.");
    dims_[0] = outer_dim;
    numel_ = std::accumulate(
        dims_.begin(),
        dims_.end(),
        static_cast<TIndex>(1),
        std::multiplies<TIndex>());
  }

  /**
   * @brief Reserve space for the underlying tensor.
   *
   * This must be called after Resize(), since we only specify the first
   * dimension This does not copy over the old data to the newly allocated space
   */
  template <class T>
  void ReserveSpace(const T& outer_dim) {
    CAFFE_ENFORCE_WITH_CALLER(
        is_contiguous_,
        "Right now ReserveSpace is only supported for contiguous Tensor.");
    CAFFE_ENFORCE(
        numel_ != -1, "size should be initialized before calling ReserveSpace");
    CAFFE_ENFORCE(
        storage_.unique(), "Can't call ReserveSpace on shared storage.");
    auto newCapacity = dims_;
    newCapacity[0] = outer_dim;
    auto newNumel = std::accumulate(
        newCapacity.begin(),
        newCapacity.end(),
        static_cast<TIndex>(1),
        std::multiplies<TIndex>());
    if (newNumel * storage_.itemsize() <= storage_.capacity()) {
      return;
    }
    // Old data is discarded
    storage_.data_ptr().clear();
    auto oldSize = numel_;
    auto oldDims = dims_;
    Resize(newCapacity);
    // Allocate new memory but don't copy over the data
    raw_mutable_data(data_type_);
    dims_ = oldDims;
    numel_ = oldSize;
    reserved_ = true;
  }

  /**
   * @brief Resizes a tensor.
   *
   * Resize takes in a vector of ints specifying the dimensions of the tensor.
   * You can pass in an empty vector to specify that it is a scalar (i.e.
   * containing one single item).
   *
   * The underlying storage may be deleted after calling Resize: if the new
   * shape leads to a different number of items in the tensor, the old memory
   * is deleted and new memory will be allocated next time you call
   * mutable_data(). However, if the shape is different but the total number of
   * items is the same, the underlying storage is kept.
   */
  template <typename... Ts>
  void Resize(Ts... dim_source) {
    bool is_init = numel_ == -1;
    bool size_changed = SetDims(dim_source...);
    if (size_changed) {
      // If needed, we will free the data. the next mutable_data() call
      // will create the data storage.
      bool reset_tensor = false;
      if (reserved_) {
        // If tensor is reserved then don't claim its memeory unless capacity()
        // is smaller than new size
        reset_tensor = storage_.capacity() < (storage_offset_ + numel_) * storage_.itemsize();
      } else {
        reset_tensor = storage_.capacity() < (storage_offset_ + numel_) * storage_.itemsize() ||
            !FLAGS_caffe2_keep_on_shrink ||
            storage_.capacity() - (storage_offset_ + numel_) * storage_.itemsize() >
                FLAGS_caffe2_max_keep_on_shrink_memory;
      }

      if (reset_tensor && !is_init) {
        FreeMemory();
      }
    }
  }

  /**
   * Resize the tensor like the source tensor. Note that this is just a
   * sugar wrapper that essentially calls Resize(src_tensor.dims()).
   */
  inline void ResizeLike(const TensorImpl& src_tensor) {
    CAFFE_ENFORCE_WITH_CALLER(
        src_tensor.is_contiguous(),
        "Right now ResizeLike is only supported for contiguous Tensor.");
    // Note: need casting for different context types.
    if (static_cast<void*>(this) != static_cast<const void*>(&src_tensor)) {
      Resize(src_tensor.dims());
    }
  }

  /**
   * Resizes the tensor without touching underlying storage.
   * This requires the total size of the tensor to remains constant.
   */
  inline void Reshape(const std::vector<TIndex>& dims) {
    CAFFE_ENFORCE_WITH_CALLER(
        is_contiguous_,
        "Right now Reshape is only supported for contiguous Tensor.");
    TIndex new_size = 1;
    for (auto d : dims) {
      CAFFE_ENFORCE_GE_WITH_CALLER(d, 0);
      new_size *= d;
    }
    CAFFE_ENFORCE_WITH_CALLER(
        new_size == numel_,
        "New size and old size are not equal. You cannot use Reshape, "
        "but should use Resize."
        // TODO(jiayq): remove the following warning after pending diffs
        // stabilize.
        " The old caffe2 mixes Reshape and Resize but this behavior has "
        "been changed. If you find this error, most likely you will need "
        "to change corresponding code from Reshape to Resize.");
    dims_ = dims;
  }

  inline void Reshape(const std::vector<int>& dims) {
    Reshape(ToVectorTIndex(dims));
  }

  /**
   * Release whatever memory the tensor was holding but keep size and type
   * information. Subsequent call to mutable_data will trigger new memory
   * allocation.
   */
  inline void FreeMemory() {
    // We'll detach from the old Storage and create a new one
    storage_ = at::Storage(storage_.device_type(), data_type_);
    storage_offset_ = 0;
  }

  /**
   * A utility function to print the debug string for the tensor. Note that this
   * is very slow since it involves quite some string operations, so do not use
   * it in your performance-critical code.
   */
  std::string DebugString() const {
    std::stringstream ss;
    ss << "A Tensor of item size " << storage_.itemsize() << " and type "
       << data_type_.name() << " and dimension (";
    for (int d : dims_) {
      ss << d << ",";
    }
    ss << ").";
    return ss.str();
  }

  /**
   * @brief Shares the data with another tensor.
   *
   * To share data between two tensors, the sizes of the two tensors must be
   * equal already. The reason we do not implicitly do a Resize to make the two
   * tensors have the same shape is that we want to allow tensors of different
   * shapes but the same number of items to still be able to share data. This
   * allows one to e.g. have a n-dimensional Tensor and a flattened version
   * sharing the same underlying storage.
   *
   * The source tensor should already have its data allocated.
   */
  void ShareData(const TensorImpl& src) {
    // Right now, we are assuming the device_type are the same, since it is
    // inherently the same in the non-templatized code. We should probably add
    // an ENFORCE here which might affect perf a little bit.
    CAFFE_ENFORCE_EQ_WITH_CALLER(
        src.numel_,
        numel_,
        "Size mismatch - did you call reshape before sharing the data?");
    // It is possible that the source tensor hasn't called mutable_data() yet,
    // in which case ShareData() doesn't make much sense since we don't really
    // know what to share yet.
    CAFFE_ENFORCE_WITH_CALLER(
        src.storage_.data() || src.numel_ == 0,
        "Source tensor has no content and has size > 0");
    // Finally, do sharing.
    /* Since we create new Storage whenever we need to change data_type/capacity
     * this still keeps the original semantics
     */
    storage_ = src.storage();
    data_type_ = src.dtype();
    storage_offset_ = src.storage_offset();
  }

  /**
   * @brief Shares the data with an externally managed pointer.
   *
   * This is similar to ShareData() but the source is a pointer with an advanced
   * deleter option. In default, no deletion takes place, and one needs to make
   * sure that the external memory is deallocated only after the tensor finishes
   * using it. If a Deleter object is passed in, when this tensor is reallocated
   * or freed, the deleter function is going to be called.
   */
  template <typename T>
  void
  ShareExternalPointer(T* src, size_t capacity = 0, MemoryDeleter d = nullptr) {
    ShareExternalPointer((void*)src, TypeMeta::Make<T>(), capacity, d);
  }

  template <typename T>
  void ShareExternalPointer(at::DataPtr&& data_ptr, size_t capacity = 0) {
    ShareExternalPointer(std::move(data_ptr), TypeMeta::Make<T>(), capacity);
  }

  void ShareExternalPointer(
      void* src,
      const TypeMeta& data_type,
      size_t capacity = 0,
      MemoryDeleter d = nullptr) {
    CAFFE_ENFORCE_WITH_CALLER(
        is_contiguous_,
        "Right now ShareExternalPointer is only supported for contiguos Tensor.");
    CAFFE_ENFORCE_WITH_CALLER(
        data_type.id() != TypeIdentifier::uninitialized(),
        "To share with a raw external pointer you need to pass in an "
        "initialized data_type(TypeMeta).");
    ShareExternalPointer(
        at::DataPtr(src, src, d, GetDeviceType()), data_type, capacity);
  }

  void ShareExternalPointer(
      at::DataPtr&& data_ptr,
      const TypeMeta& data_type,
      size_t capacity) {
    CAFFE_ENFORCE_WITH_CALLER(
        data_type.id() != TypeIdentifier::uninitialized(),
        "To share with a raw external pointer you need to pass in an "
        "initialized data_type(TypeMeta).");
    if (!capacity) {
      capacity = numel_ * data_type.itemsize();
    }
    if (storage_.unique()) {
      CAFFE_ENFORCE_WITH_CALLER(
          numel_ >= 0,
          "To share data with a raw pointer, you need to set shape first.");
      storage_.UniqueStorageShareExternalPointer(
          std::move(data_ptr), data_type, capacity);
      data_type_ = data_type;
      storage_offset_ = 0;
    } else {
      int64_t numel = capacity / data_type.itemsize();
      // Create a new Storage
      storage_ = at::Storage(data_type, numel, std::move(data_ptr), nullptr, true);
      data_type_ = data_type;
      storage_offset_ = 0;
    }
  }

  /**
   * Returns a const raw void* pointer of the underlying storage. mutable_data()
   * or raw_mutable_data() must have been called prior to this function call.
   */
  inline const void* raw_data() const {
    CAFFE_ENFORCE_WITH_CALLER(storage_.data() || numel_ == 0);
    return static_cast<void*>(static_cast<char*>(storage_.data()) + storage_offset_ * storage_.itemsize());
  }

  /**
   * Returns a typed pointer of the underlying storage. mutable_data() or
   * raw_mutable_data() must have been called prior to this function call, and
   * the data type must be of the correct type. If you want to get a void*
   * pointer instead, use raw_data().
   */
  template <typename T>
  inline const T* data() const {
    CAFFE_ENFORCE_WITH_CALLER(
        storage_.data() || numel_ == 0,
        "The tensor is of non-zero shape, but its data is not allocated yet. "
        "Caffe2 uses a lazy allocation, so you will need to call "
        "mutable_data() or raw_mutable_data() to actually allocate memory.");
    CAFFE_ENFORCE_WITH_CALLER(
        IsType<T>(),
        "Tensor type mismatch, caller expects elements to be ",
        TypeMeta::TypeName<T>(),
        ", while tensor contains ",
        data_type_.name(),
        ". ");
    return static_cast<T*>(storage_.data()) + storage_offset_;
  }

  /**
   * Returns a mutable raw pointer of the underlying storage. Since we will need
   * to know the type of the data for allocation, a TypeMeta object is passed in
   * to specify the necessary information. This is conceptually equivalent of
   * calling mutable_data<T>() where the TypeMeta parameter meta is derived from
   * the type T. This function differs from mutable_data<T>() in the sense that
   * the type T can be specified during runtime via the TypeMeta object.
   *
   * If the existing data does not match the desired type, it will be deleted
   * and a new storage will be created.
   */
  inline void* raw_mutable_data(const TypeMeta& meta) {
    // For 0-size tensors it's fine to return any pointer (including nullptr)
    if (data_type_ == meta && (storage_.data() || numel_ == 0)) {
      return static_cast<void*>(static_cast<char*>(storage_.data()) + storage_offset_ * meta.itemsize());
    } else {
      CAFFE_ENFORCE_WITH_CALLER(
          numel_ >= 0,
          "Tensor is not initialized. You probably need to call Resize() "
          "before calling mutable_data()");
      bool had_special_dtor = data_type_.dtor() != nullptr;
      storage_offset_ = 0;
      if (storage_.unique()) {
        storage_.set_dtype(meta);
      } else {
        if (data_type_ != meta) {
          storage_ = at::Storage(storage_.device_type(), meta);
        }
      }
      data_type_ = meta;

      // We can reuse the existing buffer if the current data does not have
      // a special destructor and the new data doesn't have a special
      // constructor.
      if (numel_ == 0 ||
          (meta.ctor() == nullptr && !had_special_dtor &&
           storage_.numel() >= numel_)) {
        AT_ASSERT(storage_offset_ == 0); // because we just reallocated
        return storage_.data();
      }
      const at::Allocator* allocator = storage_.allocator();
      // TODO: Get rid of StaticContext
      CAFFE_ENFORCE(
          allocator == nullptr,
          "Allocator is not used within Caffe2 functions, please use StaticContext instead.");
      if (meta.ctor()) {
        // For types that need placement new, we will call it, as well as
        // making sure that when the data is freed, it calls the right
        // destruction procedure.
        auto size = numel_;
        auto dtor = data_type_.dtor();
        void* ptr;
        at::DeleterFnPtr deleter;
        auto ptr_and_deleter = GetStaticContext()->New(
            numel_ * storage_.itemsize()); // Removing this can get rid of
                                           // InefficientStdFunctionContext
        ptr = ptr_and_deleter.first;
        deleter = ptr_and_deleter.second;
        storage_.set_data_ptr(at::InefficientStdFunctionContext::makeDataPtr(
            ptr,
            [size, dtor, deleter](void* local_ptr) -> void {
              dtor(local_ptr, size);
              deleter(local_ptr);
            },
            at::Device(storage_.device_type())));
        data_type_.ctor()(storage_.data(), numel_);
      } else {
        // For fundamental type, new and delete is easier.
        auto ptr_and_deleter =
            GetStaticContext()->New(numel_ * storage_.itemsize());
        storage_.set_data_ptr(at::InefficientStdFunctionContext::makeDataPtr(
            ptr_and_deleter.first,
            ptr_and_deleter.second,
            at::Device(storage_.device_type())));
      }
      storage_.set_numel(numel_);
      AT_ASSERT(storage_offset_ == 0); // because we just reallocated
      return storage_.data();
    }
  }

  /**
   * Returns a mutable raw pointer of the underlying storage. This can only be
   * used when you know for sure that the underlying storage of the tensor is
   * already created via an earlier raw_mutable_data(meta) call or a
   * mutable_data<T>() call.
   *
   * If the existing data does not match the desired type, it will be deleted
   * and a new storage will be created.
   */
  inline void* raw_mutable_data() {
    CAFFE_ENFORCE_WITH_CALLER(
        data_type_.id() != TypeIdentifier::uninitialized(),
        "Calling raw_mutable_data() without meta, but the current meta is "
        "of unknown type.");
    return raw_mutable_data(data_type_);
  }

  /**
   * Returns a typed pointer of the underlying storage.
   *
   * For fundamental types, we reuse possible existing storage if there
   * is sufficient capacity.
   */
  template <typename T>
  inline T* mutable_data() {
    if ((numel_ == 0 || storage_.data()) && IsType<T>()) {
      return static_cast<T*>(storage_.data()) + storage_offset_;
    }
    // Check it here statically - otherwise TypeMeta would throw the runtime
    // error in attempt to invoke TypeMeta::ctor()
    static_assert(
        std::is_default_constructible<T>::value,
        "Tensor can't hold non-default-constructible types");
    return static_cast<T*>(raw_mutable_data(TypeMeta::Make<T>()));
  }

  /**
   * Returns the number of dimensions of the data.
   */
  inline int ndim() const {
    return dims_.size();
  }
  /**
   * Returns the size (i.e. the number of items) of the tensor.
   */
  inline TIndex size() const {
    return numel_;
  }
  /**
   * Return the number of bytes each item takes in the tensor.
   */
  inline size_t itemsize() const {
    return storage_.itemsize();
  }
  /**
   * Returns the total number of bytes of the storage.
   *
   * This is equivalent to calling size() * itemsize().
   */
  inline size_t nbytes() const {
    return numel_ * itemsize();
    ;
  }

  // NB: This capacity may also include available space
  // in the storage BEFORE the tensor data, if storage_offset != 0
  inline size_t capacity_nbytes() const {
    return storage_.capacity();
  }
  /**
   * Returns the dimensions of the tensor as a vector.
   */
  inline const std::vector<TIndex>& dims() const {
    return dims_;
  }

  inline TIndex size_from_dim(int k) const {
    return size_from_dim_(k, dims_);
  }

  inline TIndex size_to_dim(int k) const {
    return size_to_dim_(k, dims_);
  }

  inline TIndex size_between_dim(int k, int l) const {
    return size_between_dim_(k, l, dims_);
  }

  /**
   * Returns the 'canonical' version of a (usually)  user-specified axis,
   * allowing for negative indexing (e.g., -1 for the last axis).
   *
   * @param axis_index the axis index.
   *        If 0 <= index < ndim(), return index.
   *        If -ndim <= index <= -1, return (ndim() - (-index)),
   *        e.g., the last axis index (ndim() - 1) if index == -1,
   *        the second to last if index == -2, etc.
   *        Dies on out of range index.
   */
  inline int canonical_axis_index(int axis_index) const {
    return canonical_axis_index_(axis_index, ndim());
  }

  inline int64_t stride(int64_t dim) const {
#ifndef NDEBUG
    // TODO: dim wrapping?
    CAFFE_ENFORCE_LT_WITH_CALLER(dim, strides_.size(), "Exceeding ndim limit");
    CAFFE_ENFORCE_GE_WITH_CALLER(
        dim, 0, "Cannot have negative dimension index");
#endif
    return strides_[dim];
  }

  // TODO: Change to ArrayRef later
  inline at::DimVector strides() {
    return strides_;
  }

  inline bool is_contiguous() const {
    return is_contiguous_;
  }

  /**
   * Checks if the tensor content is of the given data type.
   */
  template <typename T>
  inline bool IsType() const {
    return storage_.IsType<T>();
  }
  /**
   * Returns the TypeMeta object associated with the current data type.
   */
  inline const TypeMeta& meta() const {
    return data_type_;
  }

  inline const TypeMeta& dtype() const {
    return data_type_;
  }

  /**
   * Returns the i-th dimension of the tensor in int.
   *
   * This function returns an int value instead of TIndex, which depending on
   * the typedef could be int64. If you want int64 dim values, make sure you
   * call dim() instead.
   */
  inline int dim32(const int i) const {
#ifndef NDEBUG
    CAFFE_ENFORCE_LT_WITH_CALLER(i, dims_.size(), "Exceeding ndim limit");
    CAFFE_ENFORCE_GE_WITH_CALLER(i, 0, "Cannot have negative dimension index");
#endif
    CAFFE_ENFORCE_LT_WITH_CALLER(dims_[i], std::numeric_limits<int>::max());
    return static_cast<int>(dims_[i]);
  }

  /**
   * Returns the i-th dimension of the tensor. Note that the passed in index
   * must be between 0 (inclusive) and the number of dimensions, otherwise
   * this function will produce a fatal message.
   */
  inline TIndex dim(const int i) const {
#ifndef NDEBUG
    CAFFE_ENFORCE_LT_WITH_CALLER(i, dims_.size(), "Exceeding ndim limit");
    CAFFE_ENFORCE_GE_WITH_CALLER(i, 0, "Cannot have negative dimension index");
#endif
    return dims_[i];
  }

  void ExtractDeviceOption(DeviceOption* device) const {
    auto* context = GetStaticContext();
    CHECK(context);
    context->ExtractDeviceOption(device, raw_data());
  }

  const at::Storage& storage() {
    return storage_;
  }

  const at::Storage& storage() const {
    return storage_;
  }

  int64_t storage_offset() const {
    return storage_offset_;
  }

 protected:
  // TODO: change to DimVector
  std::vector<TIndex> dims_; // sizes_
  at::DimVector strides_;
  TIndex numel_ = -1; // numel_
  bool is_contiguous_ = true;
  // we decide to keep reserved_ and it will
  // live in Tensor after the split
  // The logic is that if Extend() or ReserveSpace() were ever called,
  // then subsequent Resize()s will not free up Storage.
  bool reserved_ = false;
  at::Storage storage_;
  int64_t storage_offset_ = 0;
  TypeMeta data_type_;

 private:
  template <
      typename T,
      typename = typename std::enable_if<std::is_integral<T>::value>::type>
  bool SetDims(const std::vector<T>& src) {
    auto old_numel = numel_;
    dims_.resize(src.size());
    TIndex new_numel = 1;
    for (size_t i = 0; i < src.size(); ++i) {
      new_numel *= src[i];
      dims_[i] = src[i];
    }
    update_strides();
    numel_ = new_numel;
    return numel_ != old_numel;
  }

  bool SetDims() {
    auto old_numel = numel_;
    dims_.resize(0);
    update_strides();
    numel_ = 1;
    return numel_ != old_numel;
  }

  // TODO(jiayq): maybe rewrite the following functions with initializer list.
  // NVCC does not play well with initializer lists last time, but worth
  // another shot.
  bool SetDims(const TIndex d0) {
    auto old_numel = numel_;
    dims_.resize(1);
    dims_[0] = d0;
    update_strides();
    numel_ = d0;
    return numel_ != old_numel;
  }

  bool SetDims(const TIndex d0, const TIndex d1) {
    auto old_numel = numel_;
    dims_.resize(2);
    dims_[0] = d0;
    dims_[1] = d1;
    update_strides();
    numel_ = d0 * d1;
    return numel_ != old_numel;
  }

  bool SetDims(const TIndex d0, const TIndex d1, const TIndex d2) {
    auto old_numel = numel_;
    dims_.resize(3);
    dims_[0] = d0;
    dims_[1] = d1;
    dims_[2] = d2;
    update_strides();
    numel_ = d0 * d1 * d2;
    return numel_ != old_numel;
  }

  bool
  SetDims(const TIndex d0, const TIndex d1, const TIndex d2, const TIndex d3) {
    auto old_numel = numel_;
    dims_.resize(4);
    dims_[0] = d0;
    dims_[1] = d1;
    dims_[2] = d2;
    dims_[3] = d3;
    update_strides();
    numel_ = d0 * d1 * d2 * d3;
    return numel_ != old_numel;
  }

  inline void update_strides() {
    strides_.resize(dims_.size());
    if (ndim() > 0) {
      int last_idx = ndim() - 1;
      strides_[last_idx] = 1;
      for (auto i = last_idx - 1; i >= 0; --i) {
        strides_[i] = strides_[i + 1] * std::max<int64_t>(dims_[i + 1], 1);
      }
    }
    is_contiguous_ = true;
  }
};

}
