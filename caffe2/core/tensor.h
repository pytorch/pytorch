#ifndef CAFFE2_CORE_TENSOR_H_
#define CAFFE2_CORE_TENSOR_H_

#include "caffe2/core/storage.h"

#include <ATen/core/intrusive_ptr.h>

// A global boolean variable to control whether we free memory when a Tensor
// is shrinked to a smaller size. As a result, a Tensor is always going to
// keep the memory allocated for its maximum capacity reshaped to so far.
CAFFE2_DECLARE_bool(caffe2_keep_on_shrink);

// Since we can have high variance in blob memory allocated across different
// inputs in the same run, we will shrink the blob only if the memory gain
// is larger than this flag in bytes.
CAFFE2_DECLARE_int64(caffe2_max_keep_on_shrink_memory);

namespace caffe2 {

using DimVector = std::vector<TIndex>;

/**
 * A utility function to convert vector<int> to vector<TIndex>.
 */
inline vector<TIndex> ToVectorTIndex(const std::vector<int>& src) {
  return vector<TIndex>(src.begin(), src.end());
}

/**
 * Return product of all dimensions starting from k
 */
inline TIndex size_from_dim_(int k, const vector<TIndex>& dims) {
  TIndex r = 1;
  for (size_t i = k; i < dims.size(); ++i) {
    r *= dims[i];
  }
  return r;
}

// Product of all dims up to k (not including dims[k])
inline TIndex size_to_dim_(int k, const vector<TIndex>& dims) {
  CAFFE_ENFORCE((unsigned)k <= dims.size());
  TIndex r = 1;
  for (int i = 0; i < k; ++i) {
    r *= dims[i];
  }
  return r;
}

// Product of all dims between k and l (not including dims[k] and dims[l])
inline TIndex size_between_dim_(int k, int l, const vector<TIndex>& dims) {
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
  explicit TensorImpl(DeviceType device_type) : storage_(device_type) {}

  /**
   * @brief Creates a tensor of the given dimension.
   *
   * Note that the actual data allocation is not going to be carried out until
   * the first time mutable_data() is called.
   */
  // TODO: here, we create a Storage
  // and immediately discard it in Resize() since
  // reset_tensor will be true and FreeMemory will be called,
  // we might want to avoid creating Storage twice?
  explicit TensorImpl(const vector<TIndex>& dims, DeviceType device_type)
      : storage_(device_type) {
    Resize(dims);
  }

  explicit TensorImpl(const vector<int>& dims, DeviceType device_type)
      : storage_(device_type) {
    Resize(dims);
  }

  /* Now we require that context_for_copy has the same device type as src since
   * template is removed
   */
  TensorImpl(
      const TensorImpl& src,
      BaseContext* context_for_copy,
      DeviceType device_type)
      : storage_(device_type) {
    CopyFrom(src, context_for_copy);
  }

  /**
   * @brief: Create a Tensor of DeviceType `type` and initialize it with
   * src Tensor
   */
  TensorImpl(const TensorImpl& src, DeviceType device_type)
      : storage_(device_type) {
    CopyFrom(src);
  }

  /**
   * @brief Creates a tensor, and fills its contents with the given values.
   * The type of tensor will be decided by the context parameter
   */
  template <typename T>
  TensorImpl(
      const vector<TIndex>& dims,
      const vector<T>& values,
      BaseContext* context)
      : storage_(context->GetDevicetype(), TypeMeta::Make<T>()) {
    Resize(dims);
    CAFFE_ENFORCE_EQ_WITH_CALLER(values.size(), numel_);
    context->CopyItemsFromCPU(
        storage_.dtype(), numel_, values.data(), mutable_data<T>());
  }

  /**
   * @brief Creates a scalar tensor, and fills its content with the given value.
   * The type of tensor will be decided by the context parameter
   */
  template <
      typename T,
      typename = typename std::enable_if<std::is_scalar<T>::value>::type>
  TensorImpl(const T& value, BaseContext* context)
      : storage_(context->GetDevicetype(), TypeMeta::Make<T>()) {
    Resize(vector<TIndex>{});
    context->CopyItemsFromCPU(
        storage_.dtype(), numel_, &value, mutable_data<T>());
  }

  /**
   * @brief Delete the copy constructor and use Clone explicitly
   */
  TensorImpl(const TensorImpl& src) = delete;

  TensorImpl(TensorImpl&& src) noexcept {
    swap(src);
  }

  TensorImpl& operator=(TensorImpl&&) = default;
  // Note(jiayq): possibly a rule-of-three violation, but we explicitly
  // discourage the use of = for Tensors.
  TensorImpl& operator=(const TensorImpl& src) = delete;

  virtual ~TensorImpl() noexcept {}

  /*
   * Since we removed template from tensor, we now store a static
   * context pointer in tensor, which indicates the type of the tensor.
   */
  BaseStaticContext* GetStaticContext() const {
    return GET_STATIC_CONTEXT(GetDeviceType());
  }

  /* @brief
   * Create a context that has the same device_type
   * as the tensor.
   * Note that this doesn't support passing in argument
   * TODO(jerryzh): move this to a global registry
   * that can create context for us
   */
  std::unique_ptr<BaseContext> CreateContext() const {
    return GetStaticContext()->CreateContext();
  }

  DeviceType GetDeviceType() const {
    return storage_.device_type();
  }

  /**
   * @brief Copies the data from a source tensor, with a contex provided to
   * carry out the underlying memcpy operation.
   */
  void CopyFrom(const TensorImpl& src, BaseContext* context = nullptr) {
    if ((void*)&src == (void*)this) {
      return;
    }
    CAFFE_ENFORCE_WITH_CALLER(
        src.is_contiguous(),
        "Source Tensor must be contiguous in order to be copied.");
    if (storage_.dtype() != src.meta()) {
      storage_ = Storage(GetDeviceType(), src.meta());
    }
    if (src.size() == -1) {
      dims_.clear();
      strides_.clear();
      is_contiguous_ = true;
      numel_ = -1;
      storage_.reset();
      return;
    }
    Resize(src.dims());
    if (size() > 0) {
      if (storage_.dtype().copy()) {
        CAFFE_ENFORCE(
            GetDeviceType() == CPU,
            "In CopyFrom source and dest tensors must both be CPU for meta copy");
        CAFFE_ENFORCE(
            src.GetDeviceType() == CPU,
            "In CopyFrom source and dest tensors must both be CPU for meta copy");
        storage_.dtype().copy()(src.raw_data(), raw_mutable_data(), size());
      } else {
        // We'll need to use a non-CPU context to perform the copy if
        // one of the context is not CPU since only non-CPU context
        // knows how to copy between CPU and that context
        if (src.GetDeviceType() != CPU || GetDeviceType() == CPU) {
          if (!context) {
            src.CreateContext()->CopyBytesToDevice(
                nbytes(), src.raw_data(), raw_mutable_data(), GetDeviceType());
          } else {
            CAFFE_ENFORCE(
                context->GetDevicetype() == src.GetDeviceType(),
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
  void ExtendTo(TIndex num, float growthPct, BaseContext* context) {
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
  void Extend(TIndex num, float growthPct, BaseContext* context) {
    CAFFE_ENFORCE_WITH_CALLER(
        is_contiguous_, "Tensor must be contiguous in order to call Extend.");
    CAFFE_ENFORCE_GE_WITH_CALLER(dims_.size(), 1);
    CAFFE_ENFORCE_GE_WITH_CALLER(
        num, 0, "`num` must be non-negative for Extend");
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
    auto* newData = raw_mutable_data(storage_.dtype());
    CAFFE_ENFORCE(
        context != nullptr, "Context must be provided to Extend the tensor");
    context->CopyItemsSameDevice(
        storage_.dtype(), oldSize, oldData.get(), newData);
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
        is_contiguous_, "Tensor must be contiguous in order to call ShrinkTo.");
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
        "Tensor must be contiguous in order to call ReserveSpace.");
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
    storage_.data_ptr().reset();
    auto oldSize = numel_;
    auto oldDims = dims_;
    Resize(newCapacity);
    // Allocate new memory but don't copy over the data
    raw_mutable_data(storage_.dtype());
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
        reset_tensor = storage_.capacity() < numel_ * storage_.itemsize();
      } else {
        reset_tensor = storage_.capacity() < numel_ * storage_.itemsize() ||
            !FLAGS_caffe2_keep_on_shrink ||
            storage_.capacity() - numel_ * storage_.itemsize() >
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
        "Tensor must be contiguous in order to call ResizeLike.");
    // Note: need casting for different context types.
    if (static_cast<void*>(this) != static_cast<const void*>(&src_tensor)) {
      Resize(src_tensor.dims());
    }
  }

  /**
   * Resizes the tensor without touching underlying storage.
   * This requires the total size of the tensor to remains constant.
   */
  inline void Reshape(const vector<TIndex>& dims) {
    CAFFE_ENFORCE_WITH_CALLER(
        is_contiguous_, "Tensor must be contiguous in order to call Reshape.");
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

  inline void Reshape(const vector<int>& dims) {
    Reshape(ToVectorTIndex(dims));
  }

  /**
   * Release whatever memory the tensor was holding but keep size and type
   * information. Subsequent call to mutable_data will trigger new memory
   * allocation.
   */
  inline void FreeMemory() {
    // We'll detach from the old Storage and create a new one
    storage_ = Storage(storage_.device_type(), storage_.dtype());
  }

  /**
   * A utility function to print the debug string for the tensor. Note that this
   * is very slow since it involves quite some string operations, so do not use
   * it in your performance-critical code.
   */
  string DebugString() const {
    std::stringstream ss;
    ss << "A Tensor of item size " << storage_.itemsize() << " and type "
       << storage_.dtype().name() << " and dimension (";
    for (int d : dims_) {
      ss << d << ",";
    }
    ss << ").";
    return ss.str();
  }

  void swap(TensorImpl& other) noexcept {
    std::swap(dims_, other.dims_);
    std::swap(strides_, other.strides_);
    std::swap(is_contiguous_, other.is_contiguous_);
    std::swap(numel_, other.numel_);
    std::swap(storage_, other.storage_);
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
  template <typename T, typename Deleter = MemoryDeleter>
  void ShareExternalPointer(T* src, size_t capacity = 0, Deleter d = nullptr) {
    ShareExternalPointer(src, TypeMeta::Make<T>(), capacity, d);
  }

  template <typename Deleter = MemoryDeleter>
  void ShareExternalPointer(
      void* src,
      const TypeMeta& data_type,
      size_t capacity = 0,
      Deleter d = nullptr) {
    CAFFE_ENFORCE_WITH_CALLER(
        is_contiguous_,
        "Tensor must be contiguous in order to call ShareExternalPointer.");
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
      storage_.UniqueStorageShareExternalPointer(src, data_type, capacity, d);
    } else {
      // Create a new Storage
      storage_ = Storage(src, GetDeviceType(), data_type, capacity, d);
    }
  }

  /**
   * Returns a const raw void* pointer of the underlying storage. mutable_data()
   * or raw_mutable_data() must have been called prior to this function call.
   */
  inline const void* raw_data() const {
    CAFFE_ENFORCE_WITH_CALLER(
        is_contiguous_,
        "Tensor must be contiguous in order to call raw_data()");
    CAFFE_ENFORCE_WITH_CALLER(storage_.data() || numel_ == 0);
    return storage_.data();
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
        is_contiguous_, "Tensor must be contiguous in order to call data()");
    CAFFE_ENFORCE_WITH_CALLER(
        storage_.data() || numel_ == 0,
        "The tensor is of non-zero shape, but its data is not allocated yet. "
        "Caffe2 uses a lazy allocation, so you will need to call "
        "mutable_data() or raw_mutable_data() to actually allocate memory.");
    CAFFE_ENFORCE_WITH_CALLER(
        IsType<T>(),
        "Tensor type mismatch, caller expects elements to be ",
        TypeMeta::TypeName<T>(),
        " while tensor contains ",
        storage_.dtype().name());
    return static_cast<T*>(storage_.data());
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
    CAFFE_ENFORCE_WITH_CALLER(
        is_contiguous_,
        "Tensor must be contiguous in order to call raw_mutable_data()");
    // For 0-size tensors it's fine to return any pointer (including nullptr)
    if (storage_.dtype() == meta && (storage_.data() || numel_ == 0)) {
      return storage_.data();
    } else {
      bool had_special_dtor = storage_.dtype().dtor() != nullptr;
      if (storage_.unique()) {
        storage_.set_dtype(meta);
        // TODO: recalcuate numel when we store numel instead of capacity in
        // Storage
      } else {
        if (storage_.dtype() != meta) {
          storage_ = Storage(storage_.device_type(), meta);
        }
      }
      CAFFE_ENFORCE_WITH_CALLER(
          numel_ >= 0,
          "Tensor is not initialized. You probably need to call Resize() "
          "before calling mutable_data()");

      // We can reuse the existing buffer if the current data does not have
      // a special destructor and the new data doesn't have a special
      // constructor.
      if (numel_ == 0 ||
          (meta.ctor() == nullptr && !had_special_dtor &&
           storage_.capacity() >= numel_ * storage_.itemsize())) {
        return storage_.data();
      }
      if (meta.ctor()) {
        // For types that need placement new, we will call it, as well as
        // making sure that when the data is freed, it calls the right
        // destruction procedure.
        auto size = numel_;
        auto dtor = storage_.dtype().dtor();
        auto ptr_and_deleter =
            GetStaticContext()->New(numel_ * storage_.itemsize());
        auto deleter = ptr_and_deleter.second;
        storage_.data_ptr().reset(
            ptr_and_deleter.first, [size, dtor, deleter](void* ptr) -> void {
              dtor(ptr, size);
              deleter(ptr);
            });
        storage_.dtype().ctor()(storage_.data(), numel_);
      } else {
        // For fundamental type, new and delete is easier.
        auto ptr_and_deleter =
            GetStaticContext()->New(numel_ * storage_.itemsize());
        storage_.data_ptr().reset(
            ptr_and_deleter.first, ptr_and_deleter.second);
      }
      storage_.set_numel(numel_);
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
        is_contiguous_,
        "Tensor must be contiguous in order to call raw_mutable_data()");
    CAFFE_ENFORCE_WITH_CALLER(
        storage_.dtype().id() != TypeIdentifier::uninitialized(),
        "Calling raw_mutable_data() without meta, but the current meta is "
        "of unknown type.");
    return raw_mutable_data(storage_.dtype());
  }

  /**
   * Returns a typed pointer of the underlying storage.
   *
   * For fundamental types, we reuse possible existing storage if there
   * is sufficient capacity.
   */
  template <typename T>
  inline T* mutable_data() {
    CAFFE_ENFORCE_WITH_CALLER(
        is_contiguous_,
        "Tensor must be contiguous in order to call mutable_data()");
    if ((numel_ == 0 || storage_.data()) && IsType<T>()) {
      return static_cast<T*>(storage_.data());
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

  inline size_t capacity_nbytes() const {
    return storage_.capacity();
  }
  /**
   * Returns the dimensions of the tensor as a vector.
   */
  inline const vector<TIndex>& dims() const {
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
  inline DimVector strides() {
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
    return storage_.dtype();
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
    GetStaticContext()->ExtractDeviceOption(device, raw_data());
  }

  const Storage& storage() {
    return storage_;
  }

  const Storage& storage() const {
    return storage_;
  }

 protected:
  DimVector dims_; // sizes_
  DimVector strides_;
  TIndex numel_ = -1; // numel_
  bool is_contiguous_ = true;
  // we decide to keep reserved_ and it will
  // live in Tensor after the split
  // The logic is that if Extend() or ReserveSpace() were ever called,
  // then subsequent Resize()s will not free up Storage.
  bool reserved_ = false;
  Storage storage_;
  // int64_t storage_offset_;

 private:
  template <
      typename T,
      typename = typename std::enable_if<std::is_integral<T>::value>::type>
  bool SetDims(const vector<T>& src) {
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
    for (auto i = ndim() - 1; i >= 0; --i) {
      if (i == ndim() - 1) {
        strides_[i] = 1;
      } else {
        strides_[i] = strides_[i + 1] * std::max<int64_t>(dims_[i + 1], 1);
      }
    }
    is_contiguous_ = true;
  }
};

class CAFFE2_API UndefinedTensorImpl final : public TensorImpl {
  UndefinedTensorImpl() : TensorImpl(CPU){};

 public:
  static constexpr TensorImpl* singleton() {
    return &singleton_;
  }

 private:
  static UndefinedTensorImpl singleton_;
};

/**
 * @brief Tensor class holds a shared pointer to the implementation TensorImpl,
 * redirects API calls to TensorImpl;
 * Copying of Tensor results in sharing the same underlying implementation
 * object
 */
class CAFFE2_API Tensor final {
 protected:
  using TensorImplPtr = c10::intrusive_ptr<TensorImpl, UndefinedTensorImpl>;
  TensorImplPtr impl_;

 public:
  Tensor() : impl_() {}

  operator bool() const {
    return impl_.defined();
  }

  TensorImpl* unsafeGetTensorImpl() const {
    return impl_.get();
  }

  explicit Tensor(DeviceType type)
      : impl_(c10::make_intrusive<TensorImpl, UndefinedTensorImpl>(type)) {}

  explicit Tensor(const vector<TIndex>& dims, DeviceType type)
      : impl_(
            c10::make_intrusive<TensorImpl, UndefinedTensorImpl>(dims, type)) {}

  explicit Tensor(const vector<int>& dims, DeviceType type)
      : impl_(
            c10::make_intrusive<TensorImpl, UndefinedTensorImpl>(dims, type)) {}

  Tensor(const Tensor& src, BaseContext* context_for_copy, DeviceType type)
      : impl_(c10::make_intrusive<TensorImpl, UndefinedTensorImpl>(
            *src.impl_,
            context_for_copy,
            type)) {}

  Tensor(const Tensor& src, DeviceType type)
      : impl_(c10::make_intrusive<TensorImpl, UndefinedTensorImpl>(
            *src.impl_,
            type)) {}

  template <typename T>
  Tensor(
      const vector<TIndex>& dims,
      const vector<T>& values,
      BaseContext* context)
      : impl_(c10::make_intrusive<TensorImpl, UndefinedTensorImpl>(
            dims,
            values,
            context)) {}

  template <
      typename T,
      typename = typename std::enable_if<std::is_scalar<T>::value>::type>
  Tensor(const T& value, BaseContext* context)
      : impl_(c10::make_intrusive<TensorImpl, UndefinedTensorImpl>(
            value,
            context)) {}

  Tensor Clone() const {
    Tensor x(GetDeviceType());
    x.CopyFrom(*this);
    return x;
  }

  BaseStaticContext* GetStaticContext() const {
    return impl_.get()->GetStaticContext();
  }

  std::unique_ptr<BaseContext> CreateContext() const {
    return impl_.get()->CreateContext();
  }

  DeviceType GetDeviceType() const {
    return impl_.get()->GetDeviceType();
  }

  void CopyFrom(const Tensor& src, BaseContext* context = nullptr) const {
    impl_.get()->CopyFrom(*src.impl_.get(), context);
  }

  void ExtendTo(TIndex num, float growthPct, BaseContext* context) const {
    impl_.get()->ExtendTo(num, growthPct, context);
  }

  void Extend(TIndex num, float growthPct, BaseContext* context) const {
    impl_.get()->Extend(num, growthPct, context);
  }

  void ShrinkTo(TIndex outer_dim) const {
    impl_.get()->ShrinkTo(outer_dim);
  }

  template <class T>
  void ReserveSpace(const T& outer_dim) const {
    impl_.get()->ReserveSpace(outer_dim);
  }

  template <typename... Ts>
  void Resize(Ts... dim_source) const {
    impl_.get()->Resize(dim_source...);
  }

  inline void ResizeLike(const Tensor& src_tensor) const {
    impl_.get()->ResizeLike(*src_tensor.impl_.get());
  }

  inline void Reshape(const vector<TIndex>& dims) const {
    impl_.get()->Reshape(dims);
  }

  inline void Reshape(const vector<int>& dims) const {
    impl_.get()->Reshape(dims);
  }

  inline void FreeMemory() const {
    impl_.get()->FreeMemory();
  }

  string DebugString() const {
    return impl_.get()->DebugString();
  }

  // NB: a.swap(b) is not equivalent to std::swap(a, b);
  // swap method swaps the CONTENTS of the tensors, while std::swap
  // swaps the POINTERS.
  void swap(const Tensor& other) const noexcept {
    impl_.get()->swap(*other.impl_.get());
  }

  void ShareData(const Tensor& src) const {
    impl_.get()->ShareData(*src.impl_.get());
  }

  template <typename T, typename Deleter = MemoryDeleter>
  void ShareExternalPointer(T* src, size_t capacity = 0, Deleter d = nullptr)
      const {
    impl_.get()->ShareExternalPointer<T, Deleter>(src, capacity, d);
  }

  template <typename Deleter = MemoryDeleter>
  void ShareExternalPointer(
      void* src,
      const TypeMeta& meta,
      size_t capacity = 0,
      Deleter d = nullptr) const {
    impl_.get()->ShareExternalPointer<Deleter>(src, meta, capacity, d);
  }

  inline const void* raw_data() const {
    return impl_.get()->raw_data();
  }

  template <typename T>
  inline const T* data() const {
    return impl_.get()->data<T>();
  }

  inline void* raw_mutable_data(const TypeMeta& meta) const {
    return impl_.get()->raw_mutable_data(meta);
  }

  inline void* raw_mutable_data() const {
    return impl_.get()->raw_mutable_data();
  }

  template <typename T>
  inline T* mutable_data() const {
    return impl_.get()->mutable_data<T>();
  }

  inline int ndim() const {
    return impl_.get()->ndim();
  }

  inline TIndex size() const {
    return impl_.get()->size();
  }

  inline size_t itemsize() const {
    return impl_.get()->itemsize();
  }

  inline size_t nbytes() const {
    return impl_.get()->nbytes();
  }

  inline size_t capacity_nbytes() const {
    return impl_.get()->capacity_nbytes();
  }

  inline const vector<TIndex>& dims() const {
    return impl_.get()->dims();
  }

  inline TIndex size_from_dim(int k) const {
    return impl_.get()->size_from_dim(k);
  }

  inline TIndex size_to_dim(int k) const {
    return impl_.get()->size_to_dim(k);
  }

  inline TIndex size_between_dim(int k, int l) const {
    return impl_.get()->size_between_dim(k, l);
  }

  inline int canonical_axis_index(int axis_index) const {
    return impl_.get()->canonical_axis_index(axis_index);
  }

  inline int64_t stride(int64_t dim) const {
    return impl_.get()->stride(dim);
  }

  inline DimVector strides() {
    return impl_.get()->strides();
  }

  inline bool is_contiguous() const {
    return impl_.get()->is_contiguous();
  }

  template <typename T>
  inline bool IsType() const {
    return impl_.get()->IsType<T>();
  }

  inline const TypeMeta& meta() const {
    return impl_.get()->meta();
  }

  inline int dim32(const int i) const {
    return impl_.get()->dim32(i);
  }

  inline TIndex dim(const int i) const {
    return impl_.get()->dim(i);
  }

  inline void ExtractDeviceOption(DeviceOption* device) const {
    return impl_.get()->ExtractDeviceOption(device);
  }
};

using TensorCPU = Tensor;

constexpr int k_limit_default_ = 1000;

// TODO: the following logic can be merged into regular Tensor class methods
// after MKLMemory starts to implement Tensor interface

// Type call registry
typedef TypeMeta (*TypeCall)(const void*);
TypeCall GetTypeCallFunction(TypeIdentifier id);
void RegisterTypeCallFunction(TypeIdentifier id, TypeCall c);

// Shape call registry
typedef vector<TIndex> (*TensorInfoCall)(
    const void*,
    size_t* capacity,
    DeviceOption* device);
TensorInfoCall GetTensorInfoFunction(TypeIdentifier id);
void RegisterTensorInfoFunction(TypeIdentifier id, TensorInfoCall c);

// resize helper function
void TensorVectorResize(
    std::vector<Tensor>& tensors,
    int size,
    DeviceType type);

class CAFFE2_API TensorPrinter {
 public:
  explicit TensorPrinter(
      const std::string& tensor_name = "",
      const std::string& file_name = "",
      int limit = k_limit_default_);
  ~TensorPrinter();

  template <class T>
  void Print(const Tensor& tensor);

  void PrintMeta(const Tensor& tensor);

  string MetaStr(const Tensor& tensor);

 private:
  bool to_file_;
  int limit_;
  std::unique_ptr<std::ofstream> log_file_;
  std::string tensor_name_;
};

template <class T>
void TensorPrinter::Print(const Tensor& tensor) {
  std::stringstream values_stream;
  // One most likely doesn't want to print int64-number of items for visual
  // inspection, so we cast down to int here.
  int total_count = static_cast<int>(std::min(tensor.size(), TIndex(limit_)));
  const T* tensor_data = tensor.template data<T>();
  for (int i = 0; i < total_count - 1; ++i) {
    values_stream << tensor_data[i] << ",";
  }
  // We do not add a comma after the last item.
  values_stream << tensor_data[total_count - 1];
  if (to_file_) {
    (*log_file_) << MetaStr(tensor) << values_stream.str() << std::endl;
  } else {
    // Log to console.
    LOG(INFO) << MetaStr(tensor) << values_stream.str();
  }
}

} // namespace caffe2
#endif // CAFFE2_CORE_TENSOR_H_
