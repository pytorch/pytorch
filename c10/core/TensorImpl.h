#pragma once

#include <atomic>
#include <memory>
#include <numeric>

#include <c10/core/Backend.h>
#include <c10/core/MemoryFormat.h>
#include <c10/core/Storage.h>
#include <c10/core/TensorOptions.h>
#include <c10/core/DispatchKeySet.h>
#include <c10/core/impl/LocalDispatchKeySet.h>
#include <c10/core/CopyBytes.h>

#include <c10/util/Exception.h>
#include <c10/util/Optional.h>
#include <c10/util/Flags.h>
#include <c10/util/Logging.h>
#include <c10/util/python_stub.h>

// A global boolean variable to control whether we free memory when a Tensor
// is shrinked to a smaller size. As a result, a Tensor is always going to
// keep the memory allocated for its maximum capacity reshaped to so far.
//
// This parameter is respected "upper-case" methods which call Resize()
// (e.g., CopyFrom, ResizeLike); it is NOT respected by Tensor::resize_
// or ShrinkTo, both of which guarantee to never to free memory.
C10_DECLARE_bool(caffe2_keep_on_shrink);

// Since we can have high variance in blob memory allocated across different
// inputs in the same run, we will shrink the blob only if the memory gain
// is larger than this flag in bytes.  This only applies to functions which
// respect caffe2_keep_on_shrink.
C10_DECLARE_int64(caffe2_max_keep_on_shrink_memory);


namespace at {
class Tensor;
}

namespace c10 {
class Scalar;
struct Storage;

/**
 * A utility function to convert vector<int> to vector<int64_t>.
 */
inline std::vector<int64_t> ToVectorint64_t(ArrayRef<int> src) {
  return std::vector<int64_t>(src.begin(), src.end());
}

/**
 * Return product of all dimensions starting from k
 */
inline int64_t size_from_dim_(int k, IntArrayRef dims) {
  int64_t r = 1;
  for (size_t i = k; i < dims.size(); ++i) {
    r *= dims[i];
  }
  return r;
}

// Product of all dims up to k (not including dims[k])
inline int64_t size_to_dim_(int k, IntArrayRef dims) {
  TORCH_CHECK((unsigned)k <= dims.size());
  int64_t r = 1;
  for (int i = 0; i < k; ++i) {
    r *= dims[i];
  }
  return r;
}

// Product of all dims between k and l (not including dims[k] and dims[l])
inline int64_t size_between_dim_(int k, int l, IntArrayRef dims) {
  TORCH_CHECK((unsigned)l < dims.size());
  int64_t r = 1;
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
  TORCH_CHECK(axis_index >= -ndims);
  TORCH_CHECK(axis_index < ndims);
  if (axis_index < 0) {
    return axis_index + ndims;
  }
  return axis_index;
}

using PlacementDtor = void (*)(void*, size_t);

/*
 * A Context that will call extra placement deleter during
 * deconstruction.
 *
 * Accept a already constructed DataPtr and store it as member
 * during destruction, we'll call extra deleter on the underlying
 * data pointer before the DataPtr is destructed.
 * `data_ptr_` owns the memory.
 */
struct C10_API PlacementDeleteContext {
  DataPtr data_ptr_;
  PlacementDtor placement_dtor_;
  size_t size_;
  PlacementDeleteContext(
      DataPtr&& data_ptr,
      PlacementDtor placement_dtor,
      size_t size)
      : data_ptr_(std::move(data_ptr)),
        placement_dtor_(placement_dtor),
        size_(size) {}
  static DataPtr makeDataPtr(
      DataPtr&& data_ptr,
      PlacementDtor placement_dtor,
      size_t size,
      Device device);
  ~PlacementDeleteContext() {
    placement_dtor_(data_ptr_.get(), size_);
    // original memory will be freed when data_ptr_ is destructed
  }
};

struct TensorImpl;

struct C10_API AutogradMetaInterface {
  virtual void set_requires_grad(bool requires_grad, at::TensorImpl* self_impl) = 0;
  virtual bool requires_grad() const = 0;
  virtual at::Tensor& grad() = 0;
  virtual const at::Tensor& grad() const = 0;
  virtual ~AutogradMetaInterface();
};

namespace impl {

// Unfortunately, the definition of AutogradMeta lives in a separate
// compilation unit than TensorImpl (libtorch.so versus libc10.so)
// which means that we cannot construct an AutogradMeta from TensorImpl,
// not even from the cpp file.  So we have to indirect it through a factory
// function which will be initialized when we load libtorch.so.

struct C10_API AutogradMetaFactory {
  virtual ~AutogradMetaFactory() = default;
  virtual std::unique_ptr<AutogradMetaInterface> make() const = 0;
  // This method is the dumbest method.  But I don't have access
  // to Tensor (not TensorImpl) which is undefined in this header.
  virtual const at::Tensor& undefined_tensor() const = 0;
};

C10_API void SetAutogradMetaFactory(AutogradMetaFactory* factory);
C10_API AutogradMetaFactory* GetAutogradMetaFactory();

struct C10_API AutogradMetaFactoryRegisterer {
  explicit AutogradMetaFactoryRegisterer(AutogradMetaFactory* factory) {
    SetAutogradMetaFactory(factory);
  }
};

} // namespace impl

struct C10_API NamedTensorMetaInterface {
  virtual ~NamedTensorMetaInterface() {};
  virtual std::unique_ptr<NamedTensorMetaInterface> clone() const {
    TORCH_INTERNAL_ASSERT(
      false,
      "Not implemented: NamedTensorMetaInterface::clone");
  };
  virtual int64_t slow_dim() const {
    TORCH_INTERNAL_ASSERT(
      false,
      "Not implemented: NamedTensorMetaInterface::slow_dim");
  };
};

// NOTE [ Version Counter Sharing ]
//
// Every Tensor has a version counter. Version counters are incremented whenever the
// data or size of a tensor changes through in-place Variable operations. Version
// counters are used to detect modifications to saved variables which would result in
// incorrect gradient calculations. Version counters may be shared between Variables:
//
// 1. A view shares the version counter of the base Variable,
// 2. `x.detach()` shares the version counter of `x`,
// 3. Unpacked saved variables share the version counter of the source.
//
// Version counters are not shared in these scenarios:
//
// 1. When we replace a `Variable`'s underlying `Tensor` by calling `set_data(...)`,
// 2. `x.data` does not share the version counter of `x`. (See discussion at
// https://github.com/pytorch/pytorch/issues/5396)
//
// Question: Why do we put the version counter in TensorImpl instead of AutogradMeta?
//
// Answer: After the Variable/Tensor merge, a tensor will not have AutogradMeta when
// its `requires_grad_` is false, but when we use this tensor in the forward pass of
// a function that requires saving this tensor for backward, we need to keep track of
// this tensor's version to make sure it's always valid in the autograd graph.
//
// To achieve this goal, we put the version counter in TensorImpl instead of AutogradMeta,
// and have it always be available. This allows us to have the optimization of not
// carrying AutogradMeta when a tensor doesn't require gradient.
//
// A hypothetical alternative way to achieve this goal is to initialize AutogradMeta and
// create the version counter for the non-requires-grad tensor only when it's saved for
// backward. However, since saving a tensor for backward happens in the forward pass, and
// our invariant is that forward pass needs to be thread-safe, lazy-initializing AutogradMeta
// when saving a tensor can introduce race conditions when we are running the forward
// pass in multi-thread scenarios, thus making the forward pass not thread-safe anymore,
// which breaks the invariant.
struct C10_API VariableVersion {
 private:
  struct VersionCounter : intrusive_ptr_target {
    VersionCounter(uint32_t version) : version_(version) {}
    std::atomic<uint32_t> version_;
  };
  c10::intrusive_ptr<VersionCounter> version_counter_;

 public:
  bool unique() const {
    return 1 == version_counter_.use_count();
  }
  // NOTE: As of C++11 and 14, default-constructing a std::atomic variable
  // leaves it in a persistently undefined state. See
  // https://cplusplus.github.io/LWG/issue2334.
  VariableVersion(uint32_t version = 0)
      : version_counter_(c10::make_intrusive<VersionCounter>(version)) {}

  void bump() noexcept {
    ++version_counter_->version_;
  }

  uint32_t current_version() const noexcept {
    return version_counter_->version_;
  }
};

/**
 * The low-level representation of a tensor, which contains a pointer
 * to a storage (which contains the actual data) and metadata (e.g., sizes and
 * strides) describing this particular view of the data as a tensor.
 *
 * Some basic characteristics about our in-memory representation of
 * tensors:
 *
 *  - It contains a pointer to a storage struct (Storage/StorageImpl)
 *    which contains the pointer to the actual data and records the
 *    data type and device of the view.  This allows multiple tensors
 *    to alias the same underlying data, which allows to efficiently
 *    implement differing *views* on a tensor.
 *
 *  - The tensor struct itself records view-specific metadata about
 *    the tensor, e.g., sizes, strides and offset into storage.
 *    Each view of a storage can have a different size or offset.
 *
 *  - This class is intrusively refcounted.  It is refcounted so that
 *    we can support prompt deallocation of large tensors; it is
 *    intrusively refcounted so that we can still perform reference
 *    counted operations on raw pointers, which is often more convenient
 *    when passing tensors across language boundaries.
 *
 *  - For backwards-compatibility reasons, a tensor may be in an
 *    uninitialized state.  A tensor may be uninitialized in the following
 *    two ways:
 *
 *      - A tensor may be DTYPE UNINITIALIZED.  A tensor of this
 *        form has an uninitialized dtype.  This situation most
 *        frequently arises when a user writes Tensor x(CPU).  The dtype and
 *        is subsequently initialized when mutable_data<T>() is
 *        invoked for the first time.
 *
 *      - A tensor may be STORAGE UNINITIALIZED.  A tensor of this form
 *        has non-zero size, but has a storage with a null data pointer.
 *        This situation most frequently arises when a user calls
 *        Resize() or FreeMemory().  This is because Caffe2 historically
 *        does lazy allocation: allocation of data doesn't occur until
 *        mutable_data<T>() is invoked.  A tensor with zero size is
 *        always storage initialized, because no allocation is necessary
 *        in this case.
 *
 *    All combinations of these two uninitialized states are possible.
 *    Consider the following transcript in idiomatic Caffe2 API:
 *
 *      Tensor x(CPU); // x is storage-initialized, dtype-UNINITIALIZED
 *      x.Resize(4); // x is storage-UNINITIALIZED, dtype-UNINITIALIZED
 *      x.mutable_data<float>(); // x is storage-initialized, dtype-initialized
 *      x.FreeMemory(); // x is storage-UNINITIALIZED, dtype-initialized.
 *
 *    All other fields on tensor are always initialized.  In particular,
 *    size is always valid. (Historically, a tensor declared as Tensor x(CPU)
 *    also had uninitialized size, encoded as numel == -1, but we have now
 *    decided to default to zero size, resulting in numel == 0).
 *
 *    Uninitialized storages MUST be uniquely owned, to keep our model
 *    simple.  Thus, we will reject operations which could cause an
 *    uninitialized storage to become shared (or a shared storage to
 *    become uninitialized, e.g., from FreeMemory).
 *
 *    In practice, tensors which are storage-UNINITIALIZED and
 *    dtype-UNINITIALIZED are *extremely* ephemeral: essentially,
 *    after you do a Resize(), you basically always call mutable_data()
 *    immediately afterwards.  Most functions are not designed to
 *    work if given a storage-UNINITIALIZED, dtype-UNINITIALIZED tensor.
 *
 *    We intend to eliminate all uninitialized states, so that every
 *    tensor is fully initialized in all fields.  Please do not write new code
 *    that depends on these uninitialized states.
 */
struct C10_API TensorImpl : public c10::intrusive_ptr_target {
  TensorImpl() = delete;

  /**
   * Construct a 1-dim 0-size tensor backed by the given storage.
   */
  TensorImpl(Storage&& storage, DispatchKeySet);

  /**
   * Construct a 1-dim 0 size tensor that doesn't have a storage.
   */
  TensorImpl(DispatchKeySet, const caffe2::TypeMeta& data_type, c10::optional<c10::Device> device_opt);

  // Legacy constructors so I don't have to go update call sites.
  // TODO: When Variable is added, delete these constructors
  TensorImpl(Storage&& storage, DispatchKey dispatch_key)
    : TensorImpl(std::move(storage), DispatchKeySet(dispatch_key)) {}
  TensorImpl(DispatchKey dispatch_key, const caffe2::TypeMeta& data_type, c10::optional<c10::Device> device_opt)
    : TensorImpl(DispatchKeySet(dispatch_key), data_type, device_opt) {}

 private:
  // This constructor is private, because the data_type is redundant with
  // storage.  Still, we pass it in separately because it's easier to write
  // the initializer list if we're not worried about storage being moved out
  // from under us.
  TensorImpl(Storage&& storage, DispatchKeySet, const caffe2::TypeMeta& data_type, c10::optional<c10::Device>);

 public:
  TensorImpl(const TensorImpl&) = delete;
  TensorImpl& operator=(const TensorImpl&) = delete;
  TensorImpl(TensorImpl&&) = default;
  TensorImpl& operator=(TensorImpl&&) = default;

  /**
   * Release (decref) storage, and any other external allocations.  This
   * override is for `intrusive_ptr_target` and is used to implement weak
   * tensors.
   */
  virtual void release_resources() override;

  /**
   * Return the DispatchKeySet corresponding to this Tensor, specifying
   * all of the DispatchKeys that this Tensor identifies as.  This is the
   * information used to dispatch operations on this tensor.
   */
  DispatchKeySet key_set() const { return key_set_; }

  /**
   * Return a reference to the sizes of this tensor.  This reference remains
   * valid as long as the tensor is live and not resized.
   */
  virtual IntArrayRef sizes() const;

  /**
   * Return a reference to the strides of this tensor.  This reference remains
   * valid as long as the tensor is live and not restrided.
   */
  virtual IntArrayRef strides() const;

  /**
   * Return the number of dimensions of this tensor.  Note that 0-dimension
   * represents a Tensor that is a Scalar, e.g., one that has a single element.
   */
  virtual int64_t dim() const;

  /**
   * True if this tensor has storage. See storage() for details.
   */
  virtual bool has_storage() const;

  /**
   * Return the underlying storage of a Tensor.  Multiple tensors may share
   * a single storage.  A Storage is an impoverished, Tensor-like class
   * which supports far less operations than Tensor.
   *
   * Avoid using this method if possible; try to use only Tensor APIs to perform
   * operations.
   */
  virtual const Storage& storage() const;

  /**
   * The number of elements in a tensor.
   *
   * WARNING: Previously, if you were using the Caffe2 API, you could
   * test numel() == -1 to see if a tensor was uninitialized.  This
   * is no longer true; numel always accurately reports the product
   * of sizes of a tensor.
   */
  virtual int64_t numel() const {
#ifdef DEBUG
    TORCH_INTERNAL_ASSERT(compute_numel() == numel_);
#endif
    return numel_;
  }

  bool unique_version() const {
    return version_counter_.unique();
  }

  /**
   * Whether or not a tensor is laid out in contiguous memory.
   *
   * Tensors with non-trivial strides are not contiguous.  See
   * compute_contiguous() for the exact definition of whether or not
   * a tensor is contiguous or not.
   */
  virtual bool is_contiguous(at::MemoryFormat memory_format=at::MemoryFormat::Contiguous) const;

  bool is_sparse() const {
    // NB: This method is not virtual and avoid dispatches for performance reasons.
    return key_set_.has(DispatchKey::SparseCPUTensorId) ||
           key_set_.has(DispatchKey::SparseCUDATensorId) ||
           key_set_.has(DispatchKey::SparseHIPTensorId);
  }

  bool is_quantized() const {
    // NB: This method is not virtual and avoid dispatches for performance reasons.
    return key_set_.has(DispatchKey::QuantizedCPUTensorId);
  }

  bool is_cuda() const {
    // NB: This method is not virtual and avoid dispatches for performance reasons.
    return key_set_.has(DispatchKey::CUDATensorId) ||
           key_set_.has(DispatchKey::SparseCUDATensorId);
  }

  bool is_hip() const {
    // NB: This method is not virtual and avoid dispatches for performance reasons.
    return key_set_.has(DispatchKey::HIPTensorId) ||
           key_set_.has(DispatchKey::SparseHIPTensorId);
  }

  bool is_mkldnn() const {
    return key_set_.has(DispatchKey::MkldnnCPUTensorId);
  }

  bool is_vulkan() const {
    return key_set_.has(DispatchKey::VulkanTensorId);
  }

  int64_t get_device() const {
    TORCH_CHECK(
        device_opt_.has_value(),
        "tensor does not have a device");
    // See NOTE [c10::optional operator usage in CUDA]
    return (*device_opt_).index();
  }

  Device device() const {
    TORCH_CHECK(
        device_opt_.has_value(),
        "tensor does not have a device");
    // See NOTE [c10::optional operator usage in CUDA]
    return *device_opt_;
  }

  Layout layout() const {
    // NB: This method is not virtual and avoid dispatches for perf.
    if (is_sparse()) {
      return kSparse;
    } else if (is_mkldnn()) {
      return kMkldnn;
    } else {
      return kStrided;
    }
  }

  /**
   * True if a tensor was auto-wrapped from a C++ or Python number.
   * For example, when you write 't + 2', 2 is auto-wrapped into a Tensor
   * with `is_wrapped_number_` set to true.
   *
   * Wrapped numbers do not participate in the result type computation for
   * mixed-type operations if there are any Tensors that are not wrapped
   * numbers.  This is useful, because we want 't + 2' to work with
   * any type of tensor, not just LongTensor (which is what integers
   * in Python represent).
   *
   * Otherwise, they behave like their non-wrapped equivalents.
   * See [Result type computation] in TensorIterator.h.
   *
   * Why did we opt for wrapped numbers, as opposed to just having
   * an extra function add(Tensor, Scalar)?  This helps greatly reduce
   * the amount of code we have to write for add, when actually
   * a Tensor-Scalar addition is really just a Tensor-Tensor
   * addition when the RHS is 0-dim (except for promotion behavior.)
   */
  bool is_wrapped_number() const {
    return is_wrapped_number_;
  }

  /**
   * Set whether or not a tensor was auto-wrapped from a C++ or Python
   * number.  You probably don't want to call this, unless you are
   * writing binding code.
   */
  void set_wrapped_number(bool value) {
    TORCH_INTERNAL_ASSERT(dim() == 0);
    is_wrapped_number_ = value;
  }

  // ~~~~~ Autograd API ~~~~~
  // Some methods below are defined in TensorImpl.cpp because Tensor is an
  // incomplete type.

  /**
   * Set whether or not a tensor requires gradient.
   *
   * It is only valid to call this method on a Variable.
   * See Note [Tensor versus Variable in C++].
   */
  void set_requires_grad(bool requires_grad);

  /**
   * True if a tensor requires gradient.  Tensors which require gradient
   * have history tracked for any operations performed on them, so that
   * we can automatically differentiate back to them.  A tensor that
   * requires gradient and has no history is a "leaf" tensor, which we
   * accumulate gradients into.
   *
   * It is only valid to call this method on a Variable.
   * See Note [Tensor versus Variable in C++].
   */
  bool requires_grad() const;

  /**
   * Return a mutable reference to the gradient.  This is conventionally
   * used as `t.grad() = x` to set a gradient to a completely new tensor.
   *
   * It is only valid to call this method on a Variable.
   * See Note [Tensor versus Variable in C++].
   */
  at::Tensor& grad();

  /**
   * Return the accumulated gradient of a tensor.  This gradient is written
   * into when performing backwards, when this tensor is a leaf tensor.
   *
   * It is only valid to call this method on a Variable.
   * See Note [Tensor versus Variable in C++].
   */
  const at::Tensor& grad() const;

  /**
   * Return a typed data pointer to the actual data which this tensor refers to.
   * This checks that the requested type (from the template parameter) matches
   * the internal type of the tensor.
   *
   * It is invalid to call data() on a dtype-uninitialized tensor, even if
   * the size is 0.
   *
   * WARNING: If a tensor is not contiguous, you MUST use strides when
   * performing index calculations to determine the location of elements in
   * the tensor.  We recommend using 'TensorAccessor' to handle this computation
   * for you; this class is available from 'Tensor'.
   */
  template <typename T>
  inline T * data() const {
    TORCH_CHECK(has_storage(),
        "Cannot access data pointer of Tensor that doesn't have storage");
    TORCH_CHECK(
        storage_initialized(),
        "The tensor has a non-zero number of elements, but its data is not allocated yet. "
        "Caffe2 uses a lazy allocation, so you will need to call "
        "mutable_data() or raw_mutable_data() to actually allocate memory.");
    TORCH_CHECK(
        storage_.IsType<T>(),
        "Tensor type mismatch, caller expects elements to be ",
        caffe2::TypeMeta::TypeName<T>(),
        ", while tensor contains ",
        data_type_.name(),
        ". ");
    // We managed the type check ourselves
    return storage_.unsafe_data<T>() + storage_offset_;
  }

  /**
   * Return a void* data pointer to the actual data which this tensor refers to.
   *
   * It is invalid to call data() on a dtype-uninitialized tensor, even if the
   * size is 0.
   *
   * WARNING: The data pointed to by this tensor may not contiguous; do NOT
   * assume that itemsize() * numel() is sufficient to compute the bytes that
   * can be validly read from this tensor.
   */
  inline void* data() const {
    TORCH_CHECK(has_storage(),
        "Cannot access data pointer of Tensor that doesn't have storage");
    TORCH_CHECK(dtype_initialized(),
        "Cannot access data pointer of Tensor that doesn't have initialized dtype "
        "(e.g., caffe2::Tensor x(CPU), prior to calling mutable_data<T>() on x)");
    return static_cast<void*>(
        static_cast<char*>(storage_.data()) +
        data_type_.itemsize() * storage_offset_);
  }

  /**
   * Like data<T>(), but performs no checks.  You are responsible for ensuring
   * that all invariants required by data() are upheld here.
   */
  template <typename T>
  inline T * unsafe_data() const {
    return storage_.unsafe_data<T>() + storage_offset_;
  }

  /**
   * Returns the TypeMeta of a tensor, which describes what data type
   * it is (e.g., int, float, ...)
   */
  const caffe2::TypeMeta& dtype() const {
    return data_type_;
  }

  /**
   * Return the size of a single element of this tensor in bytes.
   */
  size_t itemsize() const {
    TORCH_CHECK(dtype_initialized(),
        "Cannot report itemsize of Tensor that doesn't have initialized dtype "
        "(e.g., caffe2::Tensor x(CPU), prior to calling mutable_data<T>() on x)");
    return data_type_.itemsize();
  }

  /**
   * Return the offset in number of elements into the storage that this
   * tensor points to.  Most tensors have storage_offset() == 0, but,
   * for example, an index into a tensor will have a non-zero storage_offset().
   *
   * WARNING: This is NOT computed in bytes.
   *
   * XXX: The only thing stopping this function from being virtual is Variable.
   */
  virtual int64_t storage_offset() const {
    return storage_offset_;
  }

  /**
   * True if a tensor has no elements (e.g., numel() == 0).
   */
  inline bool is_empty() const {
    return numel() == 0;
  }

  /**
   * Change the size at some dimension.  This DOES NOT update strides;
   * thus, most changes to size will not preserve contiguity.  You probably
   * also want to call set_stride() when you call this.
   *
   * TODO: This should be jettisoned in favor of `set_sizes_and_strides`,
   * which is harder to misuse.
   */
  virtual void set_size(int64_t dim, int64_t new_size) {
    TORCH_CHECK(allow_tensor_metadata_change(), "set_size ", err_msg_tensor_metadata_change_not_allowed);
    sizes_.at(dim) = new_size;
    refresh_numel();
    refresh_contiguous();
  }

  /**
   * Change the stride at some dimension.
   *
   * TODO: This should be jettisoned in favor of `set_sizes_and_strides`,
   * which is harder to misuse.
   */
  virtual void set_stride(int64_t dim, int64_t new_stride) {
    TORCH_CHECK(allow_tensor_metadata_change(), "set_stride ", err_msg_tensor_metadata_change_not_allowed);
    strides_[dim] = new_stride;
    refresh_contiguous();
  }

  /**
   * Set the offset into the storage of this tensor.
   *
   * WARNING: This does NOT check if the tensor is in bounds for the new
   * location at the storage; the caller is responsible for checking this
   * (and resizing if necessary.)
   */
  virtual void set_storage_offset(int64_t storage_offset) {
    TORCH_CHECK(allow_tensor_metadata_change(), "set_storage_offset ", err_msg_tensor_metadata_change_not_allowed);
    storage_offset_ = storage_offset;
  }

  /**
   * Like set_sizes_and_strides but assumes contiguous strides.
   *
   * WARNING: This function does not check if the requested
   * sizes/strides are in bounds for the storage that is allocated;
   * this is the responsibility of the caller
   */
  void set_sizes_contiguous(IntArrayRef new_size) {
    TORCH_CHECK(allow_tensor_metadata_change(), "set_sizes_contiguous ", err_msg_tensor_metadata_change_not_allowed);
    auto new_dim = new_size.size();

    sizes_.resize(new_dim);
    for (size_t dim = 0; dim < new_dim; ++dim) {
      sizes_[dim] = new_size[dim];
    }

    refresh_numel();
    empty_tensor_restride(MemoryFormat::Contiguous);
  }

  /**
   * Set the sizes and strides of a tensor.
   *
   * WARNING: This function does not check if the requested
   * sizes/strides are in bounds for the storage that is allocated;
   * this is the responsibility of the caller
   */
  void set_sizes_and_strides(IntArrayRef new_size, IntArrayRef new_stride) {
    TORCH_CHECK(allow_tensor_metadata_change(), "set_sizes_and_strides ", err_msg_tensor_metadata_change_not_allowed);
    TORCH_CHECK(
        new_size.size() == new_stride.size(),
        "dimensionality of sizes (",
        new_size.size(),
        ") must match dimensionality of strides (",
        new_stride.size(),
        ")");
    auto new_dim = new_size.size();

    sizes_.resize(new_dim);
    for (size_t dim = 0; dim < new_dim; ++dim) {
      sizes_[dim] = new_size[dim];
    }

    strides_.resize(new_dim);
    if (new_dim > 0) {
      for (size_t dim = new_dim - 1; ; dim--) {
        if (new_stride[dim] >= 0) {
          strides_[dim] = new_stride[dim];
        } else {
          // XXX: This behavior is surprising and may need to be removed to
          // support negative strides. Some pytorch functions rely on it:
          // for example, torch.cat (run TestTorch.test_cat_empty).
          if (dim == new_dim - 1) {
            strides_[dim] = 1;
          } else {
            // Keep stride monotonically increasing to match NumPy.
            strides_[dim] = std::max<int64_t>(sizes_[dim + 1], 1) * strides_[dim + 1];
          }
        }
        if (dim == 0) break;
      }
    }

    refresh_numel();
    refresh_contiguous();
  }

  /**
   * Return the size of a tensor at some dimension.
   */
  virtual int64_t size(int64_t d) const;

  /**
   * Return the stride of a tensor at some dimension.
   */
  virtual int64_t stride(int64_t d) const;

  /**
   * Set whether a tensor allows changes to its metadata (e.g. sizes / strides / storage / storage_offset).
   * See NOTE [ Metadata Change for a Detached Tensor ] for details.
   */
  void set_allow_tensor_metadata_change(bool value) {
    allow_tensor_metadata_change_ = value;
  }

  /**
   * True if a tensor allows changes to its metadata (e.g. sizes / strides / storage / storage_offset).
   * See NOTE [ Metadata Change for a Detached Tensor ] for details.
   */
  bool allow_tensor_metadata_change() const {
    return allow_tensor_metadata_change_;
  }

  /**
   * Set the pointer to autograd metadata.
   */
  void set_autograd_meta(std::unique_ptr<c10::AutogradMetaInterface> autograd_meta);

  /**
   * Return the pointer to autograd metadata.  May return nullptr if the
   * tensor does not track gradients.
   */
  c10::AutogradMetaInterface* autograd_meta() const;

  /**
   * Set the pointer to named tensor metadata.
   */
  void set_named_tensor_meta(std::unique_ptr<c10::NamedTensorMetaInterface> named_tensor_meta) {
    TORCH_WARN_ONCE(
        "Named tensors and all their associated APIs are an experimental feature ",
        "and subject to change. Please do not use them for anything important ",
        "until they are released as stable.");
#ifdef DEBUG
    if (named_tensor_meta) {
      TORCH_INTERNAL_ASSERT(named_tensor_meta->slow_dim() == dim());
    }
#endif
    named_tensor_meta_ = std::move(named_tensor_meta);
  }

  /**
   * Return the pointer to named tensor metadata.
   */
  const c10::NamedTensorMetaInterface* named_tensor_meta() const {
    return named_tensor_meta_.get();
  }

  c10::NamedTensorMetaInterface* named_tensor_meta() {
    return named_tensor_meta_.get();
  }

  bool has_named_tensor_meta() {
    return named_tensor_meta_ != nullptr;
  }


  // NOTE [ TensorImpl Shallow-Copying ]
  //
  // TensorImpl shallow-copying is used when we want to have two Variables share the same tensor metadata
  // (e.g. sizes / strides / storage pointer / storage_offset), but each with a different autograd history.
  // Example call sites:
  //
  // 1. `var_detached = var.detach()` uses `shallow_copy_and_detach()` to create `var_detached` that shares
  // the same tensor metadata with `var`, but with a completely new autograd history.
  // 2. `var.set_data(tensor)` uses `shallow_copy_from()` to copy tensor metadata from
  // `tensor` into `var`, while keeping `var`'s original AutogradMeta.
  //
  // Functions that shallow-copy a TensorImpl (such as `shallow_copy_and_detach()` / `shallow_copy_from()` /
  // `copy_tensor_metadata()`) copy the tensor metadata fields (e.g. sizes / strides / storage pointer /
  // storage_offset) by value. However, the following fields are not copied:
  //
  // 1. the AutogradMeta pointer, because it is unique for each Variable.
  // 2. the version counter, because the destination TensorImpl's version counter is either set to the
  // passed-in `version_counter` (in `shallow_copy_and_detach()` and `copy_tensor_metadata()`), or it is kept
  // intact (in `shallow_copy_from()`). See NOTE [ Version Counter Sharing ] for details.
  //
  // In `shallow_copy_and_detach()` and `copy_tensor_metadata()`, the passed-in `allow_tensor_metadata_change`
  // determines whether the TensorImpl shallow-copy allows changes to its metadata (e.g. sizes / strides /
  // storage / storage_offset). See NOTE [ Metadata Change for a Detached Tensor ] for details.
  //
  // In `shallow_copy_from()`, we don't check the destination TensorImpl's `allow_tensor_metadata_change_`,
  // because `shallow_copy_from()` is used for implementing functions such as `var.set_data(tensor)`, which
  // changes `var`'s tensor metadata and expects its `allow_tensor_metadata_change_` to be ignored.

  /**
   * One TensorImpl can be copied to another TensorImpl if they have the same
   * DispatchKeySet. The only two special cases (for legacy reason) are:
   * CPUTensorId is compatible with CUDATensorId and SparseCPUTensorId is
   * compatible with SparseCUDATensorId.
   */
  inline bool has_compatible_shallow_copy_type(DispatchKeySet from) {
    auto is_dense = [](DispatchKeySet ts) {
      return ts.has(DispatchKey::CPUTensorId) ||
             ts.has(DispatchKey::CUDATensorId) ||
             ts.has(DispatchKey::HIPTensorId);
    };
    auto is_sparse = [](DispatchKeySet ts) {
      return ts.has(DispatchKey::SparseCPUTensorId) ||
             ts.has(DispatchKey::SparseCUDATensorId) ||
             ts.has(DispatchKey::SparseHIPTensorId);
    };
    return (key_set_ == from) || (is_dense(key_set_) && is_dense(from)) || (is_sparse(key_set_) && is_sparse(from));
  }

  /**
   * Return a TensorImpl that is a shallow-copy of this TensorImpl.
   *
   * For usage of `version_counter` and `allow_tensor_metadata_change`,
   * see NOTE [ TensorImpl Shallow-Copying ].
   */
  virtual c10::intrusive_ptr<TensorImpl> shallow_copy_and_detach(
      const c10::VariableVersion& version_counter,
      bool allow_tensor_metadata_change) const {
    auto impl = c10::make_intrusive<TensorImpl>(Storage(storage()), key_set_);
    copy_tensor_metadata(
      /*src_impl=*/this,
      /*dest_impl=*/impl.get(),
      /*version_counter=*/version_counter,
      /*allow_tensor_metadata_change=*/allow_tensor_metadata_change);
    impl->refresh_numel();
    impl->refresh_contiguous();
    return impl;
  }

  /**
   * Shallow-copies data from another TensorImpl into this TensorImpl.
   *
   * For why this function doesn't check this TensorImpl's `allow_tensor_metadata_change_`,
   * see NOTE [ TensorImpl Shallow-Copying ].
   */
  virtual void shallow_copy_from(const c10::intrusive_ptr<TensorImpl>& impl) {
    copy_tensor_metadata(
      /*src_impl=*/impl.get(),
      /*dest_impl=*/this,
      /*version_counter=*/version_counter(),
      /*allow_tensor_metadata_change=*/allow_tensor_metadata_change());
    refresh_numel();
    refresh_contiguous();
  }

  void set_version_counter(
    const c10::VariableVersion& version_counter) noexcept {
    version_counter_ = version_counter;
  }

  const c10::VariableVersion& version_counter() const noexcept {
    return version_counter_;
  }

  void bump_version() noexcept {
    version_counter_.bump();
  }

  inline void set_pyobj(PyObject* pyobj) noexcept {
    pyobj_ = pyobj;
  }

  inline PyObject* pyobj() const noexcept {
    return pyobj_;
  }

 private:
  // See NOTE [c10::optional operator usage in CUDA]
  // We probably don't want to expose this publicly until
  // the note is addressed.
  c10::optional<c10::Device> device_opt() const {
    return device_opt_;
  }

 public:

  /**
   * The device type of a Tensor, e.g., DeviceType::CPU or DeviceType::CUDA.
   */
  DeviceType device_type() const {
    // TODO: A useful internal assert would be to show that device_opt_ is null
    // only if you are an undefined tensor
    TORCH_CHECK(device_opt_.has_value(), "device_type cannot be run on undefined Tensor");
    // See NOTE [c10::optional operator usage in CUDA]
    return (*device_opt_).type();
  }

  /**
   * @brief Extends the outer-most dimension of this tensor by num elements,
   * preserving the existing data.
   *
   * The underlying data may be reallocated in order to accommodate the new
   * elements, in which case this tensors' capacity is grown at a factor of
   * growthPct. This ensures that Extend runs on an amortized O(1) time
   * complexity.
   *
   * This op is auto-asynchronous if the underlying device (CUDA) supports it.
   */
  void Extend(int64_t num, float growthPct) {
    TORCH_CHECK(sizes_.size() >= 1u);
    TORCH_CHECK(num >= 0, "`num` must be non-negative for Extend");
    TORCH_CHECK(
        is_contiguous_,
        "Right now Extend is only supported for contiguous Tensor.");
    auto newDims = sizes_;
    newDims[0] += num;
    if (!storage_.data()) {
      Resize(newDims);
      return;
    }
    auto newNumel = std::accumulate(
        newDims.begin(),
        newDims.end(),
        static_cast<int64_t>(1),
        std::multiplies<int64_t>());
    if (newNumel * storage_.itemsize() <= storage_.capacity()) {
      sizes_ = newDims;
      numel_ = newNumel;
      return;
    }
    auto newCapacity = sizes_;
    newCapacity[0] = std::max<size_t>(
        newDims[0], std::ceil(sizes_[0] * (growthPct + 100) / 100));
    auto oldData = std::move(storage_.data_ptr());
    auto oldSize = numel_;
    auto oldDims = sizes_;
    Resize(newCapacity);
    auto* newData = raw_mutable_data(data_type_);
    if (data_type_.copy()) {
      TORCH_CHECK(
          device_type() == DeviceType::CPU,
          "non-POD types work only on CPU");
      data_type_.copy()(oldData.get(), newData, oldSize);
    } else {
      // The following copy uses the current (thread local) stream for copying
      // and also takes the GPU id from the device() field passed in.
      //
      // TODO: Potentially more enforcements are necessary to avoid accidental
      // switch to sync copy if the currently set device is wrong.
      //
      // Specifically, we might need to switch to a different context device
      // here explicitly to avoid relying on user synchronizing things
      // properly.
      CopyBytes(
          oldSize * itemsize(),
          oldData.get(),
          device(),
          newData,
          device(),
          true); // non-blocking
    }
    reserved_ = true;
    sizes_ = newDims;
    numel_ = newNumel;
  }

  /**
   * @brief Reserve space for the underlying tensor.
   *
   * This must be called after Resize(), since we only specify the first
   * dimension This does not copy over the old data to the newly allocated space
   */
  template <class T>
  void ReserveSpace(const T& outer_dim) {
    TORCH_CHECK(
        is_contiguous_,
        "Right now ReserveSpace is only supported for contiguous Tensor.");
    TORCH_CHECK(
        storage_.unique(), "Can't call ReserveSpace on shared storage.");
    auto newCapacity = sizes_;
    newCapacity[0] = outer_dim;
    auto newNumel = std::accumulate(
        newCapacity.begin(),
        newCapacity.end(),
        static_cast<int64_t>(1),
        std::multiplies<int64_t>());
    if (newNumel * storage_.itemsize() <= storage_.capacity()) {
      return;
    }
    // Old data is discarded
    storage_.data_ptr().clear();
    auto oldSize = numel_;
    auto oldDims = sizes_;
    Resize(newCapacity);
    // Allocate new memory but don't copy over the data
    raw_mutable_data(data_type_);
    sizes_ = oldDims;
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
   *
   * This method respects caffe2_keep_on_shrink.  Consult the internal logic
   * of this method to see exactly under what circumstances this flag matters.
   */
  template <typename... Ts>
  void Resize(Ts... dim_source) {
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
        reset_tensor = storage_.capacity() <
                (storage_offset_ + numel_) * storage_.itemsize() ||
            !FLAGS_caffe2_keep_on_shrink ||
            storage_.capacity() -
                    (storage_offset_ + numel_) * storage_.itemsize() >
                static_cast<size_t>(FLAGS_caffe2_max_keep_on_shrink_memory);
      }

      if (reset_tensor && storage_initialized()) {
        FreeMemory();
      }
    }
  }

  /**
   * Resizes the tensor without touching underlying storage.
   * This requires the total size of the tensor to remains constant.
   */
  inline void Reshape(const std::vector<int64_t>& dims) {
    TORCH_CHECK(
        is_contiguous_,
        "Right now Reshape is only supported for contiguous Tensor.");
    int64_t new_size = 1;
    for (auto d : dims) {
      TORCH_CHECK(d >= 0);
      new_size *= d;
    }
    TORCH_CHECK(
        new_size == numel_,
        "New size and old size are not equal. You cannot use Reshape, "
        "but should use Resize."
        // TODO(jiayq): remove the following warning after pending diffs
        // stabilize.
        " The old caffe2 mixes Reshape and Resize but this behavior has "
        "been changed. If you find this error, most likely you will need "
        "to change corresponding code from Reshape to Resize.");
    sizes_ = dims;
    empty_tensor_restride(MemoryFormat::Contiguous);
  }

  /**
   * Release whatever memory the tensor was holding but keep size and type
   * information. Subsequent call to mutable_data will trigger new memory
   * allocation.
   */
  inline void FreeMemory() {
    // We'll detach from the old Storage and create a new one
    storage_ = Storage::create_legacy(storage_.device(), data_type_);
    storage_offset_ = 0;
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
  // To be deprecated
  void ShareData(const TensorImpl& src) {
    // Right now, we are assuming the device_type are the same, since it is
    // inherently the same in the non-templatized code. We should probably add
    // an assert here which might affect perf a little bit.
    TORCH_CHECK(
        src.numel_ == numel_,
        "Size mismatch - did you call reshape before sharing the data?");
    // It is possible that the source tensor hasn't called mutable_data() yet,
    // in which case ShareData() doesn't make much sense since we don't really
    // know what to share yet.
    // TODO: Add the assert after all uninitialized states are eliminated
    // TORCH_CHECK(src.dtype_initialized(),
    //            "Source tensor don't have a data type (did you call mutable_data<T> on the tensor?)");
    if (!src.dtype_initialized()) {
      C10_LOG_EVERY_MS(WARNING, 1000) <<
                   "Source tensor don't have a data type (did you call mutable_data<T> on the tensor?)";
    }
    TORCH_CHECK(
        src.storage_initialized(),
        "Source tensor has no content and has size > 0");
    // Finally, do sharing.
    /* Since we create new Storage whenever we need to change data_type/capacity
     * this still keeps the original semantics
     */
    storage_ = src.storage();
    data_type_ = src.dtype();
    device_opt_ = src.device_opt();
    storage_offset_ = src.storage_offset();
  }

  void ShareExternalPointer(
      DataPtr&& data_ptr,
      const caffe2::TypeMeta& data_type,
      size_t capacity) {
    TORCH_CHECK(
        data_type.id() != caffe2::TypeIdentifier::uninitialized(),
        "To share with a raw external pointer you need to pass in an "
        "initialized data_type(TypeMeta).");
    if (!capacity) {
      capacity = numel_ * data_type.itemsize();
    }
    if (storage_.unique()) {
      storage_.UniqueStorageShareExternalPointer(
          std::move(data_ptr), data_type, capacity);
      data_type_ = data_type;
      device_opt_ = storage_.device();
      storage_offset_ = 0;
    } else {
      int64_t numel = capacity / data_type.itemsize();
      // Create a new Storage
      storage_ = Storage(
          data_type,
          numel,
          std::move(data_ptr),
          /*allocator=*/nullptr,
          /*resizable=*/false);
      data_type_ = data_type;
      device_opt_ = storage_.device();
      storage_offset_ = 0;
    }
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
  inline void* raw_mutable_data(const caffe2::TypeMeta& meta) {
    // For 0-size tensors it's fine to return any pointer (including nullptr)
    if (data_type_ == meta && storage_initialized()) {
      return static_cast<void*>(static_cast<char*>(storage_.data()) + storage_offset_ * meta.itemsize());
    } else {
      bool had_special_dtor = data_type_.placementDelete() != nullptr;
      storage_offset_ = 0;
      if (storage_.unique()) {
        storage_.set_dtype(meta);
      } else {
        if (data_type_ != meta) {
          storage_ = Storage::create_legacy(storage_.device(), meta);
        }
      }
      data_type_ = meta;
      // NB: device is not changed

      // We can reuse the existing buffer if the current data does not have
      // a special destructor and the new data doesn't have a special
      // constructor.
      if (numel_ == 0 ||
          (meta.placementNew() == nullptr && !had_special_dtor &&
           storage_.numel() >= numel_)) {
        TORCH_INTERNAL_ASSERT(storage_offset_ == 0); // because we just reallocated
        return storage_.data();
      }
      const Allocator* allocator = storage_.allocator();
      // Storage might have nullptr allocator in rare cases, for example, if
      // an external memory segment has been wrapped with Tensor and we don't
      // know how to reallocate it. However, in order to preserve legacy C2
      // behavior, we allow reallocating the memory using default allocator.
      if (allocator == nullptr) {
        allocator = GetAllocator(storage_.device_type());
      }
      if (meta.placementNew()) {
        // For types that need placement new, we will call it, as well as
        // making sure that when the data is freed, it calls the right
        // destruction procedure.
        auto size = numel_;
        auto dtor = data_type_.placementDelete();
        auto data_ptr = allocator->allocate(numel_ * storage_.itemsize());
        storage_.set_data_ptr(PlacementDeleteContext::makeDataPtr(
            std::move(data_ptr), dtor, size, storage_.device()));
        data_type_.placementNew()(storage_.data(), numel_);
      } else {
        // For fundamental type, new and delete is easier.
        storage_.set_data_ptr(
            allocator->allocate(numel_ * storage_.itemsize()));
      }
      storage_.set_numel(numel_);
      TORCH_INTERNAL_ASSERT(storage_offset_ == 0); // because we just reallocated
      device_opt_ = storage_.device();
      return storage_.data();
    }
  }

  /**
   * Returns a typed pointer of the underlying storage.
   *
   * For fundamental types, we reuse possible existing storage if there
   * is sufficient capacity.
   */
  template <typename T>
  inline T* mutable_data() {
    if (storage_initialized() && storage_.IsType<T>()) {
      return static_cast<T*>(storage_.data()) + storage_offset_;
    }
    // Check it here statically - otherwise TypeMeta would throw the runtime
    // error in attempt to invoke TypeMeta::ctor()
    static_assert(
        std::is_default_constructible<T>::value,
        "Tensor can't hold non-default-constructible types");
    return static_cast<T*>(raw_mutable_data(caffe2::TypeMeta::Make<T>()));
  }

  /**
   * True if a tensor is storage initialized.  A tensor may become
   * storage UNINITIALIZED after a Resize() or FreeMemory()
   */
  bool storage_initialized() const {
    TORCH_CHECK(has_storage(), "cannot call storage_initialized on tensor that does not have storage");
    return storage_.data() || numel_ == 0;
  }

  /**
   * True if a tensor is dtype initialized.  A tensor allocated with
   * Caffe2-style constructors is dtype uninitialized until the
   * first time mutable_data<T>() is called.
   */
  bool dtype_initialized() const noexcept {
    return data_type_ != caffe2::TypeMeta();
  }

  void set_storage(at::Storage storage) {
    TORCH_CHECK(allow_tensor_metadata_change(), "set_storage ", err_msg_tensor_metadata_change_not_allowed);
    storage_ = std::move(storage);
    data_type_ = storage_.dtype();
    device_opt_ = storage_.device();
  }

  /**
   * Set the strides of the tensor to match memory_format
   *
   * WARNING: This function doesn't rearrange data and assumes tensor is a memory
   * contiguous
   */
  virtual void empty_tensor_restride(MemoryFormat memory_format) {
    #ifdef DEBUG
        TORCH_INTERNAL_ASSERT(compute_numel() == numel_,
        "If you are seeing this error, that means empty_tensor_restride was "
        "called before setting correct numel");
    #endif
    switch (memory_format) {
      case MemoryFormat::Contiguous: {
        // dim_ is a virtual call, don't repeat it
        auto dim_ = dim();
        strides_.resize(dim_);
        if (dim_ > 0) {
          int last_idx = dim_ - 1;
          strides_[last_idx] = 1;
          for (auto i = last_idx - 1; i >= 0; --i) {
            strides_[i] = strides_[i + 1] * std::max<int64_t>(sizes_[i + 1], 1);
          }
        }
        break;
      }
      case MemoryFormat::ChannelsLast: {
        TORCH_CHECK(
            dim() == 4,
            "required rank 4 tensor to use channels_last format");
        set_sizes_and_strides(sizes(), get_channels_last_strides_2d(sizes()));
        break;
      }
      case MemoryFormat::ChannelsLast3d: {
        TORCH_CHECK(
            dim() == 5,
            "required rank 5 tensor to use channels_last_3d format");
        set_sizes_and_strides(sizes(), get_channels_last_strides_3d(sizes()));
        break;
      }
      case MemoryFormat::Preserve:
        TORCH_CHECK(false, "unsupported memory format ", memory_format);
        // Cleaning warning messages, no need to break as TORCH_CHECK(false)
        // terminates flow.
        // break;
    }
    // recompute contiguous flag, as currently NHWC/NCHW flags are not mutually
    // exclusive see #24090
    refresh_contiguous();
  }

  bool is_strides_like_channels_last() const {
    return is_channels_last_;
  }

  bool is_strides_like_channels_last_3d() const {
    return is_channels_last_3d_;
  }

  bool is_non_overlapping_and_dense() const {
    return is_non_overlapping_and_dense_;
  }

private:

  // The Caffe2 Resize() method supports being called both as Resize({2,2}) as
  // well as variadic with Resize(2, 2).  These overloads provide all of the
  // supported calling configurations, while being overloads (and not templates)
  // so that implicit conversions still work.
  //
  // SetDims on ArrayRef is internally implemented as a template, so we can
  // handle both ArrayRefs of different types (there are some uses of
  // Resize in Caffe2 which pass in int, not int64_t.)

  template <
      typename T,
      typename = typename std::enable_if<std::is_integral<T>::value>::type>
  bool SetDimsTemplate(ArrayRef<T> src) {
    auto old_numel = numel_;
    sizes_.resize(src.size());
    int64_t new_numel = 1;
    for (size_t i = 0; i < src.size(); ++i) {
      new_numel *= src[i];
      sizes_[i] = src[i];
    }
    numel_ = new_numel;
    empty_tensor_restride(MemoryFormat::Contiguous);
    return numel_ != old_numel;
  }

  bool SetDims(ArrayRef<int64_t> s) {
    return SetDimsTemplate(s);
  }

  bool SetDims(ArrayRef<int> s) {
    return SetDimsTemplate(s);
  }

  bool SetDims(ArrayRef<size_t> s) {
    return SetDimsTemplate(s);
  }

  bool SetDims() {
    return SetDims(IntArrayRef{});
  }

  bool SetDims(const int64_t d0) {
    return SetDims(IntArrayRef{d0});
  }

  bool SetDims(const int64_t d0, const int64_t d1) {
    return SetDims(IntArrayRef{d0, d1});
  }

  bool SetDims(const int64_t d0, const int64_t d1, const int64_t d2) {
    return SetDims(IntArrayRef{d0, d1, d2});
  }

  bool SetDims(const int64_t d0, const int64_t d1, const int64_t d2, const int64_t d3) {
    return SetDims(IntArrayRef{d0, d1, d2, d3});
  }

  /**
   * Compute the number of elements based on the sizes of a tensor.
   */
  int64_t compute_numel() const {
    int64_t n = 1;
    for (auto s : sizes()) {
      n *= s;
    }
    return n;
  }

  /**
   * Compute whether or not a tensor is contiguous based on the sizes and
   * strides of a tensor.
   */
  bool compute_contiguous() const;

  bool compute_channels_last_contiguous_2d() const;

  bool compute_channels_last_contiguous_3d() const;

  bool compute_strides_like_channels_last_2d() const;

  bool compute_strides_like_channels_last_3d() const;

  bool compute_non_overlapping_and_dense() const;

protected:
  /**
   * Recompute the cached numel of a tensor.  Call this if you modify sizes.
   */
  void refresh_numel() {
    numel_ = compute_numel();
  }

  /**
   * Recompute the cached contiguity of a tensor.  Call this if you modify sizes
   * or strides.
   */
  void refresh_contiguous() {
    is_contiguous_ = compute_contiguous();
    // Note:
    // Dim 0, 1, 2 will never be a channels last 2d/3d format
    // Dim 3+ is possibly be a channels last 2d format (Dim 4 only at this point)
    // Dim 4+ is possibly be a channels last 3d format (Dim 5 only at this point)
    switch (dim()) {
      case 4:
        is_channels_last_contiguous_ = compute_channels_last_contiguous_2d();
        is_channels_last_3d_contiguous_ = false;
        is_channels_last_ = compute_strides_like_channels_last_2d();
        is_channels_last_3d_ = false;
        is_non_overlapping_and_dense_ = is_contiguous_ || is_channels_last_contiguous_ || compute_non_overlapping_and_dense();
        break;
      case 5:
        is_channels_last_contiguous_ = compute_channels_last_contiguous_2d();
        is_channels_last_3d_contiguous_ = !is_channels_last_contiguous_ && compute_channels_last_contiguous_3d();
        is_channels_last_ = !is_channels_last_3d_contiguous_ && compute_strides_like_channels_last_2d();
        is_channels_last_3d_ = !is_channels_last_ && compute_strides_like_channels_last_3d();
        is_non_overlapping_and_dense_ = is_contiguous_ || is_channels_last_contiguous_ || is_channels_last_3d_contiguous_|| compute_non_overlapping_and_dense();
        break;
      default:
        is_channels_last_contiguous_ = false;
        is_channels_last_3d_contiguous_ = false;
        // is_channels_last_ and is_channels_last_3d_ are suggested memory_format.
        // Being channels_last_contiguous doesn't necessarily mean the tensor is
        // strided like channels_last: for strides on channel dimension could suggest
        // desired memory_layout, but it doesn't affect memory storage
        is_channels_last_ = false;
        is_channels_last_3d_ = false;
        is_non_overlapping_and_dense_ = is_contiguous_ || compute_non_overlapping_and_dense();
    }
  }

  /**
   * Copy the tensor metadata fields (e.g. sizes / strides / storage pointer / storage_offset)
   * from one TensorImpl to another TensorImpl.
   *
   * For usage of `version_counter` and `allow_tensor_metadata_change`, see NOTE [ TensorImpl Shallow-Copying ].
   */
  static void copy_tensor_metadata(
      const TensorImpl* src_impl,
      TensorImpl* dest_impl,
      const c10::VariableVersion& version_counter,
      bool allow_tensor_metadata_change);

protected:
  // Error message to show when the user tries to change tensor metadata on
  // Tensor created from .data or .detach().
  //
  // See NOTE [ Metadata Change for a Detached Tensor ] for details.
  static const char * const err_msg_tensor_metadata_change_not_allowed;

  Storage storage_;

private:
  // This pointer points to an AutogradMeta struct that stores autograd-specific fields
  // (such as grad_ / grad_fn_ / grad_accumulator_).
  // This pointer always has unique ownership (meaning only one TensorImpl can own it
  // at a time).
  //
  // autograd_meta_ can be nullptr, as an optimization.  When this occurs, it is
  // equivalent to having an autograd_meta_ pointing to a default constructed
  // AutogradMeta; intuitively, tensors which don't require grad will have this
  // field set to null.
  //
  // This means accessors on autograd_meta_ have to be careful to test if they
  // got a nullptr, and handle default behavior appropriately in that case.
  //
  // Note that we don't enforce the invariant that if the AutogradMeta is
  // default constructed, it is nullptr (to do this, we'd have to continuously
  // check if an AutogradMeta became, by mutation, equal to the default
  // constructed form.  (This might be useful, but it seems rare enough that
  // a requires_grad=True variable will turn back into the requires_grad=False
  // version.)  So there are three representable states:
  //
  //    1. autograd_meta_ == nullptr
  //    2. autograd_meta_ is default constructed (semantically, same as (1))
  //    3. autograd_meta_ has nontrivial information content
  //
  std::unique_ptr<c10::AutogradMetaInterface> autograd_meta_ = nullptr;

protected:
  std::unique_ptr<c10::NamedTensorMetaInterface> named_tensor_meta_ = nullptr;

  c10::VariableVersion version_counter_;

  // This field contains a weak reference to a PyObject representing
  // this Tensor.  It MUST NOT be a strong reference, as that would
  // create a reference cycle between Tensor and the PyObject.  If
  // pyobj is nullptr, when we transfer Tensor to Python, we allocate
  // a new PyObject for it and set this field.  This is thread safe
  // because all Python code is protected under the GIL.  This design does
  // NOT WORK for Tensors which are shared across multiple Python
  // subinterpreters (introduced in Python 3.8) since you don't have
  // enough space to store the separate PyObject per subinterpreter.
  // When a PyObject dies, you are obligated to clear this field
  // (otherwise, you will try to use-after-free the pyobj); this currently
  // occurs in THPVariable_clear in torch/csrc/autograd/python_variable.cpp
  PyObject* pyobj_ = nullptr;

  // We could save a word or two by combining the SmallVector structs,
  // since their size is redundant, and if we need to overflow the buffer space
  // we could keep the two pointers together. However, that would require
  // implementing another struct from scratch, so only do this if we're desperate.
  SmallVector<int64_t,5> sizes_;
  SmallVector<int64_t,5> strides_;

  int64_t storage_offset_ = 0;
  // If sizes and strides are empty, the numel is 1!!  However, most of the
  // time, we will immediately set sizes to {0} and reset numel to 0.
  // (Can't do that in the default initializers, because there's no way to
  // spell "allocate a one-element array" for strides_).
  int64_t numel_ = 1;

  // INVARIANT: When storage is non-null, this type meta must
  // agree with the type meta in storage
  caffe2::TypeMeta data_type_;

  // NOTE [c10::optional operator usage in CUDA]
  // Our optional definition doesn't compile in .cu file if `value()` or
  // `operator->` are used.  Instead, we always use `operator*`.
  // See https://github.com/pytorch/pytorch/issues/18496 for more info.
  // If this is too burdensome to maintain, we can just
  // manually implement this with an additional bool.

  // INVARIANT: When storage is non-null, this Device must
  // agree with the type meta in storage.
  //
  // INVARIANT: device_opt_ is only nullopt for undefined tensors
  // (which do not have a device.)
  c10::optional<c10::Device> device_opt_;

  // The set of DispatchKeys which describe this tensor.  NB: this
  // does NOT include VariableTensorId (historically, it did, but
  // not anymore!)
  DispatchKeySet key_set_;

  // You get to have eight byte-size fields here, before you
  // should pack this into a bitfield.
  bool is_contiguous_ = true;

  // Tensor is stored in the channels last 2d memory format, when dimensions
  // order is (N)CHW and C-strides < W-strides < H-strides (< N-strides)
  // (If size of any dimension is equal to 1, this dimension strides value
  // is not taken into account).
  bool is_channels_last_ = false;

  // Channels last contiguous tensor is channel last tensor which occupies
  // contiguous memory block.
  bool is_channels_last_contiguous_ = false;

  // Tensor is stored in the channels last 3d memory format, when dimensions
  // order is (N)CDHW and C-strides < W-strides < H-strides < D - strides (< N-strides)
  // (If size of any dimension is equal to 1, this dimension strides value
  // is not taken into account).
  bool is_channels_last_3d_ = false;

  // Channels last 3d contiguous tensor is channel last 3d tensor which occupies
  // contiguous memory block.
  bool is_channels_last_3d_contiguous_ = false;

  // Dense tensor is the tensor that store values in a contiguous block of memory.
  // Non-overlapping tensor is the tensor in which elements occupy individual
  // non-repetitive memory.
  bool is_non_overlapping_and_dense_ = false;

  bool is_wrapped_number_ = false;

  // NOTE [ Metadata Change for a Detached Tensor ]
  //
  // Normally, a user is allowed to change the tensor metadata
  // (e.g. sizes / strides / storage / storage_offset) of a tensor.
  // However, if the tensor is created by `t1_detached = t1.data` in Python
  // or `t1_detached = t1.detach()` in Python/C++, those changes to the
  // tensor metadata of `t1_detached` will not be propagated back to the
  // original tensor `t1`. In order to make such changes explicitly illegal,
  // we created the `allow_tensor_metadata_change_` flag, to prevent users
  // from changing metadata of the detached tensor and expecting the original
  // tensor to also be updated.
  //
  // NOTE: For a full list of tensor metadata fields, please see
  // `copy_tensor_metadata()` in TensorImpl and its subclasses to find
  // which fields are copied by value.
  bool allow_tensor_metadata_change_ = true;

  // we decide to keep reserved_ and it will
  // live in Tensor after the split
  // The logic is that if Extend() or ReserveSpace() were ever called,
  // then subsequent Resize()s will not free up Storage.
  bool reserved_ = false;

};

// Note [TensorImpl size constraints]
// ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
// Changed the size of TensorImpl?  If the size went down, good for
// you!  Adjust the documentation below and the expected size.
// Did it go up?  Read on...
//
// Struct size matters.  In some production systems at Facebook, we have
// 400M live tensors during a training run.  Do the math: every 64-bit
// word you add to Tensor is an extra 3.2 gigabytes in RAM.
//
// If you are a Facebook employee, you can check if the run in question
// has tipped you over the point using the command here:
// https://fburl.com/q5enpv98
//
// For reference, we OOMed at 160 bytes (20 words) per TensorImpl.
// This is not counting overhead from strides out-of-line allocation and
// StorageImpl space and this is from before we inlined sizes and strides
// directly into TensorImpl as SmallVectors.
//
// Our memory usage on 32-bit systems is suboptimal, but we're not checking
// for it at the moment (to help avoid rage inducing cycles when the
// 32-bit number is wrong).
//
// Current breakdown:
//
//    vtable pointer
//    strong refcount           TODO: pack these into one word
//    weak refcount
//    storage pointer
//    autograd metadata pointer
//    version counter pointer
//    PyObject pointer
//    sizes SmallVector (begin)
//    sizes SmallVector (end)
//    sizes SmallVector (capacity)
//    sizes SmallVector (pre-allocated 0)
//    sizes SmallVector (pre-allocated 1)
//    sizes SmallVector (pre-allocated 2)
//    sizes SmallVector (pre-allocated 3)
//    sizes SmallVector (pre-allocated 4)
//    strides SmallVector (begin)
//    strides SmallVector (end)
//    strides SmallVector (capacity)
//    strides SmallVector (pre-allocated 0)
//    strides SmallVector (pre-allocated 1)
//    strides SmallVector (pre-allocated 2)
//    strides SmallVector (pre-allocated 3)
//    strides SmallVector (pre-allocated 4)
//    storage offset
//    numel
//    data type pointer
//    (optional) device
//    tensor type id
//    miscellaneous bitfield
//
static_assert(sizeof(void*) != sizeof(int64_t) || // if 64-bit...
              sizeof(TensorImpl) == sizeof(int64_t) * 31,
              "You changed the size of TensorImpl on 64-bit arch."
              "See Note [TensorImpl size constraints] on how to proceed.");
} // namespace c10
