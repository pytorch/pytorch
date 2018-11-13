#pragma once

#include <atomic>
#include <memory>

#include <ATen/core/Backend.h>
#include <ATen/core/LegacyTypeDispatch.h>
#include <ATen/core/Storage.h>
#include <ATen/core/TensorOptions.h>
#include <ATen/core/TensorTypeId.h>
#include <ATen/core/TensorTypeIdRegistration.h>
#include <ATen/core/context_base.h>

#include <c10/util/Exception.h>
#include "c10/util/Optional.h"

#include "c10/util/Flags.h"

#include "caffe2/core/allocator.h"
#include "caffe2/core/common.h"
#include "caffe2/core/logging.h"

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

namespace caffe2 {

// Defined by protobuf
class DeviceOption;

}

namespace at {
class Scalar;
struct Type;
struct Storage;
class Tensor;

/**
 * A utility function to convert vector<int> to vector<int64_t>.
 */
inline std::vector<int64_t> ToVectorint64_t(ArrayRef<int> src) {
  return std::vector<int64_t>(src.begin(), src.end());
}

/**
 * Return product of all dimensions starting from k
 */
inline int64_t size_from_dim_(int k, IntList dims) {
  int64_t r = 1;
  for (size_t i = k; i < dims.size(); ++i) {
    r *= dims[i];
  }
  return r;
}

// Product of all dims up to k (not including dims[k])
inline int64_t size_to_dim_(int k, IntList dims) {
  AT_ASSERT((unsigned)k <= dims.size());
  int64_t r = 1;
  for (int i = 0; i < k; ++i) {
    r *= dims[i];
  }
  return r;
}

// Product of all dims between k and l (not including dims[k] and dims[l])
inline int64_t size_between_dim_(int k, int l, IntList dims) {
  AT_ASSERT((unsigned)l < dims.size());
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
  AT_ASSERT(axis_index >= -ndims);
  AT_ASSERT(axis_index < ndims);
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
struct CAFFE2_API PlacementDeleteContext {
  at::DataPtr data_ptr_;
  PlacementDtor placement_dtor_;
  size_t size_;
  PlacementDeleteContext(
      at::DataPtr&& data_ptr,
      PlacementDtor placement_dtor,
      size_t size)
      : data_ptr_(std::move(data_ptr)),
        placement_dtor_(placement_dtor),
        size_(size) {}
  static at::DataPtr makeDataPtr(
      at::DataPtr&& data_ptr,
      PlacementDtor placement_dtor,
      size_t size,
      at::Device device);
  ~PlacementDeleteContext() {
    placement_dtor_(data_ptr_.get(), size_);
    // original memory will be freed when data_ptr_ is destructed
  }
};

namespace detail {
  // This is intended to be a centralized location by which we can determine
  // what an appropriate TensorTypeId for a tensor is.
  //
  // This takes a TensorOptions, rather than just a DeviceType and Layout, because
  // we reserve the right to change dispatch based on *any* aspect of
  // TensorOptions.  WARNING: If you do this, you need to fix the calls
  // to computeTensorTypeId in caffe2/tensor.h
  inline TensorTypeId computeTensorTypeId(TensorOptions options) {
    switch (options.layout()) {
      case Layout::Strided:
        switch (options.device().type()) {
          case DeviceType::CPU:
            return CPUTensorId();
          case DeviceType::CUDA:
            return CUDATensorId();
          case DeviceType::MKLDNN:
            return MKLDNNTensorId();
          case DeviceType::OPENGL:
            return OpenGLTensorId();
          case DeviceType::OPENCL:
            return OpenCLTensorId();
          case DeviceType::IDEEP:
            return IDEEPTensorId();
          case DeviceType::HIP:
            return HIPTensorId();
          default:
            AT_ERROR("Unsupported device type for dense layout: ", options.device().type());
        }
      case Layout::Sparse:
        switch (options.device().type()) {
          case DeviceType::CPU:
            return SparseCPUTensorId();
          case DeviceType::CUDA:
            return SparseCUDATensorId();
          default:
            AT_ERROR("Unsupported device type for sparse layout: ", options.device().type());
        }
      default:
        AT_ERROR("Unsupported layout: ", options.layout());
    }
  }

  inline DeviceType computeDeviceType(TensorTypeId tid) {
    if (tid == CPUTensorId()) {
      return DeviceType::CPU;
    } else if (tid == CUDATensorId()) {
      return DeviceType::CUDA;
    } else if (tid == MKLDNNTensorId()) {
      return DeviceType::MKLDNN;
    } else if (tid == OpenGLTensorId()) {
      return DeviceType::IDEEP;
    } else if (tid == OpenCLTensorId()) {
      return DeviceType::OPENCL;
    } else if (tid == IDEEPTensorId()) {
      return DeviceType::IDEEP;
    } else if (tid == HIPTensorId()) {
      return DeviceType::HIP;
    } else if (tid == SparseCPUTensorId()) {
      return DeviceType::CPU;
    } else if (tid == SparseCUDATensorId()) {
      return DeviceType::CUDA;
    } else {
      AT_ASSERTM(false, "Unknown TensorTypeId: ", tid);
    }
  }
} // namespace detail

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
struct CAFFE2_API TensorImpl : public c10::intrusive_ptr_target {
  TensorImpl() = delete;

  /**
   * Construct a 1-dim 0-size tensor with the given settings.
   * The provided allocator will be used to allocate data on
   * subsequent resize.
   */
  TensorImpl(TensorTypeId type_id, const caffe2::TypeMeta& data_type, Allocator *allocator, bool is_variable);

  /**
   * Construct a 1-dim 0-size tensor backed by the given storage.
   */
  TensorImpl(Storage&& storage, TensorTypeId type_id, bool is_variable);

 private:
  // This constructor is private, because the data_type is redundant with
  // storage.  Still, we pass it in separately because it's easier to write
  // the initializer list if we're not worried about storage being moved out
  // from under us.
  TensorImpl(Storage&& storage, TensorTypeId type_id, const caffe2::TypeMeta& data_type, bool is_variable);

 public:
  TensorImpl(const TensorImpl&) = default;
  TensorImpl& operator=(const TensorImpl&) = default;
  TensorImpl(TensorImpl&&) = default;
  TensorImpl& operator=(TensorImpl&&) = default;

  /**
   * Release (decref) storage, and any other external allocations.  This
   * override is for `intrusive_ptr_target` and is used to implement weak
   * tensors.
   */
  virtual void release_resources() override;

  // TODO: Ideally, type_id() would be the *only* key we need to consult
  // to do a dispatch, instead of having to grovel through three different
  // variables.  Here's what's standing in the way:
  //
  //  - To eliminate ScalarType, we have to allocate a TensorTypeId for
  //    each ScalarType+Backend combination, and then set it appropriately
  //    when we initially allocate a TensorImpl.
  //
  //  - To eliminate is_variable, we have to allocate two classes of
  //    TensorTypeId: ones that are variables, and ones that are not.
  //    We may not want to eliminate this in the short term, because
  //    hard-coding variable status into type_id() makes it more difficult
  //    to do the "thread-local no_grad" trick (where we process Variables
  //    "as if" they were non-Variables by setting a thread local variable.)
  //
  // TODO: type() is a very attractive name for a method, but we don't
  // actually want people to use it.  Rename this to something else.

  /**
   * Return the Type object corresponding to this Tensor, which we can
   * use to do dynamic dispatch to operators from.  This method is NOT
   * intended to be used by end-users; it is purely an implementation
   * detail.
   */
  Type & type() const {
    // NB: It's valid to use getTypeRaw here, because the TensorImpl
    // could not have been created without initializing the Type first.
    // TODO: This is not actually true via the Caffe2 codepath!  Make
    // it so.
    return *globalLegacyTypeDispatch().getTypeRaw(tensorTypeIdToBackend(type_id()), typeMetaToScalarType(dtype()), is_variable());
  }

  /**
   * Return the TensorTypeId corresponding to this Tensor.  In the future,
   * this will be the sole piece of information required to dispatch
   * to an operator; however, at the moment, it is not used for
   * dispatch.
   *
   * type_id() and type() are NOT in one-to-one correspondence; we only
   * have a single type_id() for CPU tensors, but many Types (CPUFloatTensor,
   * CPUDoubleTensor...)
   */
  TensorTypeId type_id() const { return type_id_; }

  /**
   * Return a reference to the sizes of this tensor.  This reference remains
   * valid as long as the tensor is live and not resized.
   */
  virtual IntList sizes() const;

  /**
   * Return a reference to the strides of this tensor.  This reference remains
   * valid as long as the tensor is live and not restrided.
   */
  virtual IntList strides() const;

  /**
   * Return the number of dimensions of this tensor.  Note that 0-dimension
   * represents a Tensor that is a Scalar, e.g., one that has a single element.
   */
  virtual int64_t dim() const;

  /**
   * Return the underyling storage of a Tensor.  Multiple tensors may share
   * a single storage.  A Storage is an impoverished, Tensor-like class
   * which supports far less operations than Tensor.
   *
   * Avoid using this method if possible; try to use only Tensor APIs to perform
   * operations.
   */
  virtual const Storage& storage() const;

  // TODO: Delete me.
  friend struct Type;

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
    AT_ASSERT(compute_numel() == numel_);
#endif
    return numel_;
  }

  /**
   * Whether or not a tensor is laid out in contiguous memory.
   *
   * Tensors with non-trivial strides are not contiguous.  See
   * compute_contiguous() for the exact definition of whether or not
   * a tensor is contiguous or not.
   */
  virtual bool is_contiguous() const {
#ifdef DEBUG
    AT_ASSERT(compute_contiguous() == is_contiguous_);
#endif
    return is_contiguous_;
  }

  bool is_sparse() const {
    // NB: This method is not virtual and avoid dispatches for performance reasons.
    auto tid = type_id();
    // NB: At the moment, variables have the same TensorTypeId as their
    // corresponding tensor, but if this ever changes, we need to modify this.
    return tid == SparseCPUTensorId() || tid == SparseCUDATensorId();
  }

  bool is_cuda() const {
    // NB: This method is not virtual and avoid dispatches for performance reasons.
    auto tid = type_id();
    // NB: At the moment, variables have the same TensorTypeId as their
    // corresponding tensor, but if this ever changes, we need to modify this.
    return tid == CUDATensorId() || tid == SparseCUDATensorId();
  }

  int64_t get_device() const {
    // NB: This method is not virtual and tries to avoid dispatches in the common case for perf.
    const auto tid = type_id();
    if (tid == CUDATensorId()) {
      // TODO: #12934 investigate caching device on TensorImpl to avoid this vdispatch.
      return storage().device().index();
    }
    return get_device_slow();
  }

  Device device() const {
    // Special case the common case for performance reasons
    // TODO: This is a little convoluted so it would be good to investigate
    // caching device on TensorImpl (#12934) to speed up device() calls in all cases.
    const auto tid = type_id();
    if (tid == CPUTensorId() || tid == CUDATensorId()) {
      // NB: storage(), not storage_, b/c of Variable.
      const auto& mystorage = storage();
      if (mystorage) {
        return mystorage.device();
      }
    }
    const auto device_type = detail::computeDeviceType(tid);
    bool is_cuda = device_type == DeviceType::CUDA;
    return Device(device_type, is_cuda ? get_device() : -1);
  }

  Layout layout() const {
    // NB: This method is not virtual and avoid dispatches for perf.
    if (is_sparse()) {
      return kSparse;
    } else {
      return kStrided;
    }
  }

  /**
   * If `condition_when_zero_dim` is true, and the tensor is a 1-dim, 1-size
   * tensor, reshape the tensor into a 0-dim tensor (scalar).
   *
   * This helper function is called from generated wrapper code, to help
   * "fix up" tensors that legacy code didn't generate in the correct shape.
   * For example, suppose that we have a legacy function 'add' which produces
   * a tensor which is the same shape as its inputs; however, if the inputs
   * were zero-dimensional, it produced a 1-dim 1-size tensor (don't ask).
   * result->maybe_zero_dim(lhs->dim() == 0 && rhs->dim() == 0) will be called,
   * correctly resetting the dimension to 0 when when the inputs had 0-dim.
   *
   * As we teach more and more of TH to handle 0-dim correctly, this function
   * will become less necessary.  At the moment, it is often called from functions
   * that correctly handle the 0-dim case, and is just dead code in this case.
   * In the glorious future, this function will be eliminated entirely.
   */
  virtual TensorImpl* maybe_zero_dim(bool condition_when_zero_dim);

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
   *
   * WARNING: It is NOT valid to call this method on a Variable.
   * See Note [We regret making Variable hold a Tensor]
   */
  bool is_wrapped_number() const {
    AT_ASSERT(!is_variable());
    return is_wrapped_number_;
  }

  /**
   * Set whether or not a tensor was auto-wrapped from a C++ or Python
   * number.  You probably don't want to call this, unless you are
   * writing binding code.
   *
   * WARNING: It is NOT valid to call this method on a Variable.
   * See Note [We regret making Variable hold a Tensor]
   */
  void set_wrapped_number(bool value) {
    AT_ASSERT(!is_variable());
    AT_ASSERT(dim() == 0);
    is_wrapped_number_ = value;
  }

  // ~~~~~ Autograd API ~~~~~
  // Some methods below are defined in TensorImpl.cpp because Tensor is an
  // incomplete type.
  //
  // Note [Tensor versus Variable in C++]
  // ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
  // Autograd methods are only valid for the Variable::Impl subclass
  // of Tensor.  This is due to some questionable life choices, where
  // a Variable has a Tensor (so they are not the same thing), but
  // a Variable is a Tensor (they are subclassed, so that you can write
  // code on Tensor that works both with Variables and Tensors.  Poor
  // man's polymorphism).  Variable does NOT satisfy the Liskov Substitution
  // Principle for Tensor; generally you want to work with all Variables,
  // or all Tensors, but not a mix of both.  We intend to fix this in
  // the future.
  //
  // Note [We regret making Variable hold a Tensor]
  // ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
  // Tensor has a bunch of fields in it.  Are those fields always valid?
  // Not necessarily: the Variable::Impl subclass of a tensor doesn't use these
  // fields; instead, it *forwards* them to a contained, inner tensor
  // (the 'data' tensor).  It doesn't even bother keeping the fields on the
  // outer tensor up-to-date, because an end user could grab the inner
  // tensor and directly, e.g., resize it (making any outer fields we track
  // stale).
  //
  // As you might imagine, this is a TERRIBLE state of affairs to be in.
  // It makes implementing everything on TensorImpl complicated: if
  // you directly access a field on TensorImpl, you must *virtualize*
  // the function, if you want it to work correctly when called from
  // Variable (because we need to override the method to avoid looking
  // in our fields, and look in the data tensor's fields.)  Anything that
  // isn't virtualized, won't work if called on a variable.
  //
  // The way to fix this is to make Variable::Impl stop holding a tensor;
  // instead, it should just *be* a tensor.

  /**
   * Set whether or not a tensor requires gradient.
   *
   * It is only valid to call this method on a Variable.
   * See Note [Tensor versus Variable in C++].
   */
  virtual void set_requires_grad(bool requires_grad) {
    AT_ERROR("set_requires_grad is not implemented for Tensor");
  }

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
  virtual bool requires_grad() const {
    AT_ERROR("requires_grad is not implemented for Tensor");
  }

  /**
   * Return a mutable reference to the gradient.  This is conventionally
   * used as `t.grad() = x` to set a gradient to a completely new tensor.
   *
   * It is only valid to call this method on a Variable.
   * See Note [Tensor versus Variable in C++].
   */
  virtual Tensor& grad();

  /**
   * Return the accumulated gradient of a tensor.  This gradient is written
   * into when performing backwards, when this tensor is a leaf tensor.
   *
   * It is only valid to call this method on a Variable.
   * See Note [Tensor versus Variable in C++].
   */
  virtual const Tensor& grad() const;

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
   *
   * WARNING: It is NOT valid to call this method on a Variable.
   * See Note [We regret making Variable hold a Tensor]
   */
  template <typename T>
  inline T * data() const {
    AT_ASSERT(!is_variable());
    AT_ASSERTM(
        storage_initialized(),
        "The tensor has a non-zero number of elements, but its data is not allocated yet. "
        "Caffe2 uses a lazy allocation, so you will need to call "
        "mutable_data() or raw_mutable_data() to actually allocate memory.");
    AT_ASSERTM(
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
   *
   * WARNING: It is NOT valid to call this method on a Variable.
   * See Note [We regret making Variable hold a Tensor]
   */
  inline void* data() const {
    AT_ASSERT(!is_variable());
    AT_ASSERT(storage_initialized());
    AT_ASSERT(dtype_initialized());
    return static_cast<void*>(
        static_cast<char*>(storage_.data()) +
        data_type_.itemsize() * storage_offset_);
  }

  /**
   * This is just like data(), except it works with Variables.
   * This function will go away once Variable and Tensor are merged.
   * See Note [We regret making Variable hold a Tensor]
   */
  virtual void* slow_data() const {
    return data();
  }

  /**
   * Like data<T>(), but performs no checks.  You are responsible for ensuring
   * that all invariants required by data() are upheld here.
   *
   * WARNING: It is NOT valid to call this method on a Variable.
   * See Note [We regret making Variable hold a Tensor]
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
    AT_ASSERT(dtype_initialized());
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
   * Change the dimensionality of a tensor.  This is truly a resize:
   * old sizes, if they are still valid, are preserved (this invariant
   * is utilized by some call-sites, e.g., the implementation of squeeze, which
   * mostly wants the sizes to stay the same).  New dimensions are given zero
   * size and zero stride; this is probably not what you want--you should
   * set_size/set_stride afterwards.
   *
   * TODO: This should be jettisoned in favor of `set_sizes_and_strides`,
   * which is harder to misuse.
   */
  virtual void resize_dim(int64_t ndim) {
    sizes_.resize(ndim, 0);
    strides_.resize(ndim, 0);
    refresh_numel();
    refresh_contiguous();
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
    strides_[dim] = new_stride;
    refresh_numel();
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
    storage_offset_ = storage_offset;
  }

  /**
   * Like set_sizes_and_strides but assumes contiguous strides.
   *
   * WARNING: This function does not check if the requested
   * sizes/strides are in bounds for the storage that is allocated;
   * this is the responsibility of the caller
   *
   * WARNING: It is NOT valid to call this method on a Variable.
   * See Note [We regret making Variable hold a Tensor]
   */
  void set_sizes_contiguous(at::IntList new_size) {
    AT_ASSERT(!is_variable());
    auto old_dim = sizes_.size();
    auto new_dim = new_size.size();

    sizes_.resize(new_dim);
    for (size_t dim = 0; dim < new_dim; ++dim) {
      sizes_[dim] = new_size[dim];
    }

    update_to_contiguous_strides(old_dim);
    refresh_numel();
  }

  /**
   * Set the sizes and strides of a tensor.
   *
   * WARNING: This function does not check if the requested
   * sizes/strides are in bounds for the storage that is allocated;
   * this is the responsibility of the caller
   *
   * WARNING: It is NOT valid to call this method on a Variable.
   * See Note [We regret making Variable hold a Tensor]
   */
  void set_sizes_and_strides(at::IntList new_size, at::IntList new_stride) {
    AT_ASSERT(!is_variable());
    AT_CHECK(
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
   * True if a tensor is a variable.  See Note [Tensor versus Variable in C++]
   */
  bool is_variable() const { return is_variable_; };

 private:
  // As an optimization, get_device handles the typical CUDA Tensor case and
  // calls get_device_slow if the tensor stores its device somewhere else
  // (VariableImpl, SparseTensorImpl). This methods does a virtual dispatch
  // that makes it 10-20ns slower than the special-cased CUDA Tensor case.
  virtual int64_t get_device_slow() const {
    AT_ERROR(
        "get_device is not implemented for tensors with ",
        toString(tensorTypeIdToBackend(type_id())),
        " backend");
  }

 public:

  /**
   * The device type of a Tensor, e.g., DeviceType::CPU or DeviceType::CUDA.
   */
  at::DeviceType device_type() const {
    AT_ASSERT(!is_variable());
    return storage_.device_type();
  }

  /**
   * The device of a Tensor; e.g., Device(at::kCUDA, 1) (the 1-index CUDA
   * device).
   */
  at::Device GetDevice() const {
    return storage_.device();
  }

  /**
   * @brief Copies the data from a source tensor, with a contex provided to
   * carry out the underlying memcpy operation.  This method respects
   * caffe2_keep_on_shrink.
   *
   * After CopyFrom, this function guarantees that the destination tensor will
   * have the same initialization state and dtype as src.  This function
   * preserves the DeviceType of the source tensor (so, e.g., if you allocate
   * a tensor on CPU and then CopyFrom a CUDA tensor, that will to a
   * CUDA-to-CPU transfer).
   *
   * If the function is invoked without `context` the copy would be synchronous
   */
  void CopyFrom(const TensorImpl& src, at::BaseContext* context = nullptr) {
    AT_ASSERT(!is_variable());
    AT_ASSERTM(
        src.is_contiguous(),
        "Right now only copy of contiguous source Tensor is supported.");
    AT_ASSERTM(
        src.storage_initialized(),
        "Cannot copy from an uninitialized Tensor");

    if ((void*)&src == (void*)this) {
      return;
    }

    // Test if we need to allocate a new storage
    // Uninitialized storages are guaranteed to be uniquely owned,
    // so we don't need to swap in this case.
    if (storage_initialized()) {
      // If the dtype changed, we need to reallocate storage.
      if (data_type_ != src.dtype()) {
        // NB: copy preserves device_type
        // This storage will get initialized by the mutable_data call below.
        storage_ = at::Storage(device_type(), src.dtype());
      }
    }
    data_type_ = src.dtype();
    Resize(src.sizes());

    if (numel() > 0) {
      if (data_type_.copy()) {
        AT_ASSERTM(
            device_type() == ::at::DeviceType::CPU,
            "In CopyFrom source and dest tensors must both be CPU for meta copy, "
            "but dest tensor was ", device_type());
        AT_ASSERTM(
            src.device_type() == ::at::DeviceType::CPU,
            "In CopyFrom source and dest tensors must both be CPU for meta copy, "
            "but src tensor was ", src.device_type());
        data_type_.copy()(src.data(), raw_mutable_data(data_type_), numel());
      } else {
        // The following copy uses the current (thread local) stream for copying
        // and also takes the current GPU id previously set through CUDA API
        // as we don't invoke SwitchToDevice anywhere
        // TODO: this logic is overly complex and can be replaced with simple
        // dispatch based on two device types
        //
        // We'll need to use a non-CPU context to perform the copy if
        // one of the context is not CPU since only non-CPU context
        // knows how to copy between CPU and that context
        if (src.device_type() != ::at::DeviceType::CPU || device_type() == ::at::DeviceType::CPU) {
          if (!context) {
            CreateContext(src.GetDevice())
                ->CopyBytesToDevice(
                    numel() * itemsize(),
                    src.data(),
                    raw_mutable_data(data_type_),
                    device_type());
          } else {
            AT_ASSERTM(
                context->device_type() == src.device_type(),
                "Type for provided context does not match the type of source");
            context->CopyBytesToDevice(
                numel() * itemsize(), src.data(), raw_mutable_data(data_type_), device_type());
          }
        } else {
          // In case source context is CPU, and target context is non-CPU
          // We'll have to create a Context from target and perform the
          // copy using that context
          CreateContext(GetDevice())
              ->CopyBytesFromCPU(
                  numel() * itemsize(),
                  src.data(),
                  raw_mutable_data(data_type_));
        }
      }
    }
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
  void Extend(int64_t num, float growthPct, at::BaseContext* context) {
    AT_ASSERT(sizes_.size() >= 1u);
    AT_ASSERTM(num >= 0, "`num` must be non-negative for Extend");
    AT_ASSERTM(
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
    AT_ASSERTM(
        context != nullptr, "Context must be provided to Extend the tensor");
    context->CopyItemsSameDevice(
        data_type_, oldSize, oldData.get(), newData);
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
    AT_ASSERTM(
        is_contiguous_,
        "Right now ReserveSpace is only supported for contiguous Tensor.");
    AT_ASSERTM(
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
    AT_ASSERTM(
        is_contiguous_,
        "Right now Reshape is only supported for contiguous Tensor.");
    int64_t new_size = 1;
    for (auto d : dims) {
      AT_ASSERT(d >= 0);
      new_size *= d;
    }
    AT_ASSERTM(
        new_size == numel_,
        "New size and old size are not equal. You cannot use Reshape, "
        "but should use Resize."
        // TODO(jiayq): remove the following warning after pending diffs
        // stabilize.
        " The old caffe2 mixes Reshape and Resize but this behavior has "
        "been changed. If you find this error, most likely you will need "
        "to change corresponding code from Reshape to Resize.");
    auto old_dim = sizes_.size();
    sizes_ = dims;
    update_to_contiguous_strides(old_dim);
  }

  /**
   * Release whatever memory the tensor was holding but keep size and type
   * information. Subsequent call to mutable_data will trigger new memory
   * allocation.
   */
  inline void FreeMemory() {
    // We'll detach from the old Storage and create a new one
    storage_ = at::Storage(storage_.device(), data_type_);
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
  void ShareData(const TensorImpl& src) {
    // Right now, we are assuming the device_type are the same, since it is
    // inherently the same in the non-templatized code. We should probably add
    // an assert here which might affect perf a little bit.
    AT_ASSERTM(
        src.numel_ == numel_,
        "Size mismatch - did you call reshape before sharing the data?");
    // It is possible that the source tensor hasn't called mutable_data() yet,
    // in which case ShareData() doesn't make much sense since we don't really
    // know what to share yet.
    AT_ASSERTM(
        src.storage_initialized(),
        "Source tensor has no content and has size > 0");
    // Finally, do sharing.
    /* Since we create new Storage whenever we need to change data_type/capacity
     * this still keeps the original semantics
     */
    storage_ = src.storage();
    data_type_ = src.dtype();
    storage_offset_ = src.storage_offset();
  }

  void ShareExternalPointer(
      at::DataPtr&& data_ptr,
      const caffe2::TypeMeta& data_type,
      size_t capacity) {
    AT_ASSERTM(
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
          storage_ = at::Storage(storage_.device(), meta);
        }
      }
      data_type_ = meta;

      // We can reuse the existing buffer if the current data does not have
      // a special destructor and the new data doesn't have a special
      // constructor.
      if (numel_ == 0 ||
          (meta.placementNew() == nullptr && !had_special_dtor &&
           storage_.numel() >= numel_)) {
        AT_ASSERT(storage_offset_ == 0); // because we just reallocated
        return storage_.data();
      }
      const at::Allocator* allocator = storage_.allocator();
      // TODO: Get rid of StaticContext
      if (allocator == nullptr) {
        allocator = caffe2::GetAllocator(storage_.device_type());
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
      AT_ASSERT(storage_offset_ == 0); // because we just reallocated
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
  bool storage_initialized() const noexcept {
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
  bool SetDimsTemplate(at::ArrayRef<T> src) {
    auto old_numel = numel_;
    auto old_dim = sizes_.size();
    sizes_.resize(src.size());
    int64_t new_numel = 1;
    for (size_t i = 0; i < src.size(); ++i) {
      new_numel *= src[i];
      sizes_[i] = src[i];
    }
    update_to_contiguous_strides(old_dim);
    numel_ = new_numel;
    return numel_ != old_numel;
  }

  bool SetDims(at::ArrayRef<int64_t> s) {
    return SetDimsTemplate(s);
  }

  bool SetDims(at::ArrayRef<int> s) {
    return SetDimsTemplate(s);
  }

  bool SetDims(at::ArrayRef<size_t> s) {
    return SetDimsTemplate(s);
  }

  bool SetDims() {
    return SetDims(at::IntList{});
  }

  bool SetDims(const int64_t d0) {
    return SetDims(at::IntList{d0});
  }

  bool SetDims(const int64_t d0, const int64_t d1) {
    return SetDims(at::IntList{d0, d1});
  }

  bool SetDims(const int64_t d0, const int64_t d1, const int64_t d2) {
    return SetDims(at::IntList{d0, d1, d2});
  }

  bool SetDims(const int64_t d0, const int64_t d1, const int64_t d2, const int64_t d3) {
    return SetDims(at::IntList{d0, d1, d2, d3});
  }

  inline void update_to_contiguous_strides(size_t old_dim) {
    strides_.resize(sizes_.size(), 0);
    if (dim() > 0) {
      int last_idx = dim() - 1;
      strides_[last_idx] = 1;
      for (auto i = last_idx - 1; i >= 0; --i) {
        strides_[i] = strides_[i + 1] * std::max<int64_t>(sizes_[i + 1], 1);
      }
    }
    is_contiguous_ = true;
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

protected:
  /**
   * Recompute the cached numel of a tensor.  Call this if you modify sizes.
   */
  void refresh_numel() {
    AT_ASSERT(!is_variable());
    numel_ = compute_numel();
  }

  /**
   * Recompute the cached contiguity of a tensor.  Call this if you modify sizes
   * or strides.
   */
  void refresh_contiguous() {
    AT_ASSERT(!is_variable());
    is_contiguous_ = compute_contiguous();
  }

public:
  at::Storage storage_; // TODO: Fix visibility on me

protected:
  // We could save a word or two by combining the SmallVector structs,
  // since their size is redundant, and if we need to overflow the buffer space
  // we could keep the two pointers together. However, that would require
  // implementing another struct from scratch, so only do this if we're desperate.
  at::SmallVector<int64_t,5> sizes_;
  at::SmallVector<int64_t,5> strides_;

  int64_t storage_offset_ = 0;
  // If sizes and strides are empty, the numel is 1!!  However, most of the
  // time, we will immediately set sizes to {0} and reset numel to 0.
  // (Can't do that in the default initializers, because there's no way to
  // spell "allocate a one-element array" for strides_).
  int64_t numel_ = 1;

  // INVARIANT: When storage is non-null, this type meta must
  // agree with the type meta in storage
  caffe2::TypeMeta data_type_;

  // You get to have eight byte-size fields here, before you
  // should pack this into a bitfield.
  TensorTypeId type_id_;
  bool is_contiguous_ = true;
  bool is_variable_ = false;
  bool is_wrapped_number_ = false;
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
//    miscellaneous bitfield
//
static_assert(sizeof(void*) != sizeof(int64_t) || // if 64-bit...
              sizeof(TensorImpl) == sizeof(int64_t) * 24,
              "You changed the size of TensorImpl on 64-bit arch."
              "See Note [TensorImpl size constraints] on how to proceed.");

} // namespace at
