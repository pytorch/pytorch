#pragma once

#include <algorithm>
#include <atomic>
#include <memory>
#include <numeric>

#include <c10/core/Backend.h>
#include <c10/core/CopyBytes.h>
#include <c10/core/DispatchKeySet.h>
#include <c10/core/InferenceMode.h>
#include <c10/core/MemoryFormat.h>
#include <c10/core/Storage.h>
#include <c10/core/TensorOptions.h>
#include <c10/core/impl/LocalDispatchKeySet.h>
#include <c10/core/impl/SizesAndStrides.h>
#include <c10/util/Exception.h>
#include <c10/util/Flags.h>
#include <c10/util/Logging.h>
#include <c10/util/Optional.h>
#include <c10/util/accumulate.h>
#include <c10/util/python_stub.h>

// A global boolean variable to control whether we free memory when a Tensor
// is shrunk to a smaller size. As a result, a Tensor is always going to
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
class TensorBase;
} // namespace at

namespace c10 {
class Scalar;
struct IValue;
struct Storage;
class OperatorHandle;
} // namespace c10

namespace torch {
namespace jit {
using Stack = std::vector<c10::IValue>;
}
} // namespace torch

namespace c10 {

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
  virtual void set_requires_grad(
      bool requires_grad,
      at::TensorImpl* self_impl) = 0;
  virtual bool requires_grad() const = 0;
  virtual at::Tensor& mutable_grad() = 0;
  virtual const at::Tensor& grad() const = 0;
  virtual const at::Tensor& fw_grad(uint64_t level, const at::TensorBase& self)
      const = 0;
  virtual void set_fw_grad(
      const at::TensorBase& new_grad,
      const at::TensorBase& self,
      uint64_t level,
      bool is_inplace_op) = 0;
  virtual ~AutogradMetaInterface();
};

// forward declared
struct TorchDispatchTypeObject;

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

// Note [Python interpreter tag]
// ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
// We store a PyObject on TensorImpl so that we can efficiently translate
// tensors into the Python representations.  However, in some situations
// (torchdeploy) there may be multiple Python interpreters in a single process
// and we must take care not to accidentally mix up PyObjects with the wrong
// interpreters.  Thus, we also tag every TensorImpl with the Python interpreter
// it corresponds to.
//
// With torchdeploy, we have these invariants:
//  - Any given TensorImpl can be associated with AT MOST one Python
//  interpreter.
//    We represent the interpreter tag as a memory address to an instance of
//    a virtual class that is allocated once per interpreter (this is so that
//    we can request the interpreter to perform operations for us, if
//    necessary).
//  - A given TensorImpl's interpreter tag can only go from uninitialized to
//    tagged; once tagged, this is a quiescent state (once tagged to an
//    interpreter, ALWAYS tagged to that interpreter)
//  - A thread may mutate the PyObject field of a TensorImpl if and only if it
//    holds the GIL for the interpreter tagged on the TensorImpl.  (If the
//    TensorImpl is not tagged, it must first atomically claim its tag before it
//    can validly write)

// The PyInterpreter object itself is a class that contains some function
// pointers for interacting with the interpreter.  For now this is just for
// debugging, but if a Tensor can own a PyObject, the interpreter can be used to
// free it.
//
// WARNING: This class has to be written very carefully, because it may be
// possible for a Tensor to have a reference an interpreter corresponding to
// a shared library that has ALREADY BEEN UNLOADED.  This makes blindly calling
// virtual methods very dangerous, because the vtable may be garbage at that
// point (on a good day, you might get "pure virtual method called").
//
// The idea to solve this problem is we always leak PyInterpreters (so they
// always stay live even after dlclose), and disarm the "virtual methods" by
// replacing them with function pointers that just no-op.  This can't be done
// with a traditional C++ vtable, so we have to roll our own.
//
// NB: The downside with representing PyInterpreter tags as full objects is that
// it takes an extra word on TensorImpl.  If tags were instead just integer
// indices, on 64-bit architectures we could pack the tag and PyObject together
// into a single atomic word.  On 32-bit architectures we could simply say that
// only one Python interpreter is supported (erroring if a nontrivial
// interpreter tag is attempted to be set).
//
// The difficulty with this scheme is we need to maintain an out-of-line table
// to get at the PyInterpreters so that we can do virtual method calls on them,
// and registration/deregistration to this table must be done in a thread safe
// manner.  This can be easily done if the number of possible PyInterpreters is
// small enough (e.g., 8-bit integer) by simply preallocating an array of
// sufficient size to hold all possible interpreters.  Surely 128 threads is
// more than enough for anyone!
//
// I didn't decide to do this technique at the moment, because the extra word
// added by the PyInterpreter tag takes us to 24 words, which means that we
// still fit inside three eight word cache lines.  If you need to penny pinch
// another word consider doing this!

struct PyInterpreter;
struct C10_API PyInterpreter {
  using name_sig = std::string(const PyInterpreter*);
  using decref_sig = void(const PyInterpreter*, PyObject*, bool);
  using detach_sig =
      c10::intrusive_ptr<TensorImpl>(const PyInterpreter*, const TensorImpl*);
  using dispatch_sig = void(
      const PyInterpreter*,
      const c10::OperatorHandle&,
      torch::jit::Stack* stack,
      const std::shared_ptr<TorchDispatchTypeObject>& type);

  PyInterpreter(
      name_sig* name_fn,
      decref_sig* decref_fn,
      detach_sig* detach,
      dispatch_sig* dispatch)
      : name_fn_(name_fn),
        decref_fn_(decref_fn),
        detach_fn_(detach),
        dispatch_fn_(dispatch) {}

  name_sig* name_fn_;
  decref_sig* decref_fn_;
  detach_sig* detach_fn_;
  dispatch_sig* dispatch_fn_;

  // UBSAN suppression fixes: "call to function
  // (anonymous namespace)::concrete_decref_fn(c10::impl::PyInterpreter const*,
  // _object*) through pointer to incorrect function type 'void (*)(const
  // c10::impl::PyInterpreter *, _object *)'" See
  // https://github.com/google/sanitizers/issues/911

  // Report the name of this interpreter
  __ubsan_ignore_function__ std::string name() const {
    return (*name_fn_)(this);
  }

  // Run Py_DECREF on a PyObject.  We DO NOT assume the GIL is held on call
  // See NOTE [PyInterpreter::decref takes an `is_tensor` arg]
  __ubsan_ignore_function__ void decref(PyObject* pyobj, bool is_tensor) const {
    return (*decref_fn_)(this, pyobj, is_tensor);
  }

  // Perform a detach by deferring to the __torch_dispatch__ implementation of
  // detach, which will also arrange for the PyObject to get copied in this
  // situation
  __ubsan_ignore_function__ c10::intrusive_ptr<TensorImpl> detach(
      const TensorImpl* self) const {
    return (*detach_fn_)(this, self);
  }

  // Invoke the Python boxed fallback dispatch to go back into Python
  __ubsan_ignore_function__ void dispatch(
      const c10::OperatorHandle& op,
      torch::jit::Stack* stack,
      const std::shared_ptr<TorchDispatchTypeObject>& type) const {
    return (*dispatch_fn_)(this, op, stack, type);
  }

  // Disarm this PyInterpreter, making all of its methods noops.
  // Because the function pointers are raw pointers (not atomics),
  // a disarm() invocation that is concurrent with active destructors
  // is not thread safe and will trigger TSAN.  My hope is that this
  // situations doesn't ever actually happen; tensor destruction should
  // quiesce when a dlclose happens, and any long lived tensors whose
  // destructors would be disarmed here only begin the destruction process
  // on process shutdown (long after the dlclose has occurred).
  void disarm() noexcept;
};

// PyInterpreterStatus describes what the state of its interpreter tag
// is, relative to the thread currently holding the GIL.
enum class PyInterpreterStatus {
  // We just allocated the Tensor, it hasn't escaped to other threads,
  // we know that it definitely hasn't been tagged to be associated
  // with an interpreter.
  DEFINITELY_UNINITIALIZED,
  // We queried the interpreter field and it looked uninitialized.  But
  // another thread may have raced with us to tag it with some other
  // interpreter id.  So we will have to do a CEX to make sure we can
  // actually nab it.
  MAYBE_UNINITIALIZED,
  // We queried the interpreter field and it was tagged to belong to us.
  // This means we have sole write access (as we hold the GIL for this
  // interpreter)
  TAGGED_BY_US,
  // Someone else tagged this.  We can't use this TensorImpl from Python.
  TAGGED_BY_OTHER,
};

} // namespace impl

struct C10_API NamedTensorMetaInterface {
  virtual ~NamedTensorMetaInterface(){};
  virtual std::unique_ptr<NamedTensorMetaInterface> clone() const {
    TORCH_INTERNAL_ASSERT(
        false, "Not implemented: NamedTensorMetaInterface::clone");
  };
  virtual int64_t slow_dim() const {
    TORCH_INTERNAL_ASSERT(
        false, "Not implemented: NamedTensorMetaInterface::slow_dim");
  };
};

// NOTE [What is TorchDispatchTypeObject?]
// A TorchDispatchTypeObject represents the type of a Tensor subclass that has
// a __torch_dispatch__ classmethod. Concretely, it holds the class as a
// PyObject* and a PyInterpreter* that says which python interpreter the class
// came from.
//
// See NOTE [dispatch_fn's type argument] for more details
struct C10_API TorchDispatchTypeObject {
  // Steals a reference to type_object
  TorchDispatchTypeObject(
      PyObject* type_object,
      c10::impl::PyInterpreter* pyinterpreter);

  // Releases the stolen reference to type_object
  ~TorchDispatchTypeObject();

  c10::impl::PyInterpreter* pyinterpreter() const;
  PyObject* ptr() const;

 private:
  PyObject* data_;
  c10::impl::PyInterpreter* pyinterpreter_;
};

// NOTE [ Version Counter Sharing ]
//
// Every Tensor has a version counter. Version counters are incremented whenever
// the data or size of a tensor changes through in-place Variable operations.
// Version counters are used to detect modifications to saved variables which
// would result in incorrect gradient calculations. Version counters may be
// shared between Variables:
//
// 1. A view shares the version counter of the base Variable,
// 2. `x.detach()` shares the version counter of `x`,
// 3. Unpacked saved variables share the version counter of the source.
//
// Version counters are not shared in these scenarios:
//
// 1. When we replace a `Variable`'s underlying `Tensor` by calling
// `set_data(...)`,
// 2. `x.data` does not share the version counter of `x`. (See discussion at
// https://github.com/pytorch/pytorch/issues/5396)
//
// Question: Why do we put the version counter in TensorImpl instead of
// AutogradMeta?
//
// Answer: After the Variable/Tensor merge, a tensor will not have AutogradMeta
// when its `requires_grad_` is false, but when we use this tensor in the
// forward pass of a function that requires saving this tensor for backward, we
// need to keep track of this tensor's version to make sure it's always valid in
// the autograd graph.
//
// To achieve this goal, we put the version counter in TensorImpl instead of
// AutogradMeta, and have it always be available. This allows us to have the
// optimization of not carrying AutogradMeta when a tensor doesn't require
// gradient.
//
// A hypothetical alternative way to achieve this goal is to initialize
// AutogradMeta and create the version counter for the non-requires-grad tensor
// only when it's saved for backward. However, since saving a tensor for
// backward happens in the forward pass, and our invariant is that forward pass
// needs to be thread-safe, lazy-initializing AutogradMeta when saving a tensor
// can introduce race conditions when we are running the forward pass in
// multi-thread scenarios, thus making the forward pass not thread-safe anymore,
// which breaks the invariant.
struct C10_API VariableVersion {
 private:
  struct VersionCounter : intrusive_ptr_target {
    VersionCounter(uint32_t version) : version_(version) {}
    std::atomic<uint32_t> version_;
  };
  c10::intrusive_ptr<VersionCounter> version_counter_;

 public:
  // Note [Disabled VariableVersion]
  // VariableVersion struct has an intrusive_ptr pointing VersionCounter struct
  // with an atomic variable. Thus `VariableVersion(/*version=*/0)` is not as
  // cheap as we expected. In some cases constructing a VariableVersion with
  // version 0 is not necessary so we add a cheap constructor which
  // doesn't allocate the intrusive_ptr.
  // Example use cases are:
  //  - Inference tensors don't track version counter, so they'll just always
  //    have disbaled VariableVersion.
  //  - In SavedVariable class we override version_counter_ inside its
  //  construtor
  //    so that we can use the cheap constructor there.
  enum Disabled { DISABLED };
  // It's okay to return true even for inference tensor which
  // doesn't have version counter enabled.
  // We want to be permissive here since in many cases (e.g. make_variable)
  // we can std::move a TensorImpl if there's no other uses which saves us
  // an additional TensorImpl allocation.
  bool unique() const {
    return version_counter_ ? 1 == version_counter_.use_count() : true;
  }
  // NOTE: As of C++11 and 14, default-constructing a std::atomic variable
  // leaves it in a persistently undefined state. See
  // https://cplusplus.github.io/LWG/issue2334.
  VariableVersion(uint32_t version)
      : version_counter_(c10::make_intrusive<VersionCounter>(version)) {}
  VariableVersion(Disabled = DISABLED) {}

  bool enabled() const {
    return version_counter_;
  }

  // Note [Inplace update inference tensor]
  // 1. Inplace update to inference tensor is forbidden in normal mode.
  //   For example:
  //     inference_tensor.copy_(normal_tensor_requires_grad)
  //   This inplace makes inference_tensor have requires_grad=True and
  //   have a grad_fn.  This is bad because views of `inference_tensor`
  //   created in InferenceMode won't be able to know the grad_fn since
  //   their ViewMeta were not recorded. To match NoGradMode behavior
  //   that "inplace update to a view created in NoGradMode raise an error",
  //   we just ban inplace update to inference tensor since we can't tell
  //   if an inference tensor is a view created in InferenceMode.
  //
  //   Note that views of normal tensor created in InferenceMode has proper
  //   ViewMeta so that they're aware of the grad_fn correctly.
  //
  // 2. Inplace update to inference tensor in inference tensor doesn't bump
  //    version counter.
  //    * It either doesn't call bump() by skipping ADInplaceOrView kernel,
  //      - e.g. inference_tensor.add_(1)
  //    * or bump() is a no-op for inference tensor.
  //      - e.g. inference_tensor.add_(normal_tensor)
  void bump() {
    // TODO: Replace the link to the documentation once it's available.
    TORCH_CHECK(
        version_counter_ || InferenceMode::is_enabled(),
        "Inplace update to inference tensor outside InferenceMode is not allowed."
        "You can make a clone to get a normal tensor before doing inplace update."
        "See https://github.com/pytorch/rfcs/pull/17 for more details.");
    if (version_counter_) {
      ++version_counter_->version_;
    }
  }

  // Inference tensor doesn't have version counter so it shouldn't be
  // accessed.
  uint32_t current_version() const {
    TORCH_CHECK(
        version_counter_, "Inference tensors do not track version counter.");
    return version_counter_->version_;
  }
};

/**
 * NOTE: Some TensorImpl methods are small and not overridden in the
 * PyTorch codebase itself, but may theoretically need to be
 * overridden by third-party TensorImpl subclasses. This macro allows
 * users that need maximum performance and don't need these extension
 * points to disable them with a build-time flag. (In particular,
 * XLA's XLATensorImpl currently overrides these methods, so we can't
 * enable this flag by default.)
 */
#ifdef C10_DISABLE_TENSORIMPL_EXTENSIBILITY
#define TENSORIMPL_MAYBE_VIRTUAL
#else
#define TENSORIMPL_MAYBE_VIRTUAL virtual
#endif

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
  // Note [Enum ImplType]
  // This enum is temporary. In the followup refactor we should
  // think about how to specialize TensorImpl creation for view
  // tensors. Currently we only special case its key_set_ but
  // there's also potential to share version_counter_ directly
  // without creating first and then override in as_view.
  enum ImplType { VIEW };

  /**
   * Construct a 1-dim 0-size tensor backed by the given storage.
   */
  TensorImpl(
      Storage&& storage,
      DispatchKeySet,
      const caffe2::TypeMeta data_type);

  // See Note [Enum ImplType]
  TensorImpl(
      ImplType,
      Storage&& storage,
      DispatchKeySet,
      const caffe2::TypeMeta data_type);

  /**
   * Construct a 1-dim 0 size tensor that doesn't have a storage.
   */
  TensorImpl(
      DispatchKeySet,
      const caffe2::TypeMeta data_type,
      c10::optional<c10::Device> device_opt);

  // Legacy constructors so I don't have to go update call sites.
  // TODO: When Variable is added, delete these constructors
  TensorImpl(
      Storage&& storage,
      DispatchKey dispatch_key,
      const caffe2::TypeMeta data_type)
      : TensorImpl(
            std::move(storage),
            DispatchKeySet(dispatch_key),
            data_type) {}
  TensorImpl(
      DispatchKey dispatch_key,
      const caffe2::TypeMeta data_type,
      c10::optional<c10::Device> device_opt)
      : TensorImpl(DispatchKeySet(dispatch_key), data_type, device_opt) {}

 private:
  // This constructor is private, because the data_type is redundant with
  // storage.  Still, we pass it in separately because it's easier to write
  // the initializer list if we're not worried about storage being moved out
  // from under us.
  TensorImpl(
      Storage&& storage,
      DispatchKeySet,
      const caffe2::TypeMeta data_type,
      c10::optional<c10::Device>);

 public:
  TensorImpl(const TensorImpl&) = delete;
  TensorImpl& operator=(const TensorImpl&) = delete;
  TensorImpl(TensorImpl&&) = delete;
  TensorImpl& operator=(TensorImpl&&) = delete;

  /**
   * Release (decref) storage, and any other external allocations.  This
   * override is for `intrusive_ptr_target` and is used to implement weak
   * tensors.
   */
  void release_resources() override;

  /**
   * Return the DispatchKeySet corresponding to this Tensor, specifying
   * all of the DispatchKeys that this Tensor identifies as.  This is the
   * information used to dispatch operations on this tensor.
   */
  DispatchKeySet key_set() const {
    return key_set_;
  }

  /**
   * Return a reference to the sizes of this tensor.  This reference remains
   * valid as long as the tensor is live and not resized.
   */
  TENSORIMPL_MAYBE_VIRTUAL IntArrayRef sizes() const
#ifdef C10_DISABLE_TENSORIMPL_EXTENSIBILITY
  {
    return sizes_and_strides_.sizes_arrayref();
  }
#else
      ;
#endif

  /**
   * Return a reference to the strides of this tensor.  This reference remains
   * valid as long as the tensor is live and not restrided.
   */
  virtual IntArrayRef strides() const;

  /**
   * Return the number of dimensions of this tensor.  Note that 0-dimension
   * represents a Tensor that is a Scalar, e.g., one that has a single element.
   */
  TENSORIMPL_MAYBE_VIRTUAL int64_t dim() const
#ifdef C10_DISABLE_TENSORIMPL_EXTENSIBILITY
  {
    return sizes_and_strides_.size();
  }
#else
      ;
#endif

  /**
   * True if this tensor has storage. See storage() for details.
   */
#ifdef DEBUG
  // Allow subclasses to check that their storage_ is never getting set in debug
  // builds.
  virtual
#else
  TENSORIMPL_MAYBE_VIRTUAL
#endif
      bool
      has_storage() const
  // NOTE: we devirtualize this because it arguably shouldn't be an
  // error just to ask subclasses if they have storage.
  // This used to throw for most subclasses, but OpaqueTensorImpl
  // wanted it to successfully return false, so we went ahead and made
  // it a non-error.
#ifdef C10_DISABLE_TENSORIMPL_EXTENSIBILITY
  {
    return storage_;
  }
#else
      ;
#endif

  /**
   * Return the underlying storage of a Tensor.  Multiple tensors may share
   * a single storage.  A Storage is an impoverished, Tensor-like class
   * which supports far less operations than Tensor.
   *
   * Avoid using this method if possible; try to use only Tensor APIs to perform
   * operations.
   */
  TENSORIMPL_MAYBE_VIRTUAL const Storage& storage() const {
    if (C10_UNLIKELY(storage_access_should_throw_)) {
      throw_storage_access_error();
    }
    return storage_;
  }

  /**
   * Return the underlying storage, unsafely assuming this is a basic strided
   * tensor. In cases where `storage` access would throw, this returns a
   * default-constructed Storage.
   */
  inline const Storage& unsafe_storage() const {
    return storage_;
  }

  /**
   * The number of elements in a tensor.
   *
   * WARNING: Previously, if you were using the Caffe2 API, you could
   * test numel() == -1 to see if a tensor was uninitialized.  This
   * is no longer true; numel always accurately reports the product
   * of sizes of a tensor.
   */
  TENSORIMPL_MAYBE_VIRTUAL int64_t numel() const {
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
   *
   * NOTE: is_contiguous is only `TENSORIMPL_MAYBE_VIRTUAL` for
   * backward compatibility. See `set_has_contiguity_policy` and
   * `is_contiguous_custom` for the encouraged customization point.
   */
  TENSORIMPL_MAYBE_VIRTUAL bool is_contiguous(
      at::MemoryFormat memory_format = at::MemoryFormat::Contiguous) const {
    if (C10_UNLIKELY(
            has_contiguity_ !=
            static_cast<uint8_t>(HasContiguityPolicy::Default))) {
      return is_contiguous_nondefault_policy_impl(memory_format);
    }
    TORCH_INTERNAL_ASSERT_DEBUG_ONLY(compute_contiguous() == is_contiguous_);
    if (memory_format == at::MemoryFormat::ChannelsLast) {
      return is_channels_last_contiguous_;
    } else if (memory_format == at::MemoryFormat::ChannelsLast3d) {
      return is_channels_last_3d_contiguous_;
    }
    return is_contiguous_;
  }

 private:
  bool is_contiguous_nondefault_policy_impl(at::MemoryFormat) const;

 protected:
  /**
   * Customization point for is_contiguous; must also
   * set_has_contiguity_policy(HasContiguityPolicy::Custom) for this
   * to be called.
   */
  virtual bool is_contiguous_custom(at::MemoryFormat memory_format) const;

 public:
  bool is_sparse() const {
    // NB: This method is not virtual and avoid dispatches for performance
    // reasons.
    return key_set_.has(DispatchKey::SparseCPU) ||
        key_set_.has(DispatchKey::SparseCUDA) ||
        key_set_.has(DispatchKey::SparseHIP) ||
        key_set_.has(DispatchKey::SparseXPU);
  }

  // Whether a tensor is sparse COO or not. Use is_sparse_csr for checking CSR
  // format.
  bool is_sparse_csr() const {
    return key_set_.has(DispatchKey::SparseCsrCPU) ||
        key_set_.has(DispatchKey::SparseCsrCUDA);
  }

  bool is_quantized() const {
    // NB: This method is not virtual and avoid dispatches for performance
    // reasons.
    return key_set_.has(DispatchKey::QuantizedCPU) ||
        key_set_.has(DispatchKey::QuantizedCUDA) ||
        key_set_.has(DispatchKey::QuantizedXPU);
  }

  bool is_meta() const {
    // NB: This method is not virtual and avoid dispatches for performance
    // reasons.
    return key_set_.has(DispatchKey::Meta);
  }

  bool is_cpu() const {
    // NB: This method is not virtual and avoid dispatches for performance
    // reasons.
    return key_set_.has(DispatchKey::CPU) ||
        key_set_.has(DispatchKey::SparseCPU) ||
        key_set_.has(DispatchKey::SparseCsrCPU) ||
        key_set_.has(DispatchKey::QuantizedCPU) ||
        key_set_.has(DispatchKey::MkldnnCPU);
  }

  bool is_cuda() const {
    // NB: This method is not virtual and avoid dispatches for performance
    // reasons.
    return key_set_.has(DispatchKey::CUDA) ||
        key_set_.has(DispatchKey::SparseCUDA) ||
        key_set_.has(DispatchKey::SparseCsrCUDA) ||
        key_set_.has(DispatchKey::QuantizedCUDA);
  }

  bool is_xpu() const {
    // NB: This method is not virtual and avoid dispatches for performance
    // reasons.
    return key_set_.has(DispatchKey::XPU) ||
        key_set_.has(DispatchKey::SparseXPU) ||
        key_set_.has(DispatchKey::QuantizedXPU);
  }

  bool is_xla() const {
    return key_set_.has(DispatchKey::XLA);
  }

  bool is_lazy() const {
    return key_set_.has(DispatchKey::Lazy);
  }

  bool is_hip() const {
    // NB: This method is not virtual and avoid dispatches for performance
    // reasons.
    return key_set_.has(DispatchKey::HIP) ||
        key_set_.has(DispatchKey::SparseHIP);
  }

  bool is_ve() const {
    // NB: This method is not virtual and avoid dispatches for performance
    // reasons.
    return key_set_.has(DispatchKey::VE) || key_set_.has(DispatchKey::SparseVE);
  }

  bool is_mkldnn() const {
    return key_set_.has(DispatchKey::MkldnnCPU);
  }

  bool is_vulkan() const {
    return key_set_.has(DispatchKey::Vulkan);
  }

  bool is_metal() const {
    return key_set_.has(DispatchKey::Metal);
  }

  bool is_mlc() const {
    return key_set_.has(DispatchKey::MLC);
  }

  bool is_ort() const {
    return key_set_.has(DispatchKey::ORT);
  }

  // TODO: remove this once we don't automatically enabled Autograd dispatch
  // keys
  //       in TensorImpl constructor.
  // DON'T USE THIS API!! It's only created for testing purpose in
  // file aten/src/ATen/core/boxing/impl/test_helpers.h
  void remove_autograd_key() {
    key_set_ = key_set_ - autograd_dispatch_keyset;
  }

  // Inference tensor doesn't have autograd or ADInplaceOrView key.
  // Invariant:
  //   Inference tensor has version_counter_.enabled() == false
  bool is_inference() {
    bool no_ADInplaceOrView = !key_set_.has(c10::DispatchKey::ADInplaceOrView);
    bool no_Autograd = (key_set_ & c10::autograd_dispatch_keyset).empty();
    TORCH_INTERNAL_ASSERT_DEBUG_ONLY(
        no_ADInplaceOrView == no_Autograd,
        "ADInplaceOrView and Autograd keys must be on/off at the same time.");
    return no_ADInplaceOrView && no_Autograd;
  }

  int64_t get_device() const {
    TORCH_CHECK(device_opt_.has_value(), "tensor does not have a device");
    // See NOTE [c10::optional operator usage in CUDA]
    return (*device_opt_).index();
  }

  Device device() const {
    TORCH_CHECK(device_opt_.has_value(), "tensor does not have a device");
    // See NOTE [c10::optional operator usage in CUDA]
    return *device_opt_;
  }

  Layout layout() const {
    // NB: This method is not virtual and avoid dispatches for perf.
    if (is_sparse()) {
      return kSparse;
    } else if (is_sparse_csr()) {
      return kSparseCsr;
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

  /**
   * Returns true if Tensor supports as_strided and as_strided_backward.
   * This is used in autograd to perform inplace update on view Tensors.
   * See Note [View + Inplace update for base tensor] and
   * [View + Inplace update for view tensor] for details.
   * Note this method only returns true for XLA backend, where it
   * simulates strided Tensor to support most view ops, but it cannot
   * fully support general `as_strided` case.
   * It can be expanded as needed in the future, e.g sparse Tensor.
   */
  inline bool support_as_strided() const {
    return device().supports_as_strided();
  }

  // ~~~~~ Autograd API ~~~~~
  // Some methods below are defined in TensorImpl.cpp because Tensor is an
  // incomplete type.

  /**
   * Set whether or not a tensor requires gradient.
   */
  void set_requires_grad(bool requires_grad);

  /**
   * True if a tensor requires gradient.  Tensors which require gradient
   * have history tracked for any operations performed on them, so that
   * we can automatically differentiate back to them.  A tensor that
   * requires gradient and has no history is a "leaf" tensor, which we
   * accumulate gradients into.
   */
  bool requires_grad() const;

  /**
   * Return a mutable reference to the gradient.  This is conventionally
   * used as `t.grad() = x` to set a gradient to a completely new tensor.
   */
  at::Tensor& mutable_grad();

  /**
   * Return the accumulated gradient of a tensor.  This gradient is written
   * into when performing backwards, when this tensor is a leaf tensor.
   */
  const at::Tensor& grad() const;

  /**
   * Whether or not the imaginary part of the tensor should be negated
   */
  inline bool is_conj() const {
    return key_set_.has(DispatchKey::Conjugate);
  }

  /**
   * Set whether or not to take the conjugate of the tensor (flip the imaginary
   * bit).
   */
  void _set_conj(bool value) {
    if (value) {
      key_set_ = key_set_.add(DispatchKey::Conjugate);
      TORCH_INTERNAL_ASSERT(isComplexType(typeMetaToScalarType(dtype())));
    } else {
      key_set_ = key_set_.remove(DispatchKey::Conjugate);
    }
  }

  /**
   * Whether or not the tensor should be negated
   */
  inline bool is_neg() const {
    return key_set_.has(DispatchKey::Negative);
  }

  /**
   * Set whether or not to take the conjugate of the tensor (flip the imaginary
   * bit).
   */
  void _set_neg(bool value) {
    if (value) {
      key_set_ = key_set_.add(DispatchKey::Negative);
    } else {
      key_set_ = key_set_.remove(DispatchKey::Negative);
    }
  }

  /**
   * Return the accumulated gradient of a tensor. This gradient is computed
   * using forward mode AD.
   *
   * This is an internal API that should never be used by end users.
   *
   * The API is as follows:
   *   - "level" allows to specify the level of forward AD nesting for which the
   *     gradient should be returned. Note that since levels are not fully
   *     supported yet, this argument should be 0. See documentation for
   *     torch::autograd::enter_dual_level for more details about forward AD
   * nesting.
   *   - "self" should represent the Tensor whose forward grad is accessed. It
   * is required when dealing with view.
   */
  const at::Tensor& _fw_grad(uint64_t level, const at::TensorBase& self) const;

  /**
   * Sets the forward gradient for this Tensor.
   * The given Tensor might not be used directly and its content will be copied.
   *
   * This is an internal API that should never be used by end users.
   *
   * The API is as follows:
   *   - "new_grad" is a Tensor containing the new value of the gradient that
   * should be set
   *   - "self" should represent the Tensor whose forward grad is accessed. It
   * is required when dealing with view.
   *   - "level" allows to specify the level of forward AD nesting for which the
   *     gradient should be set. Note that since levels are not fully supported
   *     yet, this argument should be 0. See documentation for
   * torch::autograd::enter_dual_level for more details about forward AD
   * nesting.
   *   - "is_inplace_op" is a boolean flag that tells if this gradient was
   * generated by an inplace operation or an out of place one. This allows
   * better error checking.
   */
  void _set_fw_grad(
      const at::TensorBase& new_grad,
      const at::TensorBase& self,
      uint64_t level,
      bool is_inplace_op);

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
  inline T* data() const {
    TORCH_CHECK(
        data_type_.Match<T>(),
        "Tensor type mismatch, caller expects elements to be ",
        caffe2::TypeMeta::TypeName<T>(),
        ", while tensor contains ",
        data_type_.name(),
        ". ");
    return data_ptr_impl<T>();
  }

  /**
   * More efficient helper for Tensor::data_ptr(). Like data<T>(), but
   * does not do a type check. Unlike the untemplated data(), does
   * check has_storage() and storage_initialized().
   */
  template <typename T>
  inline T* data_ptr_impl() const {
    TORCH_CHECK(
        has_storage(),
        "Cannot access data pointer of Tensor that doesn't have storage");
    TORCH_CHECK(
        storage_initialized(),
        "The tensor has a non-zero number of elements, but its data is not allocated yet. "
        "Caffe2 uses a lazy allocation, so you will need to call "
        "mutable_data() or raw_mutable_data() to actually allocate memory.");
    // Caller does the type check.
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
    TORCH_CHECK(
        has_storage(),
        "Cannot access data pointer of Tensor that doesn't have storage");
    TORCH_CHECK(
        dtype_initialized(),
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
  inline T* unsafe_data() const {
    return storage_.unsafe_data<T>() + storage_offset_;
  }

  /**
   * Returns the TypeMeta of a tensor, which describes what data type
   * it is (e.g., int, float, ...)
   */
  const caffe2::TypeMeta dtype() const {
    return data_type_;
  }

  /**
   * Return the size of a single element of this tensor in bytes.
   */
  size_t itemsize() const {
    TORCH_CHECK(
        dtype_initialized(),
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
   */
  TENSORIMPL_MAYBE_VIRTUAL int64_t storage_offset() const {
    return storage_offset_;
  }

 protected:
  /**
   * Returns the human-readable name of the actual type of this object (e.g.,
   * TensorImpl, BatchedTensorImpl, etc.). Used for error messages.
   */
  virtual const char* tensorimpl_type_name() const {
    return "TensorImpl";
  }

 private:
  [[noreturn]] void throw_storage_access_error() const;

 public:
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
    TORCH_CHECK(
        allow_tensor_metadata_change(),
        "set_size ",
        err_msg_tensor_metadata_change_not_allowed);
    sizes_and_strides_.size_at(dim) = new_size;
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
    TORCH_CHECK(
        allow_tensor_metadata_change(),
        "set_stride ",
        err_msg_tensor_metadata_change_not_allowed);
    sizes_and_strides_.stride_at_unchecked(dim) = new_stride;
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
    TORCH_CHECK(
        allow_tensor_metadata_change(),
        "set_storage_offset ",
        err_msg_tensor_metadata_change_not_allowed);
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
    TORCH_CHECK(
        allow_tensor_metadata_change(),
        "set_sizes_contiguous ",
        err_msg_tensor_metadata_change_not_allowed);

    sizes_and_strides_.set_sizes(new_size);

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
    TORCH_CHECK(
        allow_tensor_metadata_change(),
        "set_sizes_and_strides ",
        err_msg_tensor_metadata_change_not_allowed);
    TORCH_CHECK(
        new_size.size() == new_stride.size(),
        "dimensionality of sizes (",
        new_size.size(),
        ") must match dimensionality of strides (",
        new_stride.size(),
        ")");
    const auto new_dim = new_size.size();

    sizes_and_strides_.set_sizes(new_size);

    if (new_dim > 0) {
      for (size_t dim = new_dim - 1;; dim--) {
        if (new_stride[dim] >= 0) {
          sizes_and_strides_.stride_at_unchecked(dim) = new_stride[dim];
        } else {
          // XXX: This behavior is surprising and may need to be removed to
          // support negative strides. Some pytorch functions rely on it:
          // for example, torch.cat (run TestTorch.test_cat_empty).
          if (dim == new_dim - 1) {
            sizes_and_strides_.stride_at_unchecked(dim) = 1;
          } else {
            // Keep stride monotonically increasing to match NumPy.
            sizes_and_strides_.stride_at_unchecked(dim) =
                std::max<int64_t>(
                    sizes_and_strides_.size_at_unchecked(dim + 1), 1) *
                sizes_and_strides_.stride_at_unchecked(dim + 1);
          }
        }
        if (dim == 0)
          break;
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
   * Set whether a tensor allows changes to its metadata (e.g. sizes / strides /
   * storage / storage_offset). See NOTE [ Metadata Change for a Detached Tensor
   * ] for details.
   */
  void set_allow_tensor_metadata_change(bool value) {
    allow_tensor_metadata_change_ = value;
  }

  /**
   * True if a tensor allows changes to its metadata (e.g. sizes / strides /
   * storage / storage_offset). See NOTE [ Metadata Change for a Detached Tensor
   * ] for details.
   */
  bool allow_tensor_metadata_change() const {
    return allow_tensor_metadata_change_;
  }

  /**
   * Set the pointer to autograd metadata.
   */
  void set_autograd_meta(
      std::unique_ptr<c10::AutogradMetaInterface> autograd_meta);

  /**
   * Return the pointer to autograd metadata.  May return nullptr if the
   * tensor does not track gradients.
   */
  c10::AutogradMetaInterface* autograd_meta() const;

  /**
   * Set the pointer to named tensor metadata.
   */
  void set_named_tensor_meta(
      std::unique_ptr<c10::NamedTensorMetaInterface> named_tensor_meta) {
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
    if (named_tensor_meta_ == nullptr) {
      key_set_ = key_set_.remove(DispatchKey::Named);
    } else {
      key_set_ = key_set_.add(DispatchKey::Named);
    }
  }

  void set_python_dispatch(bool k) {
    if (k) {
      key_set_ = key_set_.add(DispatchKey::Python);
    } else {
      key_set_ = key_set_.remove(DispatchKey::Python);
    }
  }

  bool is_python_dispatch() const {
    return key_set_.has(DispatchKey::Python);
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

  bool has_named_tensor_meta() const {
    return named_tensor_meta_ != nullptr;
  }

  // NOTE [ TensorImpl Shallow-Copying ]
  //
  // TensorImpl shallow-copying is used when we want to have two Variables share
  // the same tensor metadata (e.g. sizes / strides / storage pointer /
  // storage_offset), but each with a different autograd history. Example call
  // sites:
  //
  // 1. `var_detached = var.detach()` uses `shallow_copy_and_detach()` to create
  // `var_detached` that shares the same tensor metadata with `var`, but with a
  // completely new autograd history.
  // 2. `var.set_data(tensor)` uses `shallow_copy_from()` to copy tensor
  // metadata from `tensor` into `var`, while keeping `var`'s original
  // AutogradMeta.
  //
  // Functions that shallow-copy a TensorImpl (such as
  // `shallow_copy_and_detach()` / `shallow_copy_from()` /
  // `copy_tensor_metadata()`) copy the tensor metadata fields (e.g. sizes /
  // strides / storage pointer / storage_offset) by value. However, the
  // following fields are not copied:
  //
  // 1. the AutogradMeta pointer, because it is unique for each Variable.
  // 2. the version counter, because the destination TensorImpl's version
  // counter is either set to the passed-in `version_counter` (in
  // `shallow_copy_and_detach()` and `copy_tensor_metadata()`), or it is kept
  // intact (in `shallow_copy_from()`). See NOTE [ Version Counter Sharing ] for
  // details.
  //
  // In `shallow_copy_and_detach()` and `copy_tensor_metadata()`, the passed-in
  // `allow_tensor_metadata_change` determines whether the TensorImpl
  // shallow-copy allows changes to its metadata (e.g. sizes / strides / storage
  // / storage_offset). See NOTE [ Metadata Change for a Detached Tensor ] for
  // details.
  //
  // In `shallow_copy_from()`, we don't check the destination TensorImpl's
  // `allow_tensor_metadata_change_`, because `shallow_copy_from()` is used for
  // implementing functions such as `var.set_data(tensor)`, which changes
  // `var`'s tensor metadata and expects its `allow_tensor_metadata_change_` to
  // be ignored.

  /**
   * One TensorImpl can be copied to another TensorImpl if they have the same
   * DispatchKeySet. The only two special cases (for legacy reason) are:
   * CPU is compatible with CUDA and SparseCPU is
   * compatible with SparseCUDA.
   */
  inline bool has_compatible_shallow_copy_type(DispatchKeySet from) {
    auto is_dense = [](DispatchKeySet ts) {
      return ts.has(DispatchKey::CPU) || ts.has(DispatchKey::CUDA) ||
          ts.has(DispatchKey::HIP) || ts.has(DispatchKey::XPU);
    };
    auto is_sparse = [](DispatchKeySet ts) {
      return ts.has(DispatchKey::SparseCPU) ||
          ts.has(DispatchKey::SparseCUDA) || ts.has(DispatchKey::SparseHIP) ||
          ts.has(DispatchKey::SparseXPU);
    };
    return (key_set_ == from) || (is_dense(key_set_) && is_dense(from)) ||
        (is_sparse(key_set_) && is_sparse(from));
  }

  /**
   * Return a TensorImpl that is a shallow-copy of this TensorImpl.
   *
   * For usage of `version_counter` and `allow_tensor_metadata_change`,
   * see NOTE [ TensorImpl Shallow-Copying ].
   */
  virtual c10::intrusive_ptr<TensorImpl> shallow_copy_and_detach(
      const c10::VariableVersion& version_counter,
      bool allow_tensor_metadata_change) const;

  /**
   * Return a TensorImpl that is a shallow-copy of this TensorImpl.
   *
   * For usage of `version_counter` and `allow_tensor_metadata_change`,
   * see NOTE [ TensorImpl Shallow-Copying ].
   */
  virtual c10::intrusive_ptr<TensorImpl> shallow_copy_and_detach(
      c10::VariableVersion&& version_counter,
      bool allow_tensor_metadata_change) const;

  /**
   * Shallow-copies data from another TensorImpl into this TensorImpl.
   *
   * For why this function doesn't check this TensorImpl's
   * `allow_tensor_metadata_change_`, see NOTE [ TensorImpl Shallow-Copying ].
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

  // Inference tensor doesn't have version counter,
  // set_version_counter is no-op for them.
  void set_version_counter(const c10::VariableVersion& version_counter) {
    TORCH_CHECK(
        !(is_inference() && version_counter.enabled()),
        "Cannot set version_counter for inference tensor");
    version_counter_ = version_counter;
  }

  void set_version_counter(c10::VariableVersion&& version_counter) {
    TORCH_CHECK(
        !(is_inference() && version_counter.enabled()),
        "Cannot set version_counter for inference tensor");
    version_counter_ = std::move(version_counter);
  }

  const c10::VariableVersion& version_counter() const noexcept {
    return version_counter_;
  }

  void bump_version() {
    version_counter_.bump();
  }

  // Associate the TensorImpl with the specified PyObject, and, if necessary,
  // also tag the interpreter.
  //
  // NB: This lives in a header so that we can inline away the switch on status
  //
  // NB: THIS FUNCTION CAN RAISE AN EXCEPTION.  Make sure to clean up after
  // PyObject if necessary!
  void init_pyobj(
      impl::PyInterpreter* self_interpreter,
      PyObject* pyobj,
      c10::impl::PyInterpreterStatus status) {
    impl::PyInterpreter* expected = nullptr;
    switch (status) {
      case impl::PyInterpreterStatus::DEFINITELY_UNINITIALIZED:
        // caller guarantees there is no multithreaded access; if there is
        // no data race OK to do a relaxed store
        pyobj_interpreter_.store(self_interpreter, std::memory_order_relaxed);
        break;
      case impl::PyInterpreterStatus::TAGGED_BY_US:
        // no tagging is necessary, the tag is already correct
        break;
      case impl::PyInterpreterStatus::MAYBE_UNINITIALIZED:
        // attempt to claim this TensorImpl with the specified interpreter
        // tag
        if (pyobj_interpreter_.compare_exchange_strong(
                expected, self_interpreter, std::memory_order_acq_rel)) {
          break;
        }
        // test if, actually, it was already tagged by us!  this situation can't
        // be caused by a race, but it could be caused by a situation
        // where someone conservatively tagged the tensor as MAYBE_UNINITIALIZED
        // (because they didn't pre-check the tag) when actually it was
        // owned by the interpreter
        if (expected == self_interpreter) {
          break;
        }
        // fallthrough, we lost the race.  We are guaranteed not to lose the
        // race with ourself, as calls to init_pyobj with the same interpreter
        // ID must be sequentialized by the GIL
        C10_FALLTHROUGH;
      case impl::PyInterpreterStatus::TAGGED_BY_OTHER:
        TORCH_CHECK(
            false,
            "cannot allocate PyObject for Tensor on interpreter ",
            self_interpreter,
            " that has already been used by another torch deploy interpreter ",
            pyobj_interpreter_.load());
    }

    // we are the ONLY thread that can have gotten to this point.  It is not
    // possible to conflict with another zero interpreter as access is protected
    // by GIL
    pyobj_ = pyobj;
  }

  // Query the PyObject interpreter.  This may return null if there is no
  // interpreter.  This is racy!
  impl::PyInterpreter* pyobj_interpreter() {
    return pyobj_interpreter_.load(std::memory_order_acquire);
  }

  // Test the interpreter tag.  If tagged for the current interpreter, return
  // a non-nullopt (but possibly null) PyObject.  If (possibly) untagged,
  // returns a nullopt.  If it is definitely invalid, raises an error.
  //
  // NB: this lives in header so that we can avoid actually creating the
  // c10::optional
  c10::optional<PyObject*> check_pyobj(impl::PyInterpreter* self_interpreter) {
    // Note [Memory ordering on Python interpreter tag]
    impl::PyInterpreter* interpreter =
        pyobj_interpreter_.load(std::memory_order_acquire);
    if (interpreter == nullptr) {
      // NB: This never returns DEFINITELY_UNINITIALIZED because there is
      // always the possibility that another thread races to initialize
      // after we query here.  The only time when we can conclude a tensor
      // is definitely uninitialized is when we have just allocated it and
      // it cannot have escaped to other threads yet
      return c10::nullopt;
    } else if (interpreter == self_interpreter) {
      // NB: pyobj_ could still be null!
      return c10::make_optional(pyobj_);
    } else {
      TORCH_CHECK(
          false,
          "cannot access PyObject for Tensor on interpreter ",
          self_interpreter->name(),
          " that has already been used by another torch deploy interpreter ",
          pyobj_interpreter_.load()->name());
    }
  }

  // Clear the PyObject field for an interpreter, in situations where we
  // statically know the tensor is tagged with our interpreter.
  void unchecked_clear_pyobj(impl::PyInterpreter* interpreter) {
    TORCH_INTERNAL_ASSERT_DEBUG_ONLY(interpreter == pyobj_interpreter_.load());
    pyobj_ = nullptr;
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
    TORCH_CHECK(
        device_opt_.has_value(),
        "device_type cannot be run on undefined Tensor");
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
    TORCH_CHECK(sizes_and_strides_.size() >= 1u);
    TORCH_CHECK(num >= 0, "`num` must be non-negative for Extend");
    TORCH_CHECK(
        is_contiguous_,
        "Right now Extend is only supported for contiguous Tensor.");
    using SizesVector = SmallVector<int64_t, 5>;
    SizesVector newDims(
        sizes_and_strides_.sizes_begin(), sizes_and_strides_.sizes_end());
    newDims[0] += num;
    if (!storage_.data()) {
      Resize(newDims);
      return;
    }
    const auto newNumel =
        c10::multiply_integers(newDims.begin(), newDims.end());
    if (newNumel * data_type_.itemsize() <= storage_.nbytes()) {
      sizes_and_strides_.set_sizes(newDims);
      numel_ = newNumel;
      return;
    }
    SizesVector newCapacity(
        sizes_and_strides_.sizes_begin(), sizes_and_strides_.sizes_end());
    newCapacity[0] = std::max(
        newDims[0],
        static_cast<int64_t>(std::ceil(
            sizes_and_strides_.size_at_unchecked(0) * (1 + growthPct / 100))));
    auto oldData = std::move(storage_.data_ptr());
    auto oldSize = numel_;
    Resize(newCapacity);
    auto* newData = raw_mutable_data(data_type_);
    if (data_type_.copy()) {
      TORCH_CHECK(
          device_type() == DeviceType::CPU, "non-POD types work only on CPU");
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
    sizes_and_strides_.set_sizes(newDims);
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
    // TODO: eliminate newCapacity.
    SmallVector<int64_t, 5> newCapacity(
        sizes_and_strides_.sizes_begin(), sizes_and_strides_.sizes_end());
    newCapacity[0] = outer_dim;
    auto newNumel = c10::multiply_integers(newCapacity);
    if (newNumel * data_type_.itemsize() <= storage_.nbytes()) {
      return;
    }
    // Old data is discarded
    storage_.data_ptr().clear();
    auto oldSize = numel_;
    SmallVector<int64_t, 5> oldDims(
        sizes_and_strides_.sizes_begin(), sizes_and_strides_.sizes_end());
    Resize(newCapacity);
    // Allocate new memory but don't copy over the data
    raw_mutable_data(data_type_);
    sizes_and_strides_.set_sizes(oldDims);
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
      HandleResize();
    }
  }

  template <typename T>
  void Resize(const std::vector<T>& dim_source) {
    Resize(ArrayRef<T>(dim_source));
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
    sizes_and_strides_.set_sizes(dims);
    empty_tensor_restride(MemoryFormat::Contiguous);
  }

  /**
   * Release whatever memory the tensor was holding but keep size and type
   * information. Subsequent call to mutable_data will trigger new memory
   * allocation.
   */
  inline void FreeMemory() {
    // We'll detach from the old Storage and create a new one
    storage_ = Storage::create_legacy(storage_.device());
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
    //            "Source tensor don't have a data type (did you call
    //            mutable_data<T> on the tensor?)");
    if (!src.dtype_initialized()) {
      C10_LOG_EVERY_MS(WARNING, 1000)
          << "Source tensor don't have a data type (did you call mutable_data<T> on the tensor?)";
    }
    TORCH_CHECK(
        src.storage_initialized(),
        "Source tensor has no content and has size > 0");
    // Finally, do sharing.
    /* Since we create new Storage whenever we need to change data_type/nbytes
     * this still keeps the original semantics
     */
    storage_ = src.storage();
    data_type_ = src.dtype();
    device_opt_ = src.device_opt();
    storage_offset_ = src.storage_offset();
  }

  void ShareExternalPointer(
      DataPtr&& data_ptr,
      const caffe2::TypeMeta data_type,
      size_t size_bytes) {
    TORCH_CHECK(
        data_type != ScalarType::Undefined,
        "To share with a raw external pointer you need to pass in an "
        "initialized data_type(TypeMeta).");
    if (!size_bytes) {
      size_bytes = numel_ * data_type.itemsize();
    }
    if (storage_.unique()) {
      storage_.UniqueStorageShareExternalPointer(
          std::move(data_ptr), size_bytes);
      data_type_ = data_type;
      device_opt_ = storage_.device();
      storage_offset_ = 0;
    } else {
      // Create a new Storage
      storage_ = Storage(
          Storage::use_byte_size_t(),
          size_bytes,
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
  inline void* raw_mutable_data(const caffe2::TypeMeta meta) {
    // For 0-size tensors it's fine to return any pointer (including nullptr)
    if (data_type_ == meta && storage_initialized()) {
      return static_cast<void*>(
          static_cast<char*>(storage_.data()) +
          storage_offset_ * meta.itemsize());
    } else {
      bool had_special_dtor = data_type_.placementDelete() != nullptr;
      storage_offset_ = 0;
      data_type_ = meta;
      // NB: device is not changed

      // We can reuse the existing buffer if the current data does not have
      // a special destructor and the new data doesn't have a special
      // constructor.
      if (numel_ == 0 ||
          (meta.placementNew() == nullptr && !had_special_dtor &&
           (storage_.nbytes() >= (numel_ * data_type_.itemsize())))) {
        TORCH_INTERNAL_ASSERT(
            storage_offset_ == 0); // because we just reallocated
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
        auto data_ptr = allocator->allocate(numel_ * data_type_.itemsize());
        storage_.set_data_ptr_noswap(PlacementDeleteContext::makeDataPtr(
            std::move(data_ptr), dtor, size, storage_.device()));
        data_type_.placementNew()(storage_.data(), numel_);
      } else {
        // For fundamental type, new and delete is easier.
        storage_.set_data_ptr_noswap(
            allocator->allocate(numel_ * data_type_.itemsize()));
      }
      storage_.set_nbytes(numel_ * data_type_.itemsize());
      TORCH_INTERNAL_ASSERT(
          storage_offset_ == 0); // because we just reallocated
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
    if (storage_initialized() && data_type_.Match<T>()) {
      return static_cast<T*>(storage_.data()) + storage_offset_;
    }
    // Check it here statically - otherwise TypeMeta would throw the runtime
    // error in attempt to invoke TypeMeta::ctor()
    static_assert(
        std::is_default_constructible<T>::value,
        "Tensor can't hold non-default-constructable types");
    return static_cast<T*>(raw_mutable_data(caffe2::TypeMeta::Make<T>()));
  }

  /**
   * True if a tensor is storage initialized.  A tensor may become
   * storage UNINITIALIZED after a Resize() or FreeMemory()
   */
  bool storage_initialized() const {
    TORCH_CHECK(
        has_storage(),
        "cannot call storage_initialized on tensor that does not have storage");
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

  void set_storage_keep_dtype(at::Storage storage) {
    TORCH_CHECK(
        allow_tensor_metadata_change(),
        "set_storage ",
        err_msg_tensor_metadata_change_not_allowed);
    storage_ = std::move(storage);
    device_opt_ = storage_.device();
  }

  void set_storage_and_dtype(
      at::Storage storage,
      const caffe2::TypeMeta data_type) {
    set_storage_keep_dtype(storage);
    data_type_ = data_type;
  }

  /**
   * Set the strides of the tensor to match memory_format
   *
   * WARNING: This function doesn't rearrange data and assumes tensor is a
   * memory contiguous
   */
  void empty_tensor_restride(MemoryFormat memory_format) {
#ifdef DEBUG
    TORCH_INTERNAL_ASSERT(
        compute_numel() == numel_,
        "If you are seeing this error, that means empty_tensor_restride was "
        "called before setting correct numel");
#endif
    switch (memory_format) {
      case MemoryFormat::Contiguous: {
        // dim_ is a virtual call, don't repeat it
        const auto dim_ = dim();
        sizes_and_strides_.resize(dim_);
        if (dim_ > 0) {
          const auto last_idx = dim_ - 1;
          sizes_and_strides_.stride_at_unchecked(last_idx) = 1;
          for (auto i = last_idx - 1; i >= 0; --i) {
            sizes_and_strides_.stride_at_unchecked(i) =
                sizes_and_strides_.stride_at_unchecked(i + 1) *
                std::max<int64_t>(
                    sizes_and_strides_.size_at_unchecked(i + 1), 1);
          }
        }
        break;
      }
      case MemoryFormat::ChannelsLast: {
        TORCH_CHECK(
            dim() == 4, "required rank 4 tensor to use channels_last format");
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
  void HandleResize();

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
    sizes_and_strides_.resize(src.size());
    int64_t new_numel = 1;
    for (size_t i = 0; i < src.size(); ++i) {
      new_numel *= src[i];
      sizes_and_strides_.size_at_unchecked(i) = src[i];
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

  bool SetDims(
      const int64_t d0,
      const int64_t d1,
      const int64_t d2,
      const int64_t d3) {
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
   * Compute the number of elements based on the sizes of a
   * tensor. Catches integer overflow that may occur when a tensor
   * using a sparse layout has multiple dimensions with large sizes.
   */
  int64_t safe_compute_numel() const {
    int64_t n = 1;
    for (auto s : sizes()) {
      TORCH_CHECK(
          s == 0 || n <= std::numeric_limits<int64_t>::max() / s,
          "numel: integer multiplication overflow");
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
   * Recompute the cached numel of a tensor.  Call this if you modify
   * sizes.
   *
   * For tensors with sparse layouts, use safe_refresh_numel() instead
   * because it will catch integer overflow that may occur for tensors
   * with sparse layouts and large dimensions.
   */
  void refresh_numel() {
    numel_ = compute_numel();
  }

  /**
   * Recompute the cached numel of a tensor.  Call this if you modify
   * sizes. Use only for tensors with sparse layouts because only
   * sparse tensor are likely to have sizes that may lead to integer
   * overflow when computing numel.
   */
  void safe_refresh_numel() {
    numel_ = safe_compute_numel();
  }

  /**
   * Recompute the cached contiguity of a tensor.  Call this if you modify sizes
   * or strides.
   */
  void refresh_contiguous() {
    is_contiguous_ = compute_contiguous();
    // Note:
    // Dim 0, 1, 2 will never be a channels last 2d/3d format
    // Dim 3+ is possibly be a channels last 2d format (Dim 4 only at this
    // point) Dim 4+ is possibly be a channels last 3d format (Dim 5 only at
    // this point)
    switch (dim()) {
      case 4:
        is_channels_last_contiguous_ = compute_channels_last_contiguous_2d();
        is_channels_last_3d_contiguous_ = false;
        is_channels_last_ = compute_strides_like_channels_last_2d();
        is_channels_last_3d_ = false;
        is_non_overlapping_and_dense_ = is_contiguous_ ||
            is_channels_last_contiguous_ || compute_non_overlapping_and_dense();
        break;
      case 5:
        is_channels_last_contiguous_ = compute_channels_last_contiguous_2d();
        is_channels_last_3d_contiguous_ = !is_channels_last_contiguous_ &&
            compute_channels_last_contiguous_3d();
        is_channels_last_ = !is_channels_last_3d_contiguous_ &&
            compute_strides_like_channels_last_2d();
        is_channels_last_3d_ =
            !is_channels_last_ && compute_strides_like_channels_last_3d();
        is_non_overlapping_and_dense_ = is_contiguous_ ||
            is_channels_last_contiguous_ || is_channels_last_3d_contiguous_ ||
            compute_non_overlapping_and_dense();
        break;
      default:
        is_channels_last_contiguous_ = false;
        is_channels_last_3d_contiguous_ = false;
        // is_channels_last_ and is_channels_last_3d_ are suggested
        // memory_format. Being channels_last_contiguous doesn't necessarily
        // mean the tensor is strided like channels_last: for strides on channel
        // dimension could suggest desired memory_layout, but it doesn't affect
        // memory storage
        is_channels_last_ = false;
        is_channels_last_3d_ = false;
        is_non_overlapping_and_dense_ =
            is_contiguous_ || compute_non_overlapping_and_dense();
    }
  }

  /**
   * Copy the tensor metadata fields (e.g. sizes / strides / storage pointer /
   * storage_offset) from one TensorImpl to another TensorImpl.
   *
   * For usage of `version_counter` and `allow_tensor_metadata_change`, see NOTE
   * [ TensorImpl Shallow-Copying ].
   */
  static void copy_tensor_metadata(
      const TensorImpl* src_impl,
      TensorImpl* dest_impl,
      const c10::VariableVersion& version_counter,
      bool allow_tensor_metadata_change);

  /**
   * Copy the tensor metadata fields (e.g. sizes / strides / storage pointer /
   * storage_offset) from one TensorImpl to another TensorImpl.
   *
   * For usage of `version_counter` and `allow_tensor_metadata_change`, see NOTE
   * [ TensorImpl Shallow-Copying ].
   */
  static void copy_tensor_metadata(
      const TensorImpl* src_impl,
      TensorImpl* dest_impl,
      c10::VariableVersion&& version_counter,
      bool allow_tensor_metadata_change);

 private:
  static void copy_tensor_metadata_except_version_counter(
      const TensorImpl* src_impl,
      TensorImpl* dest_impl,
      bool allow_tensor_metadata_change);

 protected:
  // Error message to show when the user tries to change tensor metadata on
  // Tensor created from .data or .detach().
  //
  // See NOTE [ Metadata Change for a Detached Tensor ] for details.
  static const char* const err_msg_tensor_metadata_change_not_allowed;

 public:
  void set_storage_access_should_throw() {
    storage_access_should_throw_ = true;
  }

  bool owns_pyobj() {
    return owns_pyobj_;
  }

  void set_owns_pyobj(bool b) {
    owns_pyobj_ = b;
  }

 protected:
  // Policy for adjusting the behavior of is_contiguous(). Allows
  // subclass customization while still being able to inline
  // is_contiguous() in the common case.
  enum class HasContiguityPolicy : uint8_t {
    // Default behavior: check is_contiguous_ and similar bitflags.
    Default,
    // Throw a generic error message that this tensor type does not
    // support is_contiguous.
    ContiguityNotSupported,
    // Call virtual is_contiguous_custom method to implement custom
    // is_contiguous behavior.
    CustomBehavior,
  };

  void set_has_contiguity_policy(HasContiguityPolicy p) {
    has_contiguity_ = static_cast<uint8_t>(p);
  }

  Storage storage_;

 private:
  // This pointer points to an AutogradMeta struct that stores autograd-specific
  // fields (such as grad_ / grad_fn_ / grad_accumulator_). This pointer always
  // has unique ownership (meaning only one TensorImpl can own it at a time).
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

  // This field contains the interpreter tag for this object.  See
  // Note [Python interpreter tag] for general context
  //
  // Note [Memory ordering on Python interpreter tag]
  // ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
  // What memory_order do we need when accessing this atomic?  We don't
  // need a single total modification order (as provided by
  // memory_order_seq_cst) as pyobj_interpreter_ is monotonic: it can only
  // transition from -1 to some positive integer and never changes afterwards.
  // Because there is only one modification, it trivially already has a total
  // modification order (e.g., we don't need fences or locked instructions on
  // x86)
  //
  // In fact, one could make a reasonable argument that relaxed reads are OK,
  // due to the presence of external locking (GIL) to ensure that interactions
  // with other data structures are still correctly synchronized, so that
  // we fall in the "Single-Location Data Structures" case as described in
  // http://www.open-std.org/jtc1/sc22/wg21/docs/papers/2020/p2055r0.pdf
  // However, on x86, it doesn't matter if I use acquire or relaxed on the load
  // as I get the same assembly in both cases.  So I just use the more
  // conservative acquire (which will impede compiler optimizations but I don't
  // care)
  std::atomic<impl::PyInterpreter*> pyobj_interpreter_;

  // This field contains a weak reference to a PyObject representing
  // this Tensor.  It MUST NOT be a strong reference, as that would
  // create a reference cycle between Tensor and the PyObject.  If
  // pyobj is nullptr, when we transfer Tensor to Python, we allocate
  // a new PyObject for it and set this field.  This field does not
  // have to be protected by an atomic as it is only allowed to be
  // accessed when you hold the GIL.
  //
  // When a PyObject dies, you are obligated to clear this field
  // (otherwise, you will try to use-after-free the pyobj); this currently
  // occurs in THPVariable_clear in torch/csrc/autograd/python_variable.cpp
  PyObject* pyobj_;

  c10::impl::SizesAndStrides sizes_and_strides_;

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

  // Tensor is contiguous
  bool is_contiguous_ : 1;
  // gcc doesn't like enum class bitfields; see
  // https://gcc.gnu.org/bugzilla/show_bug.cgi?id=61414
  /* HasContiguityPolicy */ uint8_t has_contiguity_ : 2;

  // Tensor is a subclass that does not permit storage access.
  bool storage_access_should_throw_ : 1;

  // default member initializers for bit-fields only available with -std=c++2a
  // or -std=gnu++2a
  inline void init_bitfields() {
    is_contiguous_ = true;
    has_contiguity_ = static_cast<uint8_t>(HasContiguityPolicy::Default);

    is_channels_last_ = false;
    is_channels_last_contiguous_ = false;
    is_channels_last_3d_ = false;
    is_channels_last_3d_contiguous_ = false;
    is_non_overlapping_and_dense_ = true;
    is_wrapped_number_ = false;
    allow_tensor_metadata_change_ = true;
    reserved_ = false;
    owns_pyobj_ = false;
    storage_access_should_throw_ = false;
  }

  // Tensor is stored in the channels last 2d memory format, when dimensions
  // order is (N)CHW and C-strides < W-strides < H-strides (< N-strides)
  // (If size of any dimension is equal to 1, this dimension strides value
  // is not taken into account).
  bool is_channels_last_ : 1;

  // Channels last contiguous tensor is channel last tensor which occupies
  // contiguous memory block.
  bool is_channels_last_contiguous_ : 1;

  // Tensor is stored in the channels last 3d memory format, when dimensions
  // order is (N)CDHW and C-strides < W-strides < H-strides < D - strides (<
  // N-strides) (If size of any dimension is equal to 1, this dimension strides
  // value is not taken into account).
  bool is_channels_last_3d_ : 1;

  // Channels last 3d contiguous tensor is channel last 3d tensor which occupies
  // contiguous memory block.
  bool is_channels_last_3d_contiguous_ : 1;

  // Dense tensor is the tensor that store values in a contiguous block of
  // memory. Non-overlapping tensor is the tensor in which elements occupy
  // individual non-repetitive memory.
  bool is_non_overlapping_and_dense_ : 1;

  bool is_wrapped_number_ : 1;

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
  bool allow_tensor_metadata_change_ : 1;

  // we decide to keep reserved_ and it will
  // live in Tensor after the split
  // The logic is that if Extend() or ReserveSpace() were ever called,
  // then subsequent Resize()s will not free up Storage.
  bool reserved_ : 1;

  // If pyobj_ is nullptr, this is always false.
  // Otherwise, this indicates whether or not TensorImpl owns the pyobj_
  // or vice versa.  Ordinarily, pyobj_ owns TensorImpl, but if the
  // Python object's refcount goes to zero, we flip the ownership
  // direction (to make sure the pyobj stays live).
  bool owns_pyobj_ : 1;

  // The set of DispatchKeys which describe this tensor.  NB: this
  // does NOT include Autograd (historically, it did, but
  // not anymore!)
  //
  // INVARIANT: named_tensor_meta_ != nullptr  <==>
  // key_set_.has(DispatchKey::Named)
  DispatchKeySet key_set_;
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
//    named tensor metadata pointer
//    version counter pointer
//    Python interpreter pointer
//    PyObject pointer
//    SizesAndStrides size/pointer
//    SizesAndStrides sizes (pre-allocated 0)
//    SizesAndStrides sizes (pre-allocated 1)
//    SizesAndStrides sizes (pre-allocated 2)
//    SizesAndStrides sizes (pre-allocated 3)
//    SizesAndStrides sizes (pre-allocated 4)
//    SizesAndStrides strides (pre-allocated 0)
//    SizesAndStrides strides (pre-allocated 1)
//    SizesAndStrides strides (pre-allocated 2)
//    SizesAndStrides strides (pre-allocated 3)
//    SizesAndStrides strides (pre-allocated 4)
//    storage offset
//    numel
//    data type, device, is_contiguous, storage_access_should_throw_, bitfields
//    DispatchKeySet
//
static_assert(
    sizeof(void*) != sizeof(int64_t) || // if 64-bit...
        sizeof(TensorImpl) == sizeof(int64_t) * 24,
    "You changed the size of TensorImpl on 64-bit arch."
    "See Note [TensorImpl size constraints] on how to proceed.");
} // namespace c10
