#include <c10/core/TensorImpl.h>

#include <c10/core/Backend.h>
#include <c10/core/InferenceMode.h>
#include <c10/core/SymIntArrayRef.h>
#include <c10/core/WrapDimMinimal.h>
#include <c10/core/impl/LocalDispatchKeySet.h>
#include <c10/core/impl/PyInterpreter.h>
#include <c10/core/impl/TorchDispatchModeTLS.h>
#include <c10/util/Optional.h>
#include <c10/util/irange.h>

C10_DEFINE_bool(
    caffe2_keep_on_shrink,
    true,
    "If set, keeps memory when a tensor is shrinking its size.");

C10_DEFINE_int64(
    caffe2_max_keep_on_shrink_memory,
    LLONG_MAX,
    "The maximum memory in bytes to keep on shrink, if the difference between "
    "tensor sizes is bigger than this then tensor will be reset.");

namespace c10 {

const char* const TensorImpl::err_msg_tensor_metadata_change_not_allowed =
    "is not allowed on a Tensor created from .data or .detach().\n"
    "If your intent is to change the metadata of a Tensor (such as sizes / strides / storage / storage_offset)\n"
    "without autograd tracking the change, remove the .data / .detach() call and wrap the change in a `with torch.no_grad():` block.\n"
    "For example, change:\n"
    "    x.data.set_(y)\n"
    "to:\n"
    "    with torch.no_grad():\n"
    "        x.set_(y)";

at::Tensor& TensorImpl::mutable_grad() {
  if (!autograd_meta_)
    autograd_meta_ = impl::GetAutogradMetaFactory()->make();
  return autograd_meta_->mutable_grad();
}

const at::Tensor& TensorImpl::grad() const {
  // Yes, I know this looks really weird.  But I don't really have a choice as
  // long as this function returns a const reference to Tensor.  I'm not
  // really sure how I would have designed this API differently, but it
  // is not so easy to fix right now because the mutable counterpart of
  // this function must keep working so that "x.grad() = ..." keeps working
  // (part of public API).
  if (!autograd_meta_)
    return impl::GetAutogradMetaFactory()->undefined_tensor();
  return autograd_meta_->grad();
}

const at::Tensor& TensorImpl::_fw_grad(
    uint64_t level,
    const at::TensorBase& self) const {
  // See TensorImpl::grad() above for explanation about the line below
  if (!autograd_meta_)
    return impl::GetAutogradMetaFactory()->undefined_tensor();
  return autograd_meta_->fw_grad(level, self);
}

void TensorImpl::_set_fw_grad(
    const at::TensorBase& new_grad,
    const at::TensorBase& self,
    uint64_t level,
    bool is_inplace_op) {
  if (!autograd_meta_)
    autograd_meta_ = impl::GetAutogradMetaFactory()->make();
  autograd_meta_->set_fw_grad(new_grad, self, level, is_inplace_op);
}

TensorImpl::~TensorImpl() {
  destroy_pyobj_if_needed();
}

TensorImpl::TensorImpl(
    Storage&& storage,
    DispatchKeySet key_set,
    const caffe2::TypeMeta data_type)
    // Use std::forward to suppress static analyzer false positive.
    : TensorImpl(
          std::forward<Storage>(storage),
          key_set,
          data_type,
          storage.device()) {}

// [Note: Python key removal]
// ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
// In most constructors for TensorImpl, you will see Python and
// PythonTLSSnapshot keys are removed from the passed in DispatchKeySet.  Why?
//
// INVARIANT: Python and PythonTLSSnapshot dispatch keys are set iff PyObject
// for the Tensor has a nontrivial __torch_dispatch__ implementation.
//
// When a fresh TensorImpl is created, there is *no* PyObject (this only gets
// initialized lazily at the first point in time the Tensor passes into Python).
// So we would violate the invariant.
//
// In practice, what will happen shortly afterwards is that the TensorImpl
// will get its PyObject initialized by Tensor._make_subclass; at this point
// the Python and PythonTLSSnapshot dispatch keys will be set and all is well.
// The point is to delay the dispatch key setting until that point.

// NOLINTNEXTLINE(cppcoreguidelines-pro-type-member-init)
TensorImpl::TensorImpl(
    ImplType type,
    Storage&& storage,
    DispatchKeySet key_set,
    const caffe2::TypeMeta data_type)
    : storage_(std::move(storage)),
      pyobj_interpreter_(nullptr),
      pyobj_(nullptr),
      storage_offset_(0),
      numel_(0),
      data_type_(data_type),
      device_opt_(storage_.device()),
      key_set_(key_set - c10::python_ks) { // See [Note: Python key removal]
  init_bitfields();
  // Inference tensor doesn't have version counter.
  if (!is_inference()) {
    version_counter_ = VariableVersion(/*version=*/0);
  }
}

TensorImpl::TensorImpl(
    DispatchKeySet key_set,
    const caffe2::TypeMeta data_type,
    c10::optional<c10::Device> device_opt)
    // NOLINTNEXTLINE(performance-move-const-arg)
    : TensorImpl({}, key_set, data_type, std::move(device_opt)) {}

// NOLINTNEXTLINE(cppcoreguidelines-pro-type-member-init)
TensorImpl::TensorImpl(
    Storage&& storage,
    DispatchKeySet key_set,
    const caffe2::TypeMeta data_type,
    c10::optional<c10::Device> device_opt)
    : storage_(std::move(storage)),
      pyobj_interpreter_(nullptr),
      pyobj_(nullptr),
      storage_offset_(0),
      numel_(0),
      data_type_(data_type),
      device_opt_(device_opt) {
  init_bitfields();

  if (!key_set.empty()) {
    TORCH_INTERNAL_ASSERT(
        data_type == ScalarType::Undefined || device_opt_.has_value());
    // UndefinedTensorImpl is a singleton, so we skip logging it
    C10_LOG_API_USAGE_ONCE("tensor.create");
  }

  // XXX: if updating keyset logic here also update
  // _change_backend_component_keys
  bool inference_mode = c10::InferenceMode::is_enabled();

  // TODO: be more explicit about the full key set at call sites so we
  // don't have to keep recomputing it here
  auto k = key_set.highestBackendKey();

  key_set = key_set | getAutocastRelatedKeySetFromBackend(k);

  // See [Note: Python key removal]
  key_set = key_set - c10::python_ks;

  // Inference tensor doesn't have autograd related keys.
  if (inference_mode) {
    // See Note [Expected TLS state in InferenceMode] for why we exclude
    // Autograd & ADInplaceOrView keys. Normally key_set only contains backend
    // keys but we do the substraction here to make sure.
    key_set_ = key_set - c10::autograd_dispatch_keyset_with_ADInplaceOrView;
  } else {
    // TODO: Ideally we only add AutogradBackend key when the tensor requires
    // grad.
    //       See Note [Dream: skip VariableType kernel when requires_grad=false]
    key_set_ = key_set | getAutogradRelatedKeySetFromBackend(k);
  }

  // Inference tensor doesn't have version counter.
  if (!is_inference()) {
    version_counter_ = VariableVersion(/*version=*/0);
  }
  // we would also like to check that non-cpu devices have an index, but some
  // Caffe2 operators create Storages with default devices.
}

void TensorImpl::_change_backend_component_keys(c10::Device device) {
  BackendComponent new_backend = toBackendComponent(device.type());
  BackendComponent old_backend = key_set_.highestBackendKey();

  // following logic TensorImpl::TensorImpl, update the BackendComponent related
  // keys to correspond to device

  // TODO: Autocoast should be a per-backend functionality key, once that change
  // is made this key swap will not be necessary.
  auto key_set =
      key_set_ - c10::getAutocastRelatedKeySetFromBackend(old_backend);
  key_set = key_set | c10::getAutocastRelatedKeySetFromBackend(new_backend);

  // See note [Removing keys from DispatchKeySet Only Affects Functionality
  // Keys]
  key_set = key_set.remove_backend(old_backend);
  key_set_ = key_set | DispatchKeySet(new_backend);
}

void TensorImpl::HandleResize() {
  // If needed, we will free the data. the next mutable_data() call
  // will create the data storage.
  bool reset_tensor = false;
  if (reserved_) {
    // If tensor is reserved then don't claim its memeory unless nbytes()
    // is smaller than new size
    reset_tensor =
        storage_.nbytes() < (storage_offset_ + numel_) * data_type_.itemsize();
  } else {
    reset_tensor = storage_.nbytes() <
            (storage_offset_ + numel_) * data_type_.itemsize() ||
        !FLAGS_caffe2_keep_on_shrink ||
        storage_.nbytes() - (storage_offset_ + numel_) * data_type_.itemsize() >
            static_cast<size_t>(FLAGS_caffe2_max_keep_on_shrink_memory);
  }

  if (reset_tensor && storage_initialized()) {
    FreeMemory();
  }
}

template <typename T>
bool_is_contiguous _compute_contiguous(
    ArrayRef<T> sizes,
    ArrayRef<T> strides,
    T numel) {
  bool is_contiguous = true;
  if (numel == 0)
    return bool_is_contiguous(is_contiguous);
  T z = 1;
  // NB: make sure we do signed arithmetic
  for (int64_t d = int64_t(sizes.size()) - 1; d >= 0; d--) {
    const auto size_d = sizes[d];
    if (size_d != 1) {
      if (strides[d] == z) {
        z *= size_d;
      } else {
        is_contiguous = false;
        break;
      }
    }
  }
  return bool_is_contiguous(is_contiguous);
}

// NB: intentionally bypass the normal accessors; we always want to be
// consistent with what is actually stored on the struct
#define COMPUTE_WITH_SIZES_STRIDES_NUMEL(TEMPLATE)                            \
  (has_symbolic_sizes_strides_                                                \
       ? TEMPLATE<c10::SymInt>(                                               \
             extra_meta_->sizes_, extra_meta_->strides_, extra_meta_->numel_) \
       : TEMPLATE<int64_t>(                                                   \
             sizes_and_strides_.sizes_arrayref(),                             \
             sizes_and_strides_.strides_arrayref(),                           \
             numel_))

#define COMPUTE_WITH_SIZES_STRIDES(TEMPLATE)                               \
  (has_symbolic_sizes_strides_                                             \
       ? TEMPLATE<c10::SymInt>(extra_meta_->sizes_, extra_meta_->strides_) \
       : TEMPLATE<int64_t>(                                                \
             sizes_and_strides_.sizes_arrayref(),                          \
             sizes_and_strides_.strides_arrayref()))

bool_is_contiguous TensorImpl::compute_contiguous() const {
  return COMPUTE_WITH_SIZES_STRIDES_NUMEL(_compute_contiguous);
}

template <typename T>
bool_is_channels_last_contiguous _compute_channels_last_contiguous_2d(
    ArrayRef<T> sizes,
    ArrayRef<T> strides) {
  // Please don't combine these code, constant array is used here to let
  // compiler fully unroll the loop to get better performance
  switch (sizes.size()) {
    case 4: {
      T expected = 1;
      for (auto& d : {1, 3, 2, 0}) {
        const auto size_d = sizes[d];
        if (size_d != 1) {
          if (strides[d] != expected) {
            return bool_is_channels_last_contiguous(false);
          }
          expected *= size_d;
        }
      }
      return bool_is_channels_last_contiguous(true);
    }
    // NOLINTNEXTLINE(bugprone-branch-clone)
    case 3:
      // TODO dim == 3 case will be enabled once it is fully tested
      return bool_is_channels_last_contiguous(false);
    default:
      return bool_is_channels_last_contiguous(false);
  }
}

bool_is_channels_last_contiguous TensorImpl::
    compute_channels_last_contiguous_2d() const {
  return COMPUTE_WITH_SIZES_STRIDES(_compute_channels_last_contiguous_2d);
}

template <typename T>
bool_is_channels_last_3d_contiguous _compute_channels_last_contiguous_3d(
    ArrayRef<T> sizes,
    ArrayRef<T> strides) {
  // Please don't combine these code, constant array is used here to let
  // compiler fully unroll the loop to get better performance
  switch (sizes.size()) {
    case 5: {
      T expected = 1;
      for (auto& d : {1, 4, 3, 2, 0}) {
        const auto size_d = sizes[d];
        if (size_d != 1) {
          if (strides[d] != expected) {
            return bool_is_channels_last_3d_contiguous(false);
          }
          expected *= size_d;
        }
      }
      return bool_is_channels_last_3d_contiguous(true);
    }
    // NOLINTNEXTLINE(bugprone-branch-clone)
    case 4:
      // TODO dim == 4 case will be enabled once it is fully tested
      return bool_is_channels_last_3d_contiguous(false);
    default:
      return bool_is_channels_last_3d_contiguous(false);
  }
}

bool_is_channels_last_3d_contiguous TensorImpl::
    compute_channels_last_contiguous_3d() const {
  return COMPUTE_WITH_SIZES_STRIDES(_compute_channels_last_contiguous_3d);
}

bool_is_channels_last TensorImpl::compute_strides_like_channels_last_2d()
    const {
  return bool_is_channels_last(
      COMPUTE_WITH_SIZES_STRIDES(is_channels_last_strides_2d));
}

bool_is_channels_last_3d TensorImpl::compute_strides_like_channels_last_3d()
    const {
  return bool_is_channels_last_3d(
      COMPUTE_WITH_SIZES_STRIDES(is_channels_last_strides_3d));
}

template <typename T>
bool_is_non_overlapping_and_dense _compute_non_overlapping_and_dense(
    ArrayRef<T> sizes,
    ArrayRef<T> strides) {
  auto dim = sizes.size();
  if (dim == 1) {
    return bool_is_non_overlapping_and_dense(sizes[0] < 2 || strides[0] == 1);
  }
  SmallVector<int64_t, 5> perm;
  perm.resize(dim);
  for (const auto i : c10::irange(dim)) {
    perm[i] = i;
  }
  // Sort by strides, leaving 0 and 1 sized dims at the end of the array
  std::sort(perm.begin(), perm.end(), [&](int64_t a, int64_t b) {
    if (sizes[a] < 2) {
      return false;
    } else if (sizes[b] < 2) {
      return true;
    }
    return strides[a] < strides[b];
  });
  T require_stride = 1;
  for (const auto i : c10::irange(dim)) {
    const auto size_perm_i = sizes[perm[i]];
    if (size_perm_i < 2) {
      return bool_is_non_overlapping_and_dense(true);
    }
    if (strides[perm[i]] != require_stride) {
      return bool_is_non_overlapping_and_dense(false);
    }
    require_stride *= size_perm_i;
  }
  return bool_is_non_overlapping_and_dense(true);
}

bool_is_non_overlapping_and_dense TensorImpl::
    compute_non_overlapping_and_dense() const {
  return COMPUTE_WITH_SIZES_STRIDES(_compute_non_overlapping_and_dense);
}

void TensorImpl::release_resources() {
  autograd_meta_.reset();
  if (storage_) {
    storage_ = {};
  }
  destroy_pyobj_if_needed();
}

void TensorImpl::destroy_pyobj_if_needed() {
  if (owns_pyobj()) {
    TORCH_INTERNAL_ASSERT(pyobj_interpreter_ != nullptr);
    TORCH_INTERNAL_ASSERT(pyobj_ != nullptr);
    (*pyobj_interpreter_.load(std::memory_order_acquire))
        ->decref(_unchecked_untagged_pyobj(), /*is_tensor*/ true);
    // NB: this destructor can only be entered when there are no
    // references to this C++ object (obviously), NOR any references
    // to the PyObject (if there are references to the PyObject,
    // then the PyObject holds an owning reference to the tensor).
    // So it is OK to clear pyobj_ here as it is impossible for it to
    // be used again (modulo weak reference races)
    pyobj_ = nullptr; // for safety
  }
}

#ifndef C10_DISABLE_TENSORIMPL_EXTENSIBILITY
bool TensorImpl::has_storage() const {
  return storage_;
}
#endif

void TensorImpl::throw_storage_access_error() const {
  TORCH_CHECK_NOT_IMPLEMENTED(
      false, "Cannot access storage of ", tensorimpl_type_name());
}

impl::PyInterpreter& TensorImpl::load_pyobj_interpreter() const {
  auto interpreter = pyobj_interpreter_.load(std::memory_order_acquire);
  if (interpreter) {
    return *interpreter;
  }
  TORCH_CHECK(
      false,
      "cannot access PyObject for Tensor on interpreter ",
      (*pyobj_interpreter_.load())->name());
}

bool TensorImpl::is_contiguous_custom(at::MemoryFormat memory_format) const {
  if (C10_UNLIKELY(matches_python_custom(SizesStridesPolicy::CustomStrides))) {
    return load_pyobj_interpreter()->is_contiguous(this, memory_format);
  }
  return is_contiguous_default(memory_format);
}

bool TensorImpl::is_strides_like_custom(at::MemoryFormat memory_format) const {
  if (C10_UNLIKELY(matches_python_custom(SizesStridesPolicy::CustomStrides))) {
    return load_pyobj_interpreter()->is_strides_like(this, memory_format);
  }
  return is_strides_like_default(memory_format);
}

bool TensorImpl::is_non_overlapping_and_dense_custom() const {
  if (C10_UNLIKELY(matches_python_custom(SizesStridesPolicy::CustomStrides))) {
    return load_pyobj_interpreter()->is_non_overlapping_and_dense(this);
  }
  return is_non_overlapping_and_dense_default();
}

IntArrayRef TensorImpl::sizes_custom() const {
  if (C10_UNLIKELY(matches_python_custom(SizesStridesPolicy::CustomSizes))) {
    return load_pyobj_interpreter()->sizes(this);
  }
  return sizes_default();
}

c10::SymIntArrayRef TensorImpl::sym_sizes_custom() const {
  if (C10_UNLIKELY(matches_python_custom(SizesStridesPolicy::CustomSizes))) {
    return load_pyobj_interpreter()->sym_sizes(this);
  }
  return sym_sizes_default();
}

c10::SymInt TensorImpl::sym_numel_custom() const {
  if (C10_UNLIKELY(matches_python_custom(SizesStridesPolicy::CustomSizes))) {
    return load_pyobj_interpreter()->sym_numel(this);
  }
  return sym_numel_default();
}

c10::SymIntArrayRef TensorImpl::sym_strides_custom() const {
  if (C10_UNLIKELY(matches_python_custom(SizesStridesPolicy::CustomStrides))) {
    return load_pyobj_interpreter()->sym_strides(this);
  }
  return sym_strides_default();
}

c10::Device TensorImpl::device_custom() const {
  if (C10_UNLIKELY(python_custom_device_)) {
    return load_pyobj_interpreter()->device(this);
  }
  return device_default();
}

IntArrayRef TensorImpl::strides_custom() const {
  if (C10_UNLIKELY(matches_python_custom(SizesStridesPolicy::CustomStrides))) {
    return load_pyobj_interpreter()->strides(this);
  }
  return strides_default();
}

int64_t TensorImpl::dim_custom() const {
  if (C10_UNLIKELY(matches_python_custom(SizesStridesPolicy::CustomSizes))) {
    return load_pyobj_interpreter()->dim(this);
  }
  return dim_default();
}

int64_t TensorImpl::numel_custom() const {
  if (C10_UNLIKELY(matches_python_custom(SizesStridesPolicy::CustomSizes))) {
    // TODO: fix this
    return load_pyobj_interpreter()->sym_numel(this).expect_int();
  }
  return numel_default();
}

c10::Layout TensorImpl::layout_custom() const {
  if (C10_UNLIKELY(python_custom_layout_)) {
    return load_pyobj_interpreter()->layout(this);
  }
  // TODO: fix this
  TORCH_CHECK(
      0, "Tensors of type ", tensorimpl_type_name(), " do not have layout")
  // return layout_default();
}

int64_t TensorImpl::storage_offset_custom() const {
  if (C10_UNLIKELY(matches_python_custom(SizesStridesPolicy::CustomSizes))) {
    // TODO: fix this
    return load_pyobj_interpreter()->sym_storage_offset(this).expect_int();
  }
  return storage_offset_default();
}

c10::SymInt TensorImpl::sym_storage_offset_custom() const {
  if (C10_UNLIKELY(matches_python_custom(SizesStridesPolicy::CustomSizes))) {
    return load_pyobj_interpreter()->sym_storage_offset(this);
  }
  return sym_storage_offset_default();
}

static void deletePlacementDeleteContext(void* ptr) {
  delete static_cast<PlacementDeleteContext*>(ptr);
}

at::DataPtr PlacementDeleteContext::makeDataPtr(
    at::DataPtr&& data_ptr,
    PlacementDtor placement_dtor,
    size_t size,
    at::Device device) {
  auto* ptr = data_ptr.get();
  return {
      ptr,
      new PlacementDeleteContext(std::move(data_ptr), placement_dtor, size),
      &deletePlacementDeleteContext,
      device};
}

AutogradMetaInterface::~AutogradMetaInterface() = default;

// Setting requires_grad to true on inference tensor outside InferenceMode
// is forbidden.  Ideally it would also be illegal inside InferenceMode.
// But there's no way that we can directly allocate a tensor to have
// requires_grad = true in C++ constructor so set_requires_grad is widely
// used in C++ frontend. Forbidding it inside InferenceMode will force users
// to delete these setter code in their code which is not ideal.
void TensorImpl::set_requires_grad(bool requires_grad) {
  TORCH_CHECK(
      !(requires_grad && is_inference() && !c10::InferenceMode::is_enabled()),
      "Setting requires_grad=True on inference tensor outside InferenceMode is not allowed.");
  if (!requires_grad && !autograd_meta_)
    return;
  if (!autograd_meta_)
    autograd_meta_ = impl::GetAutogradMetaFactory()->make();
  // NB: In principle, setting requires_grad to false could result in
  // the AutogradMeta becoming equal to a default constructed state,
  // in which case we could apply the nullptr AutogradMeta optimization
  // (see autograd_meta_ docs).  But we don't do this right now.  Note
  // that it is unsound to unconditionally set AutogradMeta to false
  // when you set requires_grad to False, as there may be nontrivial
  // information content in the other fields; for example, we may
  // have set the string name for a Variable, or there may be hooks
  // registered for it.
  autograd_meta_->set_requires_grad(requires_grad, this);
}

bool TensorImpl::requires_grad() const {
  if (!autograd_meta_)
    return false;
  return autograd_meta_->requires_grad();
}

void TensorImpl::set_autograd_meta(
    std::unique_ptr<c10::AutogradMetaInterface> autograd_meta) {
  // NB: autograd_meta may be null!  That just means it's the default
  // constructor
  autograd_meta_ = std::move(autograd_meta);
}

c10::AutogradMetaInterface* TensorImpl::autograd_meta() const {
  // NB: Might return null!
  return autograd_meta_.get();
}

template <typename VariableVersion>
c10::intrusive_ptr<TensorImpl> TensorImpl::shallow_copy_and_detach_core(
    VariableVersion&& version_counter,
    bool allow_tensor_metadata_change) const {
  c10::intrusive_ptr<TensorImpl> r;
  const auto mode_stack_len = c10::impl::TorchDispatchModeTLS::stack_len();
  // TODO: do we have to exclude after Python dispatch key set?
  if (mode_stack_len > 0 &&
      !c10::impl::tls_is_dispatch_key_excluded(DispatchKey::Python)) {
    const auto& cur_torch_dispatch_mode_state =
        c10::impl::TorchDispatchModeTLS::get_stack_at(mode_stack_len - 1);
    r = cur_torch_dispatch_mode_state->pyinterpreter()->detach(this);
  } else if (
      key_set_.has(DispatchKey::Python) &&
      !c10::impl::tls_is_dispatch_key_excluded(DispatchKey::Python)) {
    r = (*pyobj_interpreter_.load(std::memory_order_acquire))->detach(this);
  }
  if (r) {
    r->set_version_counter(std::forward<VariableVersion>(version_counter));
    r->set_allow_tensor_metadata_change(allow_tensor_metadata_change);
    return r;
  }
  // otherwise just copy the TensorImpl and not the PyObject.  Since
  // the interpreter is dead no one can call us out on it
  auto impl = c10::make_intrusive<TensorImpl>(
      // No need to populate Storage; copy_tensor_metadata will do it for us.
      key_set_,
      data_type_,
      device_opt_);
  copy_tensor_metadata(
      /*src_impl=*/this,
      /*dest_impl=*/impl.get(),
      /*version_counter=*/std::forward<VariableVersion>(version_counter),
      /*allow_tensor_metadata_change=*/allow_tensor_metadata_change);

  impl->refresh_numel();
  impl->refresh_contiguous();
  return impl;
}

c10::intrusive_ptr<TensorImpl> TensorImpl::shallow_copy_and_detach(
    const c10::VariableVersion& version_counter,
    bool allow_tensor_metadata_change) const {
  return shallow_copy_and_detach_core(
      version_counter, allow_tensor_metadata_change);
}

c10::intrusive_ptr<TensorImpl> TensorImpl::shallow_copy_and_detach(
    c10::VariableVersion&& version_counter,
    bool allow_tensor_metadata_change) const {
  return shallow_copy_and_detach_core(
      std::move(version_counter), allow_tensor_metadata_change);
}

// This function copies all of the metadata from the src tensor except for:
// - key_set_
// - storage_
// - storage_access_should_throw_
// - sizes_strides_policy_
// - version_counter_
// - allow_tensor_metadata_change_
// The idea is that if we have a "wrapper tensor" (like in functionalization),
// all of the above are properties that the wrapper will want to customize,
// while everything else should be mirrored between the wrapper and the inner
// tensor.
void TensorImpl::copy_generic_tensor_metadata(
    const TensorImpl* src_impl,
    TensorImpl* dest_impl) {
  dest_impl->sizes_and_strides_ = src_impl->sizes_and_strides_;
  dest_impl->has_symbolic_sizes_strides_ =
      src_impl->has_symbolic_sizes_strides_;

  dest_impl->storage_offset_ = src_impl->storage_offset_;
  dest_impl->data_type_ = src_impl->data_type_;
  dest_impl->device_opt_ = src_impl->device_opt_;
  dest_impl->is_contiguous_ = src_impl->is_contiguous_;
  dest_impl->is_channels_last_contiguous_ =
      src_impl->is_channels_last_contiguous_;
  dest_impl->is_channels_last_3d_contiguous_ =
      src_impl->is_channels_last_3d_contiguous_;
  dest_impl->is_channels_last_ = src_impl->is_channels_last_;
  dest_impl->is_channels_last_3d_ = src_impl->is_channels_last_3d_;
  dest_impl->is_non_overlapping_and_dense_ =
      src_impl->is_non_overlapping_and_dense_;
  dest_impl->is_wrapped_number_ = src_impl->is_wrapped_number_;
  dest_impl->reserved_ = src_impl->reserved_;
  if (src_impl->extra_meta_ != nullptr) {
    dest_impl->extra_meta_ = src_impl->extra_meta_->clone();
  }

  // NB: symbolic sizes and strides are copied as is custom policy, but python
  // policy is NOT (you have no Python object to dispatch to!)
  // NB: subclass relevant policy doesn't have to be copied; the
  // constructor sets this up

  dest_impl->refresh_sizes_strides_policy();
  dest_impl->refresh_layout_policy();
  dest_impl->refresh_device_policy();
}

void TensorImpl::copy_tensor_metadata_except_version_counter(
    const TensorImpl* src_impl,
    TensorImpl* dest_impl,
    bool allow_tensor_metadata_change) {
  // First call the generic copy function
  copy_generic_tensor_metadata(src_impl, dest_impl);
  // Then copy everything else (see the comment at copy_generic_tensor_metadata
  // for the list of metadata that it does not directly copy).
  dest_impl->storage_ = src_impl->storage_;
  // Copying tensor metadata doesn't change the PyObject (maybe
  // it should), which means that we have to preserve whatever the
  // original Python keyset was (as it's associated with the PyObject
  // being a tensor subclass or not)
  dest_impl->key_set_ = (src_impl->key_set_ - c10::python_ks) |
      (dest_impl->key_set_ & c10::python_ks);
  dest_impl->set_allow_tensor_metadata_change(allow_tensor_metadata_change);
  dest_impl->storage_access_should_throw_ =
      src_impl->storage_access_should_throw_;
}

void TensorImpl::copy_tensor_metadata(
    const TensorImpl* src_impl,
    TensorImpl* dest_impl,
    const c10::VariableVersion& version_counter,
    bool allow_tensor_metadata_change) {
  copy_tensor_metadata_except_version_counter(
      src_impl, dest_impl, allow_tensor_metadata_change);
  // TODO: In the ideal end state, it's okay to set disabled version_counter
  // on inference tensor since it's a no-op. This requires refactor on call
  // sites.
  if (!dest_impl->is_inference()) {
    dest_impl->set_version_counter(version_counter);
  }
}

void TensorImpl::copy_tensor_metadata(
    const TensorImpl* src_impl,
    TensorImpl* dest_impl,
    c10::VariableVersion&& version_counter,
    bool allow_tensor_metadata_change) {
  copy_tensor_metadata_except_version_counter(
      src_impl, dest_impl, allow_tensor_metadata_change);
  if (!dest_impl->is_inference()) {
    dest_impl->set_version_counter(std::move(version_counter));
  }
}

// Legacy Caffe2 operations

void TensorImpl::Extend(int64_t num, float growthPct) {
  TORCH_CHECK(sizes_and_strides_.size() >= 1u);
  TORCH_CHECK(num >= 0, "`num` must be non-negative for Extend");
  TORCH_CHECK(
      is_contiguous_,
      "Right now Extend is only supported for contiguous Tensor.");
  TORCH_CHECK(
      !has_symbolic_sizes_strides_,
      "Extend() called on tensor with symbolic shape")

  using SizesVector = SmallVector<int64_t, 5>;
  IntArrayRef sizes_and_strides = sizes_and_strides_.sizes_arrayref();
  SizesVector newDims(sizes_and_strides.begin(), sizes_and_strides.end());
  newDims[0] += num;
  if (!storage_.data()) {
    Resize(newDims);
    return;
  }
  const auto newNumel = c10::multiply_integers(newDims.begin(), newDims.end());
  if (newNumel * data_type_.itemsize() <= storage_.nbytes()) {
    sizes_and_strides_.set_sizes(newDims);
    numel_ = newNumel;
    return;
  }
  SizesVector newCapacity(sizes_and_strides.begin(), sizes_and_strides.end());
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

void TensorImpl::ReserveSpace(int64_t outer_dim) {
  TORCH_CHECK(
      is_contiguous_,
      "Right now ReserveSpace is only supported for contiguous Tensor.");
  TORCH_CHECK(
      !has_symbolic_sizes_strides_,
      "ReserveSpace() called on tensor with symbolic shape")

  TORCH_CHECK(storage_.unique(), "Can't call ReserveSpace on shared storage.");
  // TODO: eliminate newCapacity.
  IntArrayRef sizes_and_strides = sizes_and_strides_.sizes_arrayref();
  SmallVector<int64_t, 5> newCapacity(
      sizes_and_strides.begin(), sizes_and_strides.end());
  newCapacity[0] = outer_dim;
  auto newNumel = c10::multiply_integers(newCapacity);
  if (newNumel * data_type_.itemsize() <= storage_.nbytes()) {
    return;
  }
  // Old data is discarded
  storage_.data_ptr().clear();
  auto oldSize = numel_;
  SmallVector<int64_t, 5> oldDims(
      sizes_and_strides.begin(), sizes_and_strides.end());
  Resize(newCapacity);
  // Allocate new memory but don't copy over the data
  raw_mutable_data(data_type_);
  sizes_and_strides_.set_sizes(oldDims);
  numel_ = oldSize;
  reserved_ = true;
}

void TensorImpl::Reshape(const std::vector<int64_t>& dims) {
  TORCH_CHECK(
      is_contiguous_,
      "Right now Reshape is only supported for contiguous Tensor.");
  TORCH_CHECK(
      !has_symbolic_sizes_strides_,
      "Reshape() called on tensor with symbolic shape")

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

void TensorImpl::FreeMemory() {
  // We'll detach from the old Storage and create a new one
  if (storage_.use_count() != 1 || !storage_.resizable() ||
      !storage_.allocator()) {
    storage_ = Storage::create_legacy(storage_.device());
  } else {
    storage_.reset_legacy();
  }
  storage_offset_ = 0;
}

void TensorImpl::ShareData(const TensorImpl& src) {
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

void TensorImpl::ShareExternalPointer(
    DataPtr&& data_ptr,
    const caffe2::TypeMeta data_type,
    size_t size_bytes) {
  TORCH_CHECK(
      data_type != ScalarType::Undefined,
      "To share with a raw external pointer you need to pass in an "
      "initialized data_type(TypeMeta).");
  TORCH_CHECK(
      !has_symbolic_sizes_strides_,
      "ShareExternalPointer() called on tensor with symbolic shape");
  if (!size_bytes) {
    size_bytes = numel_ * data_type.itemsize();
  }
  if (storage_.unique()) {
    storage_.UniqueStorageShareExternalPointer(std::move(data_ptr), size_bytes);
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

void clone_symvec(SymIntArrayRef src, SymDimVector& dst) {
  dst.clear();
  dst.reserve(src.size());
  for (size_t i = 0; i < src.size(); i++) {
    dst.emplace_back(src[i].clone());
  }
}

// NB: this doesn't check that the sizes/strides/offset are in bound for the
// storage, and furthermore, it CANNOT do so as in some cases we temporarily
// violate invariants by first setting sizes/strides, and then updating the
// storage
void TensorImpl::set_sizes_and_strides(
    c10::SymIntArrayRef sizes,
    c10::SymIntArrayRef strides,
    c10::optional<c10::SymInt> storage_offset) {
  auto int_sizes = asIntArrayRefSlowOpt(sizes);
  auto int_strides = asIntArrayRefSlowOpt(strides);
  if (int_sizes && int_strides &&
      (!storage_offset.has_value() || !storage_offset->is_symbolic()) &&
      !has_symbolic_sizes_strides_) {
    set_sizes_and_strides(*int_sizes, *int_strides);
    if (storage_offset.has_value())
      set_storage_offset(storage_offset->as_int_unchecked());
    return;
  }

  has_symbolic_sizes_strides_ = true;
  refresh_sizes_strides_policy();
  if (!extra_meta_) {
    extra_meta_ = std::make_unique<ExtraMeta>();
    if (!storage_offset.has_value()) {
      extra_meta_->storage_offset_ = storage_offset_;
    }
  }
  clone_symvec(sizes, extra_meta_->sizes_);
  clone_symvec(strides, extra_meta_->strides_);
  if (storage_offset.has_value())
    extra_meta_->storage_offset_ = storage_offset->clone();

  refresh_numel();
  refresh_contiguous();
}

namespace impl {

namespace {
AutogradMetaFactory* meta_factory = nullptr;
} // namespace

void SetAutogradMetaFactory(AutogradMetaFactory* factory) {
  meta_factory = factory;
}
AutogradMetaFactory* GetAutogradMetaFactory() {
  TORCH_CHECK(
      meta_factory,
      "Support for autograd has not been loaded; have you linked against libtorch.so?")
  return meta_factory;
}

} // namespace impl

} // namespace c10
