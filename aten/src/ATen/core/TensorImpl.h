#pragma once

#include <atomic>
#include <memory>

#include "ATen/core/Storage.h"
#include "ATen/core/optional.h"
#include "ATen/core/TensorTypeId.h"
#include "ATen/core/TensorTypeIdRegistration.h"
#include "ATen/core/LegacyTypeDispatch.h"
#include "ATen/core/Backend.h"

struct THTensor;

namespace at {
class Scalar;
struct Type;
struct Storage;
struct Tensor;
} // namespace at

namespace at {
struct TensorImplOptions {
  bool is_variable_;
  bool has_storage_;
  bool has_strides_;
  TensorImplOptions(bool is_variable)
                    : is_variable_(is_variable),
                      has_storage_(true),
                      has_strides_(true) {};
  TensorImplOptions(bool is_variable,
                    bool has_storage,
                    bool has_strides)
                    : is_variable_(is_variable),
                      has_storage_(has_storage),
                      has_strides_(has_strides) {};
};
} // namespace at

namespace at {
struct AT_API TensorImpl : public c10::intrusive_ptr_target {
  TensorImpl() = delete;
  TensorImpl(TensorTypeId type_id, ScalarType scalar_type, Allocator *allocator, TensorImplOptions options);
  TensorImpl(Storage&& storage, TensorTypeId type_id, TensorImplOptions options);

  virtual void release_resources() override;

  Type & type() const {
    // NB: It's valid to use getTypeRaw here, because the TensorImpl
    // could not have been created without initializing the Type first.
    // TODO: This is not actually true via the Caffe2 codepath!  Make
    // it so.
    return *globalLegacyTypeDispatch().getTypeRaw(tensorTypeIdToBackend(type_id()), scalar_type(), is_variable());
  }

  TensorTypeId type_id() const { return type_id_; }
  virtual IntList sizes() const;
  virtual IntList strides() const;
  virtual int64_t dim() const;
  virtual const Storage& storage() const;
  friend struct Type;

  virtual int64_t numel() const {
#ifdef DEBUG
    AT_ASSERT(compute_numel() == numel_);
#endif
    return numel_;
  }

  virtual bool is_contiguous() const {
    if (!options_.has_strides_) {
      AT_ERROR(type_id_, " does not have strides, and hence does not have is_contiguous");
    }
#ifdef DEBUG
    AT_ASSERT(compute_contiguous() == is_contiguous_);
#endif
    return is_contiguous_;
  }

  // this is called by the generated wrapper code when there are conditions
  // when this output tensor should be zero dimensional. e.g. when all inputs
  // to a function 'add' were zero dimensional, then condition_when_zero_dim == true.
  // we also prevent this from getting marked as a zero dim tensor if it is not
  // the right shape afterall.
  virtual TensorImpl* maybe_zero_dim(bool condition_when_zero_dim);

  // True if a tensor was auto-wrapped from a C++ or Python number.
  // Wrapped numbers do not participate in the result type computation for
  // mixed-type operations if there are any Tensors that are not wrapped
  // numbers. Otherwise, they behave like their non-wrapped equivalents.
  // See [Result type computation] in TensorIterator.h.
  bool is_wrapped_number() const {
    return is_wrapped_number_;
  }
  void set_wrapped_number(bool value) {
    AT_ASSERT(dim() == 0);
    is_wrapped_number_ = value;
  }

  // ~~~~~ Autograd API ~~~~~
  // Some methods below are defined in TensorImpl.cpp because Tensor is an
  // incomplete type.

  virtual void set_requires_grad(bool requires_grad) {
    AT_ERROR("set_requires_grad is not implemented for Tensor");
  }
  virtual bool requires_grad() const {
    AT_ERROR("requires_grad is not implemented for Tensor");
  }

  virtual Tensor& grad();
  virtual const Tensor& grad() const;

  // TODO: make these protected
  // Note: storage->size() may be greater than the recorded size
  // of a tensor
  at::Storage storage_;

  template <typename T>
  inline T * data() const {
    return storage_.data<T>() + storage_offset_;
  }

  template <typename T>
  inline T * unsafe_data() const {
    return storage_.unsafe_data<T>() + storage_offset_;
  }

  inline at::ScalarType scalar_type() const {
    return scalar_type_;
  }

  virtual int64_t storage_offset() const {
    if (!options_.has_storage_) {
      AT_ERROR(type_id_, " does not have storage, and hence does not have storage_offset");
    }
    return storage_offset_;
  }

  // represents that numel() == 0.
  inline bool is_empty() const {
    return numel() == 0;
  }

  virtual void resize_dim(int64_t ndim) {
    // NB: This is *truly* a resize; calling code (e.g., squeeze)
    // assumes that old values are preserved
    sizes_.resize(ndim);
    refresh_numel();
    if (options_.has_strides_) {
      strides_.resize(ndim);
      refresh_contiguous();
    }
  }

  virtual void set_size(int64_t dim, int64_t new_size) {
    sizes_[dim] = new_size;
    refresh_numel();
    if (options_.has_strides_) {
      refresh_contiguous();
    }
  }

  virtual void set_stride(int64_t dim, int64_t new_stride) {
    if (!options_.has_strides_) {
      AT_ERROR(type_id_, " does not have strides, and hence does not have set_stride");
    }
    strides_[dim] = new_stride;
    refresh_numel();
    refresh_contiguous();
  }

  virtual void set_storage_offset(int64_t storage_offset) {
    if (!options_.has_storage_) {
      AT_ERROR(type_id_, " does not have storage, and hence does not have set_storage_offset");
    }
    storage_offset_ = storage_offset;
    refresh_numel();
    if (options_.has_strides_) {
      refresh_contiguous();
    }
  }

  // WARNING: This function does not check if the requested
  // sizes/strides are in bounds for the storage that is allocated;
  // this is the responsibility of the caller
  void set_sizes_and_strides(at::IntList new_size, at::optional<at::IntList> new_stride) {
    if (options_.has_strides_) {
      AT_CHECK(
          new_size.size() == new_stride->size(),
          "dimensionality of sizes (",
          new_size.size(),
          ") must match dimensionality of strides (",
          new_stride->size(),
          ")");
      strides_ = new_stride->vec();
      refresh_contiguous();
    } else {
      AT_CHECK(!new_stride, type_id_, " does not have strides, but new_stride is non-null");
    }
    sizes_ = new_size.vec();
    refresh_numel();
  }

  virtual int64_t size(int64_t d) const;
  virtual int64_t stride(int64_t d) const;

  bool is_variable() const { return options_.is_variable_; };

 private:
  int64_t storage_offset_;
  std::vector<int64_t> sizes_;
  std::vector<int64_t> strides_;

  bool is_contiguous_;
  int64_t numel_;

  int64_t compute_numel() const {
    int64_t n = 1;
    for (auto s : sizes()) {
      n *= s;
    }
    return n;
  }
  bool compute_contiguous() const;
  void refresh_numel() {
    numel_ = compute_numel();
  }
  void refresh_contiguous() {
    is_contiguous_ = compute_contiguous();
  }

 protected:
  TensorTypeId type_id_;
  // INVARIANT: When storage is non-null, this scalar type must
  // agree with the scalar type in storage
  ScalarType scalar_type_;
  TensorImplOptions options_;
  bool is_wrapped_number_ = false;

 private:
  TensorImpl(Storage&& storage, TensorTypeId type_id, ScalarType scalar_type, TensorImplOptions options);
};
} // namespace at
