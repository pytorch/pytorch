#include "caffe2/core/tensor.h"
#include "caffe2/core/tensor_int8.h"

#include "caffe2/core/blob_stats.h"

#if defined(EXPOSE_C2_OPS) || \
    !defined(CAFFE2_IS_XPLAT_BUILD) && !defined(C10_MOBILE)
#include <ATen/core/grad_mode.h>
#include "ATen/core/Tensor.h"
#endif

namespace caffe2 {

CAFFE_DEFINE_KNOWN_TYPE(Tensor, Caffe2Tensor);

TensorPrinter::TensorPrinter(
    // NOLINTNEXTLINE(modernize-pass-by-value)
    const std::string& tensor_name,
    const std::string& file_name,
    int limit)
    : to_file_(!file_name.empty()),
      limit_(limit ? limit : k_limit_default_),
      tensor_name_(tensor_name) {
  if (to_file_) {
    // We will output to file instead of printing on screen.
    // We will write each individual tensor to its individual file.
    // NOLINTNEXTLINE(modernize-make-unique)
    log_file_.reset(new std::ofstream(
        file_name, std::ofstream::out | std::ofstream::trunc));
    CAFFE_ENFORCE(
        log_file_->good(),
        "Failed to open TensorPrinter file ",
        file_name,
        ". rdstate() = ",
        log_file_->rdstate());
  }
}

// NOLINTNEXTLINE(bugprone-exception-escape)
TensorPrinter::~TensorPrinter() {
  if (log_file_.get()) {
    log_file_->close();
  }
}

void TensorPrinter::PrintMeta(const Tensor& tensor) {
  if (to_file_) {
    (*log_file_) << MetaStr(tensor) << std::endl;
  } else {
    LOG(INFO) << MetaStr(tensor);
  }
}

std::string TensorPrinter::MetaStr(const Tensor& tensor) {
  std::stringstream meta_stream;
  meta_stream << "Tensor " << tensor_name_ << " of type "
              << tensor.dtype().name() << ". Dims: (";
  for (const auto dim : tensor.sizes()) {
    meta_stream << dim << ",";
  }
  meta_stream << "): ";
  return meta_stream.str();
}

TypeMeta GetTensorType(const void* c) {
  const Tensor* tc = static_cast<const Tensor*>(c);
  return tc->dtype();
}

TypeMeta GetInt8TensorType(const void* c) {
  const int8::Int8TensorCPU* int8_tensor =
      static_cast<const int8::Int8TensorCPU*>(c);
  return (int8_tensor->t).dtype();
}

// TODO(jerryzh): Remove
static CaffeMap<TypeIdentifier, TypeCall> type_call_registry_{
    {TypeMeta::Id<Tensor>(), GetTensorType},
    {TypeMeta::Id<int8::Int8TensorCPU>(), GetInt8TensorType},
};

TypeCall GetTypeCallFunction(TypeIdentifier id) {
  auto f = type_call_registry_.find(id);
  if (f == type_call_registry_.end()) {
    return nullptr;
  }
  return f->second;
}

void RegisterTypeCallFunction(TypeIdentifier id, TypeCall c) {
  type_call_registry_[id] = c;
}

int GetGPUIDForPointer(const void* ptr);

vector<int64_t>
GetTensorInfo(const void* c, size_t* capacity, DeviceOption* device) {
  CHECK(capacity);
  const Tensor* tc = static_cast<const Tensor*>(c);
  CHECK(tc);
  // NOLINTNEXTLINE(clang-analyzer-core.CallAndMessage)
  CHECK(tc->unsafeGetTensorImpl());
  CHECK(tc->unsafeGetTensorImpl()->storage().unsafeGetStorageImpl());
  *capacity = tc->storage().nbytes();
  ExtractDeviceOption(device, tc->GetDevice());
  return tc->sizes().vec();
}

vector<int64_t>
GetInt8TensorInfo(const void* c, size_t* capacity, DeviceOption* device) {
  const int8::Int8TensorCPU* int8_tensor =
      static_cast<const int8::Int8TensorCPU*>(c);
  return GetTensorInfo(&(int8_tensor->t), capacity, device);
}

// since we only have one tensor, probably need to remove this at some point?
static CaffeMap<TypeIdentifier, TensorInfoCall> tensor_info_call_registry_{
    {TypeMeta::Id<Tensor>(), GetTensorInfo},
    {TypeMeta::Id<int8::Int8TensorCPU>(), GetInt8TensorInfo},
};

// TODO: Remove this code in a separate diff, since we only have one
// GetTensorInfo function now
TensorInfoCall GetTensorInfoFunction(TypeIdentifier id) {
  auto f = tensor_info_call_registry_.find(id);
  if (f == tensor_info_call_registry_.end()) {
    return nullptr;
  }
  return f->second;
}

void RegisterTensorInfoFunction(TypeIdentifier id, TensorInfoCall c) {
  tensor_info_call_registry_[id] = c;
}

void TensorVectorResize(
    std::vector<Tensor>& tensors,
    int size,
    DeviceType type) {
  tensors.reserve(size);
  for (auto i = 0; i < size; ++i) {
    tensors.emplace_back(type);
  }
}

Tensor empty(at::IntArrayRef dims, at::TensorOptions options) {
  // TODO: merge this with at::empty after Tensor is merged
  auto tensor = Tensor(dims, options.device());
  tensor.raw_mutable_data(options.dtype());
  return tensor;
}

void ReinitializeTensor(
    Tensor* tensor,
    at::IntArrayRef dims,
    at::TensorOptions options) {
  CAFFE_ENFORCE(options.device_opt() != c10::nullopt);
  if (*tensor) {
    // Note: we don't compare device_id here because of the purpose of
    // ReinitializeTensor: https://github.com/pytorch/pytorch/pull/13147
    // In the original code, we don't have device_id defined, therefore, we
    // should not include device_id in the comparison
    if (tensor->GetDeviceType() == options.device().type()) {
      if (tensor->sizes() != dims) {
        // Resize when the dims doesn't match
        tensor->Resize(dims);
      }
      if (tensor->dtype() == options.dtype()) {
        tensor->raw_mutable_data();
      } else {
        // This C10 logging API is not thread-safe, and should not be called here
        // This can lead to a memory corruption in glog.
        // C10_LOG_FIRST_N(WARNING, 1)
        //     << "Changing the data type of Tensor is discouraged."
        //     << " Attempt to change data type from: " << tensor->dtype()
        //     << " to: " << options.dtype();
        // create a new Tensor when the data_type doesn't match
        *tensor = caffe2::empty(dims, options);
      }
      return;
    }
    // create a new Tensor when device doesn't match
  }

  VLOG(1) << "Create new mutable object " << TypeMeta::TypeName<Tensor>()
          << " dims: " << dims;
  *tensor = caffe2::empty(dims, options);
}

void ReinitializeAndCopyFrom(
    Tensor* t,
    at::TensorOptions options,
    const Tensor& src,
    bool async) {
  auto device_type = options.device().type();
  CAFFE_ENFORCE(t != nullptr, "Target tensor ptr is null.");
  if (!*t || device_type != t->GetDeviceType()) {
    *t = Tensor(device_type);
  }
  CAFFE_ENFORCE(
      !t->dtype_initialized() || t->dtype() == src.dtype(),
      "We don't allow a change of data type in ReinitializeAndCopyFrom. Attempt to "
      " change from: ",
      t->dtype(),
      " to: ",
      src.dtype());
  t->CopyFrom(src, async);
}

void Tensor::enforce_invariants() {
  if (impl_.get() == nullptr) {
    throw std::runtime_error("TensorImpl with nullptr is not supported");
  }
  // TODO: only check `!impl_->requires_grad()` after Variable and Tensor are
  // merged
#if !defined(CAFFE2_IS_XPLAT_BUILD) && !defined(C10_MOBILE)
  CAFFE_ENFORCE(
      !(impl_->requires_grad() && at::GradMode::is_enabled()),
      "Caffe2 tensor wrapper doesn't support autograd variables that require grad");
#endif
  CAFFE_ENFORCE_EQ(
      impl_->layout(),
      at::kStrided,
      "Caffe2 tensor wrapper supports only regular non-sparse tensors");
  CAFFE_ENFORCE(
      impl_->is_contiguous(),
      "Caffe2 tensor wrapper supports only contiguous tensors");
}

void Tensor::CopyFrom(const Tensor& src, bool async) {
  // TODO: only check `!impl_->requires_grad()` after Variable and Tensor are
  // merged
#if defined(EXPOSE_C2_OPS) || \
    !defined(CAFFE2_IS_XPLAT_BUILD) && !defined(C10_MOBILE)
  AT_ASSERT(!(impl_->requires_grad() && at::GradMode::is_enabled()));
#endif
  AT_ASSERTM(
      src.impl_->is_contiguous(),
      "Right now only copy of contiguous source Tensor is supported.");
  AT_ASSERTM(
      src.impl_->storage_initialized(),
      "Cannot copy from an uninitialized Tensor");

  if (src.impl_.get() == impl_.get()) {
    return;
  }

  // Test if we need to allocate a new storage
  // Uninitialized storages are guaranteed to be uniquely owned,
  // so we don't need to swap in dst case.
  // If the dtype changed, we need to reallocate storage.
  if (impl_->dtype() != src.impl_->dtype()) {
    // NB: copy preserves device_type
    // This storage will get initialized by the mutable_data call below.
    impl_->set_storage_and_dtype(
        at::Storage::create_legacy(impl_->device_type()), src.impl_->dtype());
  }
  impl_->Resize(src.impl_->sizes());

  if (impl_->numel() > 0) {
    if (impl_->dtype().copy()) {
      AT_ASSERTM(
          impl_->device_type() == ::at::DeviceType::CPU,
          "In CopyFrom source and dest tensors must both be CPU for "
          "non-POD copy, but dest tensor was ",
          impl_->device_type());
      AT_ASSERTM(
          src.impl_->device_type() == ::at::DeviceType::CPU,
          "In CopyFrom source and dest tensors must both be CPU for "
          "non-POD copy, but src tensor was ",
          src.impl_->device_type());
      impl_->dtype().copy()(
          src.impl_->data(),
          impl_->raw_mutable_data(impl_->dtype()),
          impl_->numel());
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
      //
      // note: raw_mutable_data initializes device here
      void* new_data = impl_->raw_mutable_data(impl_->dtype());
      at::CopyBytes(
          impl_->numel() * impl_->itemsize(),
          src.impl_->data(),
          src.impl_->device(),
          new_data,
          impl_->device(),
          async);
    }
  }
}

#if defined(EXPOSE_C2_OPS) || \
    !defined(CAFFE2_IS_XPLAT_BUILD) && !defined(C10_MOBILE)
Tensor::Tensor(at::Tensor tensor) : impl_(tensor.unsafeReleaseIntrusivePtr()) {
  enforce_invariants();
}

Tensor::operator at::Tensor() const& {
  return at::Tensor::wrap_tensor_impl(impl_);
}

Tensor::operator at::Tensor() && {
  return at::Tensor::wrap_tensor_impl(std::move(impl_));
}
#endif

namespace {

struct TensorStatGetter : BlobStatGetter {
  size_t sizeBytes(const Blob& blob) const override {
    const auto& tensor = blob.Get<Tensor>();
    auto nbytes = tensor.nbytes();
    if (nbytes > 0 && tensor.IsType<std::string>()) {
      const auto* data = tensor.data<std::string>();
      for (int i = 0; i < tensor.numel(); ++i) {
        nbytes += data[i].size();
      }
    }
    return nbytes;
  }
};
REGISTER_BLOB_STAT_GETTER(Tensor, TensorStatGetter);
} // namespace

} // namespace caffe2
