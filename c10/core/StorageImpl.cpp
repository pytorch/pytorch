#include <c10/core/StorageImpl.h>

namespace c10 {

StorageImpl::StorageImpl(
    caffe2::TypeMeta data_type,
    int64_t numel,
    at::DataPtr data_ptr,
    at::Allocator* allocator,
    bool resizable)
    : data_type_(data_type),
      data_ptr_(std::move(data_ptr)),
      numel_(numel),
      resizable_(resizable),
      allocator_(allocator),
      received_cuda_(false) {
  if (numel > 0) {
    if (data_type_.id() == caffe2::TypeIdentifier::uninitialized()) {
      AT_ERROR(
          "Constructing a storage with meta of unknown type and non-zero numel");
    }
  }
}

void StorageImpl::reset() {
  data_ptr_.clear();
  numel_ = 0;
}

StorageImpl::~StorageImpl() {}

void StorageImpl::release_resources() {
  data_ptr_.clear();
}

void StorageImpl::UniqueStorageShareExternalPointer(
    at::DataPtr&& data_ptr,
    const caffe2::TypeMeta& data_type,
    size_t capacity) {
  data_type_ = data_type;
  // TODO: Use CAFFE_ENFORCE_WITH_CALLER equivalent
  // For now causes lots of redefine issues if caffe2/core/logging.h is used
  if (data_type_.id() == caffe2::TypeIdentifier::uninitialized()) {
    AT_ERROR(
        "To share with a raw external pointer you need to have meta "
        "already set.");
  }
  data_ptr_ = std::move(data_ptr);
  // NOTE: data_type might change and so it's also possible that capacity
  // might not be divisible by itemsize. There is no way for us to keep track
  // of the exact capacity if we're not explicity storing is. More conrectely
  // capacity() might not return the value that was set here, if itemsize does
  // not evenly divide it.
  numel_ = capacity / data_type_.itemsize();
}

at::DataPtr StorageImpl::set_data_ptr(at::DataPtr&& data_ptr) {
  std::swap(data_ptr_, data_ptr);
  return std::move(data_ptr);
}

} // namespace c10
