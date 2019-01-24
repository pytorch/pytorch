#include <c10/core/StorageImpl.h>

namespace c10 {
  std::atomic<int> global_counter (0);
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
        /*epoch_(global_counter++),*/
        allocator_(allocator) {
    if (numel > 0) {
      if (data_type_.id() == caffe2::TypeIdentifier::uninitialized()) {
        AT_ERROR(
            "Constructing a storage with meta of unknown type and non-zero numel");
      }
    }
    epoch_ = global_counter.load();
    std::cout << "creating storage " << epoch_ << " with size " << numel_ << std::endl;
    global_counter++;
  }

} // namespace c10
