#include "caffe2/core/tensor.h"

#include "caffe2/core/blob_stats.h"
#include "caffe2/core/flags.h"

CAFFE2_DEFINE_bool(
    caffe2_keep_on_shrink,
    true,
    "If set, keeps memory when a tensor is shrinking its size.");

CAFFE2_DEFINE_int64(
    caffe2_max_keep_on_shrink_memory,
    LLONG_MAX,
    "The maximum memory in bytes to keep on shrink, if the difference between "
    "tensor sizes is bigger than this then tensor will be reset.");

namespace caffe2 {
// declaring it here instead of context.cc because tensor.h includes context.h
CAFFE_KNOWN_TYPE(Tensor<CPUContext>);

TensorPrinter::TensorPrinter(
    const std::string& tensor_name,
    const std::string& file_name,
    int limit)
    : to_file_(!file_name.empty()),
      limit_(limit ? limit : k_limit_default_),
      tensor_name_(tensor_name) {
  if (to_file_) {
    // We will output to file instead of printing on screen.
    // We will write each individual tensor to its individual file.
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

TensorPrinter::~TensorPrinter() {
  if (log_file_.get()) {
    log_file_->close();
  }
}

static CaffeMap<CaffeTypeId, TypeCall> type_call_registry_ {
  {TypeMeta::Id<Tensor<CPUContext>>(), GetTensorType<CPUContext>}
};

TypeCall GetTypeCallFunction(CaffeTypeId id) {
  auto f = type_call_registry_.find(id);
  if (f == type_call_registry_.end()) {
    return nullptr;
  }
  return f->second;
}

void RegisterTypeCallFunction(CaffeTypeId id, TypeCall c) {
  type_call_registry_[id] = c;
}

static CaffeMap<CaffeTypeId, TensorInfoCall> tensor_info_call_registry_{
    {TypeMeta::Id<Tensor<CPUContext>>(), GetTensorInfo<CPUContext>}};

TensorInfoCall GetTensorInfoFunction(CaffeTypeId id) {
  auto f = tensor_info_call_registry_.find(id);
  if (f == tensor_info_call_registry_.end()) {
    return nullptr;
  }
  return f->second;
}

void RegisterTensorInfoFunction(CaffeTypeId id, TensorInfoCall c) {
  tensor_info_call_registry_[id] = c;
}

namespace {

struct TensorCPUStatGetter : BlobStatGetter {
  size_t sizeBytes(const Blob& blob) const override {
    const auto& tensor = blob.Get<TensorCPU>();
    auto nbytes = tensor.nbytes();
    if (nbytes > 0 && tensor.IsType<std::string>()) {
      const auto* data = tensor.data<std::string>();
      for (size_t i = 0; i < tensor.size(); ++i) {
        nbytes += data[i].size();
      }
    }
    return nbytes;
  }
};
REGISTER_BLOB_STAT_GETTER(TensorCPU, TensorCPUStatGetter);
}

} // namespace caffe2
