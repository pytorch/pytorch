#include "caffe2/core/blob_stats.h"

namespace caffe2 {

const BlobStatGetter* BlobStatRegistry::get(CaffeTypeId id) {
  auto it = map_.find(id);
  if (it == map_.end()) {
    return nullptr;
  }
  return it->second.get();
}

BlobStatRegistry& BlobStatRegistry::instance() {
  static BlobStatRegistry registry;
  return registry;
}

void BlobStatRegistry::doRegister(
    CaffeTypeId id,
    std::unique_ptr<BlobStatGetter>&& v) {
  // don't use CAFFE_ENFORCE_EQ to avoid static initialization order fiasco.
  if (map_.count(id) > 0) {
    throw std::runtime_error("BlobStatRegistry: Type already registered.");
  }
  map_[id] = std::move(v);
}

namespace BlobStat {

size_t sizeBytes(const Blob& blob) {
  auto* p = BlobStatRegistry::instance().get(blob.meta().id());
  return p ? p->sizeBytes(blob) : 0;
}

} // namespace BlobStats
}
