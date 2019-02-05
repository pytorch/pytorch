#include <torch/csrc/CudaIPCTypes.h>
#include <sys/types.h>
#include <unistd.h>

CudaIPCReceivedData::CudaIPCReceivedData(std::shared_ptr<void> shared_ptr)
    : shared_ptr_(std::move(shared_ptr)) {}

int64_t CudaIPCSentData::get() {
  return *(int64_t*)(shared_storage_ptr_.get());
}

CudaIPCSentDataLimbo::~CudaIPCSentDataLimbo() {
  collect();
  if (shared_blocks_.size() > 0) {
    AT_ERROR(
        "Producer process has been terminated before all shared CUDA tensors released.");
  }
}

void CudaIPCSentDataLimbo::collect() {
  std::vector<CudaIPCSentData*> kept_blocks;
  for (auto const& sd : shared_blocks_) {
    if (sd->get() > 0) {
      kept_blocks.push_back(sd);
    } else {
      delete sd;
    }
  }
  shared_blocks_ = kept_blocks;
}

CudaIPCSentDataLimbo CudaIPCSentDataLimbo;

CudaIPCSentData::~CudaIPCSentData() {}

void CudaIPCSentDataLimbo::add(CudaIPCSentData* shared_block) {
  shared_blocks_.push_back(shared_block);
}

CudaIPCSentData::CudaIPCSentData(at::DataPtr shared_storage_ptr)
    : shared_storage_ptr_(std::move(shared_storage_ptr)) {}

void CudaIPCSentDataDelete(void* ptr) {
  auto rc = (CudaIPCSentData*)ptr;
  if (rc->get() > 0) {
    CudaIPCSentDataLimbo.add(rc);
  } else {
    delete (CudaIPCSentData*)ptr;
  }
  CudaIPCSentDataLimbo.collect();
}
