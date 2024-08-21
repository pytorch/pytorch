#pragma once
#include <sys/types.h>
#include <torch/csrc/api/include/torch/serialize/read_adapter_interface.h>

#include <cstring>

namespace torch::serialize {

class MemoryReadAdapter final : public torch::serialize::ReadAdapterInterface {
 public:
  explicit MemoryReadAdapter(const void* data, off_t size)
      : data_(data), size_(size) {}

  size_t size() const override {
    return size_;
  }

  size_t read(
      uint64_t pos,
      void* buf,
      size_t n,
      [[maybe_unused]] const char* what = "") const override {
    memcpy(buf, (int8_t*)(data_) + pos, n);
    return n;
  }

 private:
  const void* data_;
  off_t size_;
};

} // namespace torch::serialize
