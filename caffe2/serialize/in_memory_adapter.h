#pragma once
#include <caffe2/serialize/read_adapter_interface.h>
#include <sys/types.h>
#include <cstring>

namespace caffe2::serialize {

class MemoryReadAdapter final : public caffe2::serialize::ReadAdapterInterface {
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
      const char* what [[maybe_unused]] = "") const override {
    memcpy(buf, (int8_t*)(data_) + pos, n);
    return n;
  }

 private:
  const void* data_;
  off_t size_{};
};

} // namespace caffe2::serialize
