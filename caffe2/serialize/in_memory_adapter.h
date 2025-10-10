#pragma once
#include <caffe2/serialize/read_adapter_interface.h>
#include <cstring>

namespace caffe2 {
namespace serialize {

class MemoryReadAdapter final : public caffe2::serialize::ReadAdapterInterface {
 public:
  explicit MemoryReadAdapter(const void* data, off_t size)
      : data_(data), size_(size) {}

  size_t size() const override {
    return size_;
  }

  size_t read(uint64_t pos, void* buf, size_t n, const char* what = "")
      const override {
    (void)what;
    memcpy(buf, (int8_t*)(data_) + pos, n);
    return n;
  }

 private:
  const void* data_;
  off_t size_;
};

} // namespace serialize
} // namespace caffe2
