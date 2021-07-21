#include "caffe2/serialize/file_adapter.h"
#include <c10/util/Exception.h>
#include <cstdio>

#include "caffe2/core/common.h"

namespace caffe2 {
namespace serialize {

// FileAdapter directly calls C file API.
FileAdapter::FileAdapter(const std::string& file_name) {
  fp_ = fopen(file_name.c_str(), "rb");
  if (fp_ == nullptr) {
    AT_ERROR("open file failed, file path: ", file_name);
  }
  fseek(fp_, 0L, SEEK_END);
  size_ = ftell(fp_);
  rewind(fp_);
}

size_t FileAdapter::size() const {
  return size_;
}

size_t FileAdapter::read(uint64_t pos, void* buf, size_t n, const char* what)
    const {
  pos = pos < size_ ? pos : size_;
  if (pos + n >= size_) {
    n = size_ - pos;
  }
  fseek(fp_, pos, SEEK_SET);
  return fread(buf, 1, n, fp_);
}

FileAdapter::~FileAdapter() {
  fclose(fp_);
}

} // namespace serialize
} // namespace caffe2
