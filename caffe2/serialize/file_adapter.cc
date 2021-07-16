#include "caffe2/serialize/file_adapter.h"
#include <c10/util/Exception.h>
#include "caffe2/core/common.h"

namespace caffe2 {
namespace serialize {

FileAdapter::FileAdapter(const std::string& file_name) {
  file_stream_.open(file_name, std::ifstream::in | std::ifstream::binary);
  if (!file_stream_) {
    AT_ERROR("open file failed, file path: ", file_name);
  }
  istream_adapter_ = std::make_unique<IStreamAdapter>(&file_stream_);
}

size_t FileAdapter::size() const {
  return istream_adapter_->size();
}

size_t FileAdapter::read(uint64_t pos, void* buf, size_t n, const char* what)
    const {
  return istream_adapter_->read(pos, buf, n, what);
}

// NOLINTNEXTLINE(modernize-use-equals-default)
FileAdapter::~FileAdapter() {}

} // namespace serialize
} // namespace caffe2
