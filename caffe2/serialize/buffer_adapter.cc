#include "buffer_adapter.h"
#include <c10/util/Exception.h>
#include "caffe2/serialize/istream_adapter.h"

#include <cstring>

namespace caffe2 {
namespace serialize {

BufferAdapter::BufferAdapter(void *buffer, size_t size)
    : buffer_(buffer), size_(size)
{
}

size_t BufferAdapter::size() const
{
  return size_;
}

size_t BufferAdapter::read(uint64_t pos,
                           void *buf,
                           size_t n,
                           const char *what) const
{
  if (!buffer_) {
    AT_ERROR("buffer reader failed: ", what, ".");
  }
  void *start = static_cast<char *>(buffer_) + pos;
  size_t bytes = std::min<size_t>(size_ - pos, n);
  std::memcpy(buf, start, bytes);
  return bytes;
}

BufferAdapter::~BufferAdapter() {}

}  // namespace serialize
}  // namespace caffe2
