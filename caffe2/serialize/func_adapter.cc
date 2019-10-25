#include "caffe2/serialize/func_adapter.h"
#include <c10/util/Exception.h>

namespace caffe2 {
namespace serialize {

FuncAdapter::FuncAdapter(std::function<size_t(char*, size_t)> in) : in_(in) {}

size_t FuncAdapter::size() const {
//   auto prev_pos = istream_->tellg();
//   validate("getting the current position");
//   istream_->seekg(0, istream_->end);
//   validate("seeking to end");
//   auto result = istream_->tellg();
//   validate("getting size");
//   istream_->seekg(prev_pos);
//   validate("seeking to the original position");
  return 100000;
  return 0;
}

size_t FuncAdapter::read(uint64_t pos, void* buf, size_t n, const char* what)
    const {
  return in_(reinterpret_cast<char*>(buf), n);
}

FuncAdapter::~FuncAdapter() {}

} // namespace serialize
} // namespace caffe2
