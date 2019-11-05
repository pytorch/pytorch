#pragma once

#include <functional>
#include "c10/macros/Macros.h"
#include "caffe2/serialize/read_adapter_interface.h"

namespace caffe2 {
namespace serialize {

// Read a number of bytes into a buffer and return the number of bytes actually
// read
using ReaderFunc = std::function<size_t(void*, size_t)>;

// Used to seek the buffer to the specified position and returns the position
using SeekerFunc = std::function<size_t(size_t)>;

class CAFFE2_API FuncAdapter final : public ReadAdapterInterface {
 public:
  C10_DISABLE_COPY_AND_ASSIGN(FuncAdapter);
  explicit FuncAdapter(ReaderFunc in, SeekerFunc seeker, size_t size);
  size_t size() const override;
  size_t read(uint64_t pos, void* buf, size_t n, const char* what = "")
      const override;
  ~FuncAdapter();

 private:
  ReaderFunc in_;
  SeekerFunc seeker_;
  size_t size_;
};

} // namespace serialize
} // namespace caffe2
