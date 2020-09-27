#pragma once

#include "c10/macros/Macros.h"
#include "caffe2/serialize/read_adapter_interface.h"

namespace caffe2 {
namespace serialize {

// This is a reader implemented for data buffers.
class BufferAdapter final : public ReadAdapterInterface {
 public:
  explicit BufferAdapter(void *buffer, size_t size);
  size_t size() const override;
  size_t read(uint64_t pos,
              void *buf,
              size_t n,
              const char *what = "") const override;
  ~BufferAdapter();

 private:
  void *buffer_;
  size_t size_;
};

}  // namespace serialize
}  // namespace caffe2
