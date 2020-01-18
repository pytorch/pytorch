#pragma once

#include <cstddef>
#include <cstdint>

#include "c10/macros/Macros.h"
#include "c10/core/Allocator.h"

namespace caffe2 {
namespace serialize {

// this is the interface for the (file/stream/memory) reader in
// PyTorchStreamReader. with this interface, we can extend the support
// besides standard istream
class CAFFE2_API ReadAdapterInterface {
 public:
  virtual size_t size() const = 0;
  virtual size_t read(uint64_t pos, void* buf, size_t n, const char* what = "")
      const = 0;

  virtual bool canMMap() const {
    return false;
  }

  virtual c10::DataPtr mmap(uint64_t pos, size_t n) {
    AT_ERROR("Unimplemented");
  }

  virtual ~ReadAdapterInterface();
};

} // namespace serialize
} // namespace caffe2
