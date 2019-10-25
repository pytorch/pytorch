#pragma once

#include <functional>
#include "c10/macros/Macros.h"
#include "caffe2/serialize/read_adapter_interface.h"

namespace caffe2 {
namespace serialize {

class CAFFE2_API FuncAdapter final : public ReadAdapterInterface {
 public:
  C10_DISABLE_COPY_AND_ASSIGN(FuncAdapter);
  explicit FuncAdapter(std::function<size_t(char*, size_t)> in);
  size_t size() const override;
  size_t read(uint64_t pos, void* buf, size_t n, const char* what = "")
      const override;
  ~FuncAdapter();

 private:
  std::function<size_t(char*, size_t)> in_;
};

} // namespace serialize
} // namespace caffe2
