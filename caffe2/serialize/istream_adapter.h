#pragma once

#include <istream>

#include "c10/macros/Macros.h"
#include "caffe2/serialize/read_adapter_interface.h"


namespace caffe2::serialize {

// this is a reader implemented by std::istream
class TORCH_API IStreamAdapter final : public ReadAdapterInterface {
 public:
  C10_DISABLE_COPY_AND_ASSIGN(IStreamAdapter);
  explicit IStreamAdapter(std::istream* istream);
  size_t size() const override;
  size_t read(uint64_t pos, void* buf, size_t n, const char* what = "")
      const override;
  ~IStreamAdapter() override;

 private:
  std::istream* istream_;
  void validate(const char* what) const;
};

} // namespace caffe2::serialize
