#pragma once

#include <c10/macros/Macros.h>
#include <fstream>
#include <memory>

#include "caffe2/serialize/istream_adapter.h"
#include "caffe2/serialize/read_adapter_interface.h"


namespace caffe2::serialize {

class TORCH_API FileAdapter final : public ReadAdapterInterface {
 public:
  C10_DISABLE_COPY_AND_ASSIGN(FileAdapter);
  explicit FileAdapter(const std::string& file_name);
  size_t size() const override;
  size_t read(uint64_t pos, void* buf, size_t n, const char* what = "")
      const override;
  ~FileAdapter() override;

 private:
  // An RAII Wrapper for a FILE pointer. Closes on destruction.
  struct RAIIFile {
    FILE* fp_;
    explicit RAIIFile(const std::string& file_name);
    ~RAIIFile();
  };

  RAIIFile file_;
  // The size of the opened file in bytes
  uint64_t size_;
};

} // namespace caffe2::serialize
