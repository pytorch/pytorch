#pragma once

#include <fstream>
#include <memory>

#include "c10/macros/Macros.h"
#include "caffe2/serialize/istream_adapter.h"
#include "caffe2/serialize/read_adapter_interface.h"

namespace caffe2 {
namespace serialize {

struct MMapFile;

class CAFFE2_API FileAdapter final : public ReadAdapterInterface {
 public:
  C10_DISABLE_COPY_AND_ASSIGN(FileAdapter);
  explicit FileAdapter(const std::string& file_name);
  size_t size() const override;
  size_t read(uint64_t pos, void* buf, size_t n, const char* what = "")
      const override;
  ~FileAdapter();
  bool canMMap() const override;
  c10::DataPtr mmap(uint64_t pos, size_t n) override;

 private:
  std::string file_name_;
  std::ifstream file_stream_;
  std::unique_ptr<IStreamAdapter> istream_adapter_;
  std::shared_ptr<MMapFile> mmap_file_;
};

} // namespace serialize
} // namespace caffe2
