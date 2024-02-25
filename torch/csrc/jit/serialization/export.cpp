#include <torch/csrc/jit/serialization/export.h>

#include <memory>
#include <string>
#include <vector>

namespace torch::jit {

void writeArchiveAndTensors(
    const std::string& archive_name,
    const char* data,
    size_t size,
    const std::vector<at::Tensor>& tensors,
    caffe2::serialize::PyTorchStreamWriter& out) {
  std::string prefix = archive_name + "/";
  size_t i = 0;
  for (const auto& td : tensors) {
    WriteableTensorData writable_td = getWriteableTensorData(td);
    std::string fname = prefix + std::to_string(i++);
    out.writeRecord(fname, writable_td.data(), writable_td.sizeInBytes());
  }
  std::string fname = archive_name + ".pkl";
  out.writeRecord(fname, data, size);
}

} // namespace torch::jit
