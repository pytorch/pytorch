#include <torch/serialize/writer.h>

#include <torch/tensor.h>

#include <string>
#include <vector>

namespace torch {
namespace serialize {
void Writer::write(
    const std::string& key,
    const std::vector<Tensor>& tensors,
    bool is_buffer) {
  write(key, tensors.begin(), tensors.end(), is_buffer);
}

void Writer::finish() {}
} // namespace serialize
} // namespace torch
