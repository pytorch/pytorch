#include <torch/serialize/reader.h>

#include <torch/tensor.h>

#include <iterator>
#include <string>
#include <vector>

namespace torch {
namespace serialize {
void Reader::read(
    const std::string& key,
    std::vector<Tensor>& tensors,
    bool is_buffer) {
  tensors.clear();
  read(key, std::back_inserter(tensors), is_buffer);
}

void Reader::finish() {}
} // namespace serialize
} // namespace torch
