#include <torch/serialize.h>

#include <torch/optim/optimizer.h>
#include <torch/serialize/archive.h>
#include <torch/tensor.h>

#include <string>
#include <utility>

namespace torch {
namespace serialize {
void save(const Tensor& tensor, OutputArchive& archive) {
  archive.write("0", tensor);
}

void save(const optim::Optimizer& optimizer, OutputArchive& archive) {
  optimizer.save(archive);
}

void load(optim::Optimizer& optimizer, InputArchive& archive) {
  optimizer.load(archive);
}
} // namespace serialize

Tensor load(const std::string& filename) {
  serialize::InputArchive archive = serialize::load_from_file(filename);
  Tensor tensor;
  archive.read("0", tensor);
  return tensor;
}
} // namespace torch
