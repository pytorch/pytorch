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

void load(Tensor& tensor, InputArchive& archive) {
  archive.read("0", tensor);
}

void save(const optim::Optimizer& optimizer, OutputArchive& archive) {
  optimizer.save(archive);
}

void load(optim::Optimizer& optimizer, InputArchive& archive) {
  optimizer.load(archive);
}
} // namespace serialize
} // namespace torch
