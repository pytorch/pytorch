#include <torch/serialize.h>

#include <torch/optim/optimizer.h>
#include <torch/serialize/reader.h>
#include <torch/serialize/writer.h>
#include <torch/tensor.h>

#include <utility>

namespace torch {
namespace serialize {
void save(const Tensor& tensor, serialize::Writer& writer) {
  writer.write("0", tensor);
}

void save(const optim::Optimizer& optimizer, serialize::Writer& writer) {
  optimizer.save(writer);
}

void load(Tensor& tensor, serialize::Reader& reader) {
  reader.read("0", tensor);
}

void load(optim::Optimizer& optimizer, serialize::Reader& reader) {
  optimizer.load(reader);
}
} // namespace serialize
} // namespace torch
