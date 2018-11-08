#include <torch/types.h>
#include <torch/serialize/archive.h>

namespace torch {
serialize::OutputArchive& operator<<(
    serialize::OutputArchive& archive,
    const Tensor& tensor) {
  archive.write("0", tensor);
  return archive;
}

serialize::InputArchive& operator>>(
    serialize::InputArchive& archive,
    Tensor& tensor) {
  archive.read("0", tensor);
  return archive;
}
} // namespace torch
