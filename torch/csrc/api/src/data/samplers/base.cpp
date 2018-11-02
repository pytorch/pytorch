#include <torch/data/samplers/base.h>

#include <torch/serialize/archive.h>

namespace torch {
namespace data {
namespace samplers {
serialize::OutputArchive& operator<<(
    serialize::OutputArchive& archive,
    const Sampler& sampler) {
  sampler.save(archive);
  return archive;
}

serialize::InputArchive& operator>>(
    serialize::InputArchive& archive,
    Sampler& sampler) {
  sampler.load(archive);
  return archive;
}
} // namespace samplers
} // namespace data
} // namespace torch
