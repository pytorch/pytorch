#include "lazy_tensors/shape_util.h"

#include "c10/util/logging_is_not_google_glog.h"
#include "lazy_tensors/core/platform/errors.h"
#include "lazy_tensors/core/platform/hash.h"

namespace torch {
namespace lazy {

torch::lazy::hash_t SingleShapeHash(const torch::lazy::Shape& shape, torch::lazy::hash_t seed) {
  for (auto dim : shape.sizes()) {
    seed = HashCombine(seed, (uint64_t)dim);
  }
  return HashCombine(seed, static_cast<int>(shape.scalar_type()));
}

// The hash is deterministic to enable easier debugging between separate runs.
torch::lazy::hash_t ShapeHash(const torch::lazy::Shape& shape) {
  torch::lazy::hash_t hash = (uint32_t)0xa5d2d6916;
  return torch::lazy::SingleShapeHash(shape, hash);
}


torch::lazy::hash_t Hash(const torch::lazy::Shape& shape) {
  return ShapeHash(shape);
}

}  // namespace lazy
}  // namespace torch

namespace lazy_tensors {

/*static*/ size_t ShapeUtil::Hash(const torch::lazy::Shape& shape) {
  using lazy_tensors::hash;
  using lazy_tensors::Hash64Combine;

  size_t hash_value = hash<c10::ScalarType>()(shape.scalar_type());

  for (int i = 0; i < shape.dim(); ++i) {
    hash_value =
        Hash64Combine(hash_value, hash<int64_t>()(shape.size(i)));
  }
  return hash_value;
}

}  // namespace lazy_tensors
