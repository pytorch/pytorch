#include "lazy_tensors/computation_client/ltc_util.h"

#include "lazy_tensors/computation_client/util.h"
#include "lazy_tensors/shape_util.h"

namespace lazy_tensors {
namespace util {
namespace {

hash_t SingleShapeHash(const Shape& shape, hash_t seed) {
  for (auto dim : shape.layout().minor_to_major()) {
    seed = HashCombine(seed, dim);
  }
  for (auto dim : shape.dimensions()) {
    seed = HashCombine(seed, dim);
  }
  return HashCombine(seed, static_cast<int>(shape.element_type()));
}

}  // namespace

// The hash is deterministic to enable easier debugging between separate runs.
hash_t ShapeHash(const Shape& shape) {
  hash_t hash = 0xa5d2d6916;
  ShapeUtil::ForEachSubshape(shape,
                             [&](const Shape& subshape, const ShapeIndex&) {
                               hash = SingleShapeHash(subshape, hash);
                             });
  return hash;
}

}  // namespace util
}  // namespace lazy_tensors
