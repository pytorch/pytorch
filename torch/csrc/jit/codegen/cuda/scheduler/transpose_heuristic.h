#pragma once

#include <torch/csrc/jit/codegen/cuda/scheduler/heuristic.h>

#include <sstream>

namespace torch {
namespace jit {
namespace fuser {
namespace cuda {

// Parameters of the transpose heuristic to describe the optimial schedule.
// Warning: equal operator is intended for use in caching the kernel associated
// with these reduction parameters. It does not check if the launch parameters
// are equivelent!
class TransposeParams : public HeuristicParams {
 public:
  // Vectorization factor for tensors in the first group
  size_t vectorize_factor1 = 1;

  // Vectorization factor for tensors in the second group
  size_t vectorize_factor2 = 1;

  // TODO: support symbolic tile size
  // https://github.com/csarofeen/pytorch/pull/1854#discussion_r928143729

  // Tile size for the inner most dim of tensors in the first group
  size_t tile_size1 = 32;

  // Tile size for the inner most dim of tensors in the second group
  size_t tile_size2 = 32;

  using HeuristicParams::HeuristicParams;

  // Warning: Does not check launch parameters!
  bool sameAs(
      const std::shared_ptr<HeuristicParams>& other_base) const override {
    auto other_casted = std::dynamic_pointer_cast<TransposeParams>(other_base);
    if (other_casted == nullptr) {
      return false;
    }
    const TransposeParams& other = *other_casted;
    bool attr_equal = other.vectorize_factor1 == vectorize_factor1 &&
        other.vectorize_factor2 == vectorize_factor2 &&
        other.tile_size1 == tile_size1 && other.tile_size2 == tile_size2;
    return attr_equal;
  }

  std::string toString() const override {
    std::stringstream ss;
    ss << "\n===== Transpose Parameters ========\n"
       << (tag == "" ? "" : "Tag: ") << tag << " Transpose Characteristics:\n"
       << " Gridx: " << lparams.gdimx() << " BlckY: " << lparams.bdimy()
       << " BlckX: " << lparams.bdimx() << "\n";
    ss << " input tile size: " << tile_size1 << "\n";
    ss << " output tile size: " << tile_size2 << "\n";
    int elements_per_tile = tile_size1 * tile_size2;
    ss << " elements per tile: " << elements_per_tile << "\n";
    int elements_per_thread =
        elements_per_tile / (lparams.bdimy() * lparams.bdimx());
    ss << " elements per thread: " << elements_per_thread << "\n";
    if (vectorize_factor1 > 1) {
      ss << "Vectorize set 1, Factor: " << vectorize_factor1 << "\n";
    }
    int unroll_factor1 = elements_per_thread / vectorize_factor1;
    if (unroll_factor1 > 1) {
      ss << "Unroll set 1, Factor: " << unroll_factor1 << "\n";
    }
    if (vectorize_factor2 > 1) {
      ss << "Vectorize set 2, Factor: " << vectorize_factor2 << "\n";
    }
    int unroll_factor2 = elements_per_thread / vectorize_factor2;
    if (unroll_factor2 > 1) {
      ss << "Unroll set 2, Factor: " << unroll_factor2 << "\n";
    }
    ss << "====================================\n";
    return ss.str();
  }

  size_t hash() const override {
    size_t attr_hash = vectorize_factor1 ^ (vectorize_factor2 << 16) ^
        (tile_size1 << 32) ^ (tile_size2 << 48);
    return attr_hash;
  }

  std::shared_ptr<HeuristicParams> clone() const override {
    return std::make_shared<TransposeParams>(*this);
  }
};

} // namespace cuda
} // namespace fuser
} // namespace jit
} // namespace torch
