#pragma once

#include <c10/util/hash.h>
#include <torch/csrc/jit/codegen/cuda/scheduler/heuristic.h>
#include <torch/csrc/jit/codegen/cuda/utils.h>

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
  static constexpr size_t getMaxThreadsPerBlock() {
    return 128;
  }

  // See note [Supporting small transpose dimensions], all dims are positions in
  // reference1
  std::vector<std::pair<size_t, size_t>> split_before_tiling = {};
  std::vector<size_t> dims_merged_with_1 = {};
  std::vector<size_t> dims_merged_with_2 = {};

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
    bool attr_equal = other.split_before_tiling == split_before_tiling &&
        other.dims_merged_with_1 == dims_merged_with_1 &&
        other.dims_merged_with_2 == dims_merged_with_2 &&
        other.vectorize_factor1 == vectorize_factor1 &&
        other.vectorize_factor2 == vectorize_factor2 &&
        other.tile_size1 == tile_size1 && other.tile_size2 == tile_size2;
    return attr_equal;
  }

  std::string toString() const override {
    std::stringstream ss;
    ss << "\n===== Transpose Parameters ========\n"
       << (tag == "" ? "" : "Tag: ") << tag << " Transpose Characteristics:\n"
       << " Gridx: " << lparams.gdimx() << " BlckX: " << lparams.bdimx()
       << "\n";
    ss << " input tile size: " << tile_size1 << "\n";
    ss << " output tile size: " << tile_size2 << "\n";
    int elements_per_tile = tile_size1 * tile_size2;
    ss << " elements per tile: " << elements_per_tile << "\n";
    int elements_per_thread = elements_per_tile / lparams.bdimx();
    ss << " elements per thread: " << elements_per_thread << "\n";
    if (vectorize_factor1 > 1) {
      ss << "Vectorize group 1, Factor: " << vectorize_factor1 << "\n";
    }
    int unroll_factor1 = elements_per_thread / vectorize_factor1;
    if (unroll_factor1 > 1) {
      ss << "Unroll group 1, Factor: " << unroll_factor1 << "\n";
    }
    if (vectorize_factor2 > 1) {
      ss << "Vectorize group 2, Factor: " << vectorize_factor2 << "\n";
    }
    int unroll_factor2 = elements_per_thread / vectorize_factor2;
    if (unroll_factor2 > 1) {
      ss << "Unroll group 2, Factor: " << unroll_factor2 << "\n";
    }
    if (!split_before_tiling.empty() || !dims_merged_with_1.empty() ||
        !dims_merged_with_2.empty()) {
      ss << "Virtual inner-most dim:\n";
      if (!split_before_tiling.empty()) {
        ss << "  ";
        bool first = true;
        for (auto pair : split_before_tiling) {
          if (!first) {
            ss << ", ";
          }
          first = false;
          ss << "split(" << pair.first << ", " << pair.second << ")";
        }
        ss << "\n";
      }
      if (!dims_merged_with_1.empty()) {
        ss << "  merge ";
        bool first = true;
        for (auto dim : dims_merged_with_1) {
          if (!first) {
            ss << ", ";
          }
          first = false;
          ss << dim;
        }
        ss << " with innermost1\n";
      }
      if (!dims_merged_with_2.empty()) {
        ss << "  merge ";
        bool first = true;
        for (auto dim : dims_merged_with_2) {
          if (!first) {
            ss << ", ";
          }
          first = false;
          ss << dim;
        }
        ss << " with innermost2\n";
      }
    }
    ss << "====================================\n";
    return ss.str();
  }

  size_t hash() const override {
    return c10::get_hash(
        split_before_tiling,
        dims_merged_with_1,
        dims_merged_with_2,
        vectorize_factor1,
        vectorize_factor2,
        tile_size1,
        tile_size2);
  }

  std::shared_ptr<HeuristicParams> clone() const override {
    return std::make_shared<TransposeParams>(*this);
  }

  int getThreadsPerBlock() const {
    size_t tile_vectors1 = ceilDiv(tile_size1 * tile_size2, vectorize_factor1);
    size_t tile_vectors2 = ceilDiv(tile_size1 * tile_size2, vectorize_factor2);
    size_t tile_vectors = std::min(tile_vectors1, tile_vectors2);
    return std::min(getMaxThreadsPerBlock(), tile_vectors);
  }
};

} // namespace cuda
} // namespace fuser
} // namespace jit
} // namespace torch
