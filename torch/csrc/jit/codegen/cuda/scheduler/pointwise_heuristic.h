#pragma once

#include <torch/csrc/jit/codegen/cuda/scheduler/heuristic.h>

#include <sstream>

namespace torch {
namespace jit {
namespace fuser {
namespace cuda {

// Parameters of the pointwise heuristic to describe the optimial schedule.
// Warning: equal operator is intended for use in caching the kernel associated
// with these pointwise parameters. It does not check if the launch parameters
// are equivelent!
class PointwiseParams : public HeuristicParams {
 public:
  // vectorize if true, otherwise unroll
  bool vectorize = false;

  // Treat pointwise operation as 2-Dimensional, this is the location where we
  // split from left side of the domain to right. i.e. 0 means problem is
  // treated as 1-D, 1 of 3 would mean we treat the first dimension as the outer
  // dimension, and all the others as an inner dimension.
  int break_point = 0;

  // Split block across left and right dimension
  bool split_block = false;

  // Split grid y dimension, if otherwise it would be too large
  bool split_grid_y_dim = false;

  // For many instances having BIDx on the inner most dimension is the most
  // performant parallel binding. However, if we're broadcasting the outer
  // dimension with a large inner dimension, it can be more performant to bind
  // BIDy on the inner most dimension.
  bool flip_grid_binding = false;

  // Unroll or vectorization factor
  size_t unroll_factor = 1;

  using HeuristicParams::HeuristicParams;

  // Warning: Does not check launch parameters!
  bool sameAs(
      const std::shared_ptr<HeuristicParams>& other_base) const override {
    auto other_casted = std::dynamic_pointer_cast<PointwiseParams>(other_base);
    if (other_casted == nullptr) {
      return false;
    }
    const PointwiseParams& other = *other_casted;
    bool attr_equal = other.vectorize == vectorize &&
        other.break_point == break_point && other.split_block == split_block &&
        other.split_grid_y_dim == split_grid_y_dim &&
        other.unroll_factor == unroll_factor &&
        other.flip_grid_binding == flip_grid_binding;
    return attr_equal;
  }

  std::string toString() const override {
    std::stringstream ss;
    ss << "\n===== Pointwise Parameters ========\n"
       << (tag == "" ? "" : "Tag: ") << tag << " Pointwise Characteristics:\n"
       << " Gridx: " << lparams.gdimx() << " BlckY: " << lparams.bdimy()
       << " BlckX: " << lparams.bdimx() << "\n";
    if (break_point) {
      ss << "2D Schedule\n"
         << "  Bcast break point: " << break_point << "\n";
      if (split_block) {
        ss << "Split block into y-dim\n";
      }
      if (split_grid_y_dim) {
        ss << "  Split y grid dim\n";
      }
    }
    if (unroll_factor > 1) {
      if (vectorize) {
        ss << "Vectorize, Factor: " << unroll_factor << "\n";
      } else {
        ss << "Unroll, Factor: " << unroll_factor << "\n";
      }
    }
    if (flip_grid_binding) {
      ss << "Flip BIDx/BIDy bindings\n";
    }
    ss << "====================================\n";
    return ss.str();
  }

  // Warning: Hash is not based on launch parameters!
  size_t hash() const override {
    size_t attr_hash = static_cast<size_t>(vectorize) ^
        static_cast<size_t>(break_point) << 4 ^
        static_cast<size_t>(split_block) << 5 ^
        static_cast<size_t>(split_grid_y_dim) << 6 ^
        static_cast<size_t>(unroll_factor) << 9 ^
        static_cast<size_t>(flip_grid_binding) << 10;
    return attr_hash;
  }

  std::shared_ptr<HeuristicParams> clone() const override {
    return std::make_shared<PointwiseParams>(*this);
  }
};

} // namespace cuda
} // namespace fuser
} // namespace jit
} // namespace torch
