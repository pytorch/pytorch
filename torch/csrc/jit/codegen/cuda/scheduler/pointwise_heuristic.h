#pragma once

#include <torch/csrc/jit/codegen/cuda/executor_launch_params.h>

#include <sstream>

namespace torch {
namespace jit {
namespace fuser {
namespace cuda {

// Parameters the Reduction Heuristic Generates to describe the optimial
// schedule. Warning: equal operator is intended for use in caching the kernel
// associated with these reduction parameters. It does not check if the launch
// parameters are equivelent!
class PointwiseParams {
 public:
  // vectorize if true, otherwise unroll
  bool vectorize = false;
  // Unroll or vectorization factor
  int64_t inner_factor = 1;

  std::string tag = "";

  LaunchParams lparams;

  // Warning: Does not check launch parameters!
  bool operator==(const PointwiseParams& other) const {
    bool attr_equal =
        other.vectorize == vectorize && other.inner_factor == inner_factor;
    return attr_equal;
  }

  std::string toString() const {
    std::stringstream ss;
    ss << "\n===== Pointwise Parameters ========\n"
       << (tag == "" ? "" : "Tag: ") << tag << "Pointwise Characteristics:\n"
       << " Gridx: " << lparams.gdimx() << " BlckX: " << lparams.bdimx()
       << "\n";
    if (inner_factor > 1) {
      if (vectorize) {
        ss << "Vectorize, Factor: " << inner_factor << "\n";
      } else {
        ss << "Unroll, Factor: " << inner_factor << "\n";
      }
    }
    ss << "====================================\n";
    return ss.str();
  }
};

// Warning: Hash is not based on launch parameters!
class PointwiseParamsHash {
 public:
  size_t operator()(const PointwiseParams& pp) const {
    constexpr size_t bits = sizeof(std::size_t) * 8;
    size_t attr_hash = static_cast<size_t>(pp.vectorize) << (bits - 1) |
        static_cast<size_t>(pp.inner_factor) << (bits - 3);
    return attr_hash;
  }
};

} // namespace cuda
} // namespace fuser
} // namespace jit
} // namespace torch
