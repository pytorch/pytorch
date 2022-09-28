#pragma once

#include <torch/csrc/jit/codegen/cuda/inlining.h>
#include <torch/csrc/jit/codegen/cuda/root_domain_map.h>
#include <torch/csrc/jit/codegen/cuda/transform_replay.h>

#include <c10/macros/Export.h>
#include <c10/util/Exception.h>

#include <deque>
#include <unordered_map>
#include <unordered_set>
#include <vector>

namespace torch {
namespace jit {
namespace fuser {
namespace cuda {

class TensorDomain;
class TensorView;

struct ComputeAt {
 public:
  // Runs the compute at pass making producer look like consumer, computing
  // producer relative to consumer
  static void runAt(
      TensorView* producer,
      TensorView* consumer,
      int64_t consumer_position,
      ComputeAtMode mode = ComputeAtMode::Standard);

  // Runs the compute with pass making consumer look like producer, computing
  // producer relative to consumer
  static void runWith(
      TensorView* producer,
      TensorView* consumer,
      int64_t producer_position,
      ComputeAtMode mode = ComputeAtMode::Standard);
};

} // namespace cuda
} // namespace fuser
} // namespace jit
} // namespace torch
