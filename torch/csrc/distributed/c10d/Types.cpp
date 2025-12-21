#include <torch/csrc/distributed/c10d/Types.hpp>

namespace c10d {

bool isComplexViewAsRealAllowed(const ReduceOp& reduceOp) {
  switch (reduceOp) {
    // NOLINTNEXTLINE(bugprone-branch-clone)
    case ReduceOp::SUM:
      return true;
    case ReduceOp::AVG:
      return true;
    case ReduceOp::PREMUL_SUM:
      return true;
    case ReduceOp::UNUSED:
      return true;
    default:
      return false;
  }
  return false;
}

} // namespace c10d
