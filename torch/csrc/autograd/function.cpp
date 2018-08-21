#include "torch/csrc/autograd/function.h"

#include "torch/csrc/autograd/engine.h"
#include "torch/csrc/autograd/variable.h"
#include "torch/csrc/jit/ir.h"

#include <ATen/ATen.h>

#include <algorithm>
#include <cstdint>
#include <memory>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>
#include <deque>

namespace torch { namespace autograd {

/// Monotonically incrementing (thread local!) counter to supply sequence
/// numbers.
thread_local uint64_t Function_next_sequence_nr_ = 0;

uint64_t& Function::get_next_sequence_nr() {
  return Function_next_sequence_nr_;
}

auto Function::name() const -> std::string {
  return at::demangle(typeid(*this).name());
}

AnomalyMetadata* Function::metadata() noexcept {
  if (!anomaly_metadata_) {
    anomaly_metadata_ = Engine::get_default_engine().make_anomaly_metadata();
  }
  return anomaly_metadata_.get();
}

/*
 * Fix for #5534: prevent stack overflow on deletion of deep computation graph
 *
 * Sometimes one can end up with a very big computation graph of Functions
 * and Edges. Each std::shared_ptr<Function> contains a list of Edge, and
 * each Edge contains a std::shared_ptr<Function>. Deleting a
 * std::shared_ptr<Function> can trigger the recursive deletion of other
 * std::shared_ptr<Function>'s: this can stack overflow if the graph
 * is deep enough. Here is an example of such a graph:
 *
 * shared_ptr<Function> -> Edge -> shared_ptr<Function> -> Edge -> ... -> shared_ptr<Function>
 *
 * The solution here is to use a custom deleter with each
 * std::shared_ptr<Function>. The custom deleter keeps track of how many
 * nested deleters it is in. When this number exceeds the maximum allowed
 * depth, the Function* to be deleted are accumulated in a per-thread
 * delete queue and handled by one of the deleters.
 *
 * Note that these custom deleters are NOT necessary for deleting PyFunction.
 * This is because a THPFunction Python object owns a PyFunction that is in a
 * computation graph. When Python objects get recursively destroyed, they
 * are also queued into a delete list. This happens very early for them
 * (at 50 deleters): https://github.com/python/cpython/blob/f320be77ffb73e3b9e7fc98c37b8df3975d84b40/Include/object.h#L1024-L1063
 * so we don't need to worry about them.
 */

thread_local std::deque<Function*> deleteFunctionQueue;
thread_local size_t deleteFunctionRecursionDepth = 0;

/*
 * If this number is set too high, a deep computation graph can still
 * stack overflow. The procedure for setting this number was to
 * 1) find the smallest value that would not guard against stack overflows
 *    on various machines
 * 2) Take the minimum of all such values and subtract some leeway because
 *    the memory of these stack frames will probably grow as time passes.
 * Testing on a few machines machines, the magic numbers were:
 * - Mac OSX (Macbook Pro 15) : ~60000
 * - A beefy Ubuntu 16.04 box : ~15000
 * - Windows AWS instance (g3.4xlarge): variable. My two attempts at different
 *   times have gotten the following numbers: ~8300, 3669
 */
#ifdef _WIN32
constexpr size_t kDeleteFunctionMaxRecursionDepth = 3000;
#else
constexpr size_t kDeleteFunctionMaxRecursionDepth = 10000;
#endif

struct RecursionDepthCounter {
 public:
  explicit RecursionDepthCounter() {
    ++deleteFunctionRecursionDepth;
  }
  ~RecursionDepthCounter() {
    --deleteFunctionRecursionDepth;
  }

  size_t value() {
    return deleteFunctionRecursionDepth;
  }
};

/*
 * Note that the custom deleter deletes in BFS style. Without using
 * the custom deleter, the computation graph is deleted in a DFS style.
 * The BFS deletion is valid (and safe) because if a shared_ptr<Function>
 * 's reference count hits 0, nothing else will access it.
 */
void deleteFunction(Function* function) {
  RecursionDepthCounter recursion_depth;

  if (recursion_depth.value() > kDeleteFunctionMaxRecursionDepth) {
    deleteFunctionQueue.push_back(function);
    return;
  }

  delete function;

  if (deleteFunctionQueue.size() == 0) {
    return;
  }
  if (recursion_depth.value() != kDeleteFunctionMaxRecursionDepth) {
    AT_ERROR("Only one deleter per thread should be able to process "
             "the delete queue. Please open an issue.");
  }
  while (deleteFunctionQueue.size() > 0) {
    auto queued_function = deleteFunctionQueue.front();
    deleteFunctionQueue.pop_front();
    delete queued_function;
  }
}

}} // namespace torch::autograd
