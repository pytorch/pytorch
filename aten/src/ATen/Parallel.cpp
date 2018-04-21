#include <ATen/CPUGeneral.h>
#include <ATen/Parallel.h>
#include <tbb/blocked_range.h>
#include <tbb/parallel_reduce.h>
#include <tbb/partitioner.h>
#include <tbb/tbb.h>
#include <cassert>
#include <thread>

namespace at { namespace internal {

// thread_local variable with internal linkage
// requires no guarding as it's storage duration is defined to be per thread
static thread_local tbb::task_scheduler_init tbbinit(
    tbb::task_scheduler_init::deferred);
// Tracks number of threads uses which TBB doesn't track.
static thread_local int num_threads_ = -1;

// Negative number of threads means default value
void init_tbb_num_threads() {
  static thread_local bool first_call = true;
  int num_threads = at::get_num_threads();
  // In order to have control over the number of threads this function
  // must be called first before any other tbb parallel construct is
  // excercised within a particular thread. Otherwise the default
  // scheduler will be created over which we do not have control.
  // The following code will and must throw an error if tbb has
  // already been initialized before this function was called.
  if (!tbbinit.is_active() && !first_call)
    throw std::runtime_error(
        "tbb initialization failed: scheduler not active after first call");
  if (first_call) {
    if (tbbinit.is_active())
      throw std::runtime_error(
          "tbb initialization failed: scheduler active on first call");
    if (num_threads < 0) {
      int max_threads = tbbinit.default_num_threads();
      tbbinit.initialize(max_threads);
    } else {
      tbbinit.initialize(num_threads);
    }
    first_call = false;
  }
  if (num_threads == 0) {
    // TODO: For PyTorch 0 means 1
    num_threads = 1;
  }
  if (num_threads > 0 && (num_threads_ != num_threads)) {
    tbbinit.terminate();
    tbbinit.initialize(num_threads);
    num_threads_ = num_threads;
  }
}
}} // namespace at::internal
