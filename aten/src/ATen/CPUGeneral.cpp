#include <ATen/Config.h>
#include <ATen/CPUGeneral.h>
#include <c10/util/Exception.h>
#include <atomic>
#include <memory>
#include <thread>
#include <cstdint>
#include <mutex>

#ifdef _OPENMP
#include <omp.h>
#endif

#if AT_MKL_ENABLED()
#include <mkl.h>
#endif


namespace at {

// Global value
// This is probably lock-free, but is_lock_free is determined at runtime so we
// can't static_assert. See notes at https://en.cppreference.com/w/cpp/atomic/atomic_is_lock_free
std::atomic<int> num_threads_global(-1);

// Global lock
std::mutex num_threads_global_mutex;

// Thread local value
thread_local int num_threads_thread_local = -1;


static void _set_num_threads_local(int num_threads) {
  if (num_threads_thread_local != num_threads) {
  #ifdef _OPENMP
    omp_set_num_threads(num_threads);
  #endif
  #if AT_MKL_ENABLED()
    mkl_set_num_threads(num_threads);
  #endif
    num_threads_thread_local = num_threads;
  }
}

// NOTE [ Scopes of MKL and OpenMP num_threads values ]
//
// In PyTorch we have one variable to control both MKL and OpenMP num_threads
// value, with setter `torch.set_num_threads` and getter `torch.get_num_threads`.
// This variable is treated as a global setting, controlling all OpenMP and MKL
// operations in the process.
//
// However, in MKL and OpenMP, num_threads actually have different scopes.
//
// MKL: global flag controlling all opertions in the process.
//
// OpenMP:
//   The number of threads (`nthreads-var`) is an internal control variable
//   (ICV). Such ICVs are usually task-specific, i.e., having one copy per task
//   environment. When a task is started, the ICV is inherited from the parent
//   task, and changing it afterwards has no effect on its value in other tasks.
//   In other words, inside a OpenMP parallel region (including a MKL one),
//   inner scope inherits this value from the outer. But for a thread launched
//   via other things like `std::thread`, since it isn't using OpenMP and is not
//   an OpenMP task, the inner scope does not inherit this value from the outer,
//   and, although is intentially not specified in the standard (as of 4.5),
//   empirically instead initializes the value as if it is starting a new
//   process using the default value and the environmental variables (if present).
//
//   See these examples of querying `omp_get_max_threads`:
//     - in an OpenMP region: https://stackoverflow.com/a/21278897
//     - in a `std::thread`: https://coliru.stacked-crooked.com/a/ecadb1755cea68c2
//     - on a Windows: https://software.intel.com/en-us/forums/intel-c-compiler/topic/494242#comment-1773199
//

// Initialize num_threads using env var, and OpenMP/MKL values for this thread.
// This should run **once per thread** before any OpenMP/MKL calls. We achieve
// this in `Context::initThreadIfNeeded`, which calls this via a
// `std::call_once(thread_local_flag, ...)` and is invoked at `globalContext()`.
// MKL:
// https://software.intel.com/en-us/mkl-macos-developer-guide-techniques-to-set-the-number-of-threads
//
// Why do we need to do this?
// See NOTE [ Scopes of MKL and OpenMP num_threads values ]
void init_num_threads_for_this_thread() {
  // Global init flag
  static std::once_flag num_threads_global_init;

  // The first thread reach here infers the number.
  std::call_once(num_threads_global_init,[&] {
    if (num_threads_global == -1) {
      // Use env var if exists
      if (const char *env_p = std::getenv("OMP_NUM_THREADS")) {
        num_threads_global = std::stoi(env_p);
      } else if (const char *env_p = std::getenv("MKL_NUM_THREADS")) {
        num_threads_global = std::stoi(env_p);
      } else {
      #if AT_MKL_ENABLED()
        num_threads_global = mkl_get_max_threads();

        // Because PyTorch uses OpenMP outside of MKL invocations
        // as well, we want this flag to be false, so that
        // threads aren't destroyed and recreated across every
        // MKL / non-MKL boundary of OpenMP usage
        // See https://github.com/pytorch/pytorch/issues/13757
        mkl_set_dynamic(false);
      #elif defined(_OPENMP)
        num_threads_global = omp_get_max_threads();
      #else
        num_threads_global = 1;
      #endif
      }
    }
  });


  // All threads reach here need to synchronize the number.

  // Make sure that MKL and OpenMP use the above inferred value. This is
  // particularly important if we are using both MKL and OpenMP. If the numbers
  // don't match, MKL and our OpenMP-enabled functions will keep changing the
  // size of the OpenMP thread pool, resulting in worse performance (and memory
  // leaks in GCC 5.4)

  _set_num_threads_local(num_threads_global);  // possible race
}

void set_num_threads(int num_threads) {
  AT_CHECK(num_threads >= 1,
           "number of threads must be greater than 0, but got ", num_threads);
#if !defined(_OPENMP) && !AT_MKL_ENABLED()
  if (num_threads != 1) {
    AT_WARN("PyTorch is compiled with neither OpenMP nor MKL. Setting the "
            "number of threads used to a value other than 1 has no effect.")
    return;
  }
#endif
  num_threads_global = num_threads;
  _set_num_threads_local(num_threads);
}

int get_num_threads() {
  AT_ASSERTM(num_threads_global > 0, "num_threads should be initialized");
  _set_num_threads_local(num_threads_global);
  return num_threads_global;
}

}
