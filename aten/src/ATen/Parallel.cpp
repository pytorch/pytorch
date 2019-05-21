#include <ATen/Parallel.h>

#include <ATen/Config.h>
#include <ATen/Version.h>

#include <atomic>
#include <sstream>
#include <thread>

#ifdef TH_BLAS_MKL
#include <mkl.h>
#endif

namespace at {

namespace {
const int NOT_SET = -1;
const int CONSUMED = -2;

// Number of threads set by the user
std::atomic<int> num_threads{NOT_SET};

// Number of inter-op threads set by the user;
// NOT_SET -> positive value -> CONSUMED
// (CONSUMED - thread pool is initialized)
// or
// NOT_SET -> CONSUMED
std::atomic<int> num_interop_threads{NOT_SET};

// thread pool global instance is hidden,
// users should use at::launch and get/set_num_interop_threads interface
TaskThreadPoolBase& get_pool() {
  static std::shared_ptr<TaskThreadPoolBase> pool =
      ThreadPoolRegistry()->Create(
          "C10",
          /* device_id */ 0,
          /* pool_size */ num_interop_threads.exchange(CONSUMED),
          /* create_new */ true);
  return *pool;
}

 // Factory function for ThreadPoolRegistry
std::shared_ptr<TaskThreadPoolBase> create_c10_threadpool(
    int device_id,
    int pool_size,
    bool create_new) {
  // For now, the only accepted device id is 0
  AT_CHECK(device_id == 0);
  // Create new thread pool
  AT_CHECK(create_new);
  return std::make_shared<PTThreadPool>(pool_size);
}

}

void init_num_threads() {
  auto nthreads = num_threads.load();
  if (nthreads > 0) {
    set_num_threads(nthreads);
  } else {
#if defined(_OPENMP) && defined(TH_BLAS_MKL)
  // If we are using MKL an OpenMP make sure the number of threads match.
  // Otherwise, MKL and our OpenMP-enabled functions will keep changing the
  // size of the OpenMP thread pool, resulting in worse performance (and memory
  // leaks in GCC 5.4)
  omp_set_num_threads(mkl_get_max_threads());
#endif
  }
}

void set_num_threads(int nthreads) {
  AT_CHECK(nthreads > 0, "Expected positive number of threads");

  num_threads.store(nthreads);
#ifdef _OPENMP
  omp_set_num_threads(nthreads);
#endif
#ifdef TH_BLAS_MKL
  mkl_set_num_threads(nthreads);

  // because PyTorch uses OpenMP outside of MKL invocations
  // as well, we want this flag to be false, so that
  // threads aren't destroyed and recreated across every
  // MKL / non-MKL boundary of OpenMP usage
  // See https://github.com/pytorch/pytorch/issues/13757
  mkl_set_dynamic(false);
#endif
}

// Explicitly calling omp_get_max_threads() as the size of the parallel
// region might be different in the new thread;
// Use init_num_threads() during thread initialization to ensure
// consistent size of parallel region in different threads
int get_num_threads() {
#ifdef _OPENMP
  return omp_get_max_threads();
#else
  return 1;
#endif
}

namespace {
const char* get_env_var(const char* var_name) {
  const char* value = std::getenv(var_name);
  return value ? value : "[not set]";
}
}

std::string get_parallel_info() {
  std::ostringstream ss;

  ss << "ATen/Parallel:\n\tat::get_num_threads() : "
     << at::get_num_threads() << std::endl;

  ss << at::get_openmp_version() << std::endl;
#ifdef _OPENMP
  ss << "\tomp_get_max_threads() : " << omp_get_max_threads() << std::endl;
#endif

  ss << at::get_mkl_version() << std::endl;
#ifdef TH_BLAS_MKL
  ss << "\tmkl_get_max_threads() : " << mkl_get_max_threads() << std::endl;
#endif

  ss << at::get_mkldnn_version() << std::endl;

  ss << "std::thread::hardware_concurrency() : "
     << std::thread::hardware_concurrency() << std::endl;

  ss << "Environment variables:" << std::endl;
  ss << "\tOMP_NUM_THREADS : " << get_env_var("OMP_NUM_THREADS") << std::endl;
  ss << "\tMKL_NUM_THREADS : " << get_env_var("MKL_NUM_THREADS") << std::endl;

  return ss.str();
}

PTThreadPool::PTThreadPool(
    int pool_size,
    int numa_node_id)
    : c10::ThreadPool(pool_size, numa_node_id) {}

void PTThreadPool::init_thread() {
  c10::setThreadName("PTThreadPool");
  at::init_num_threads();
}

C10_REGISTER_CREATOR(ThreadPoolRegistry, C10, create_c10_threadpool);

void set_num_interop_threads(int nthreads) {
  AT_CHECK(nthreads > 0, "Expected positive number of threads");

  int no_value = NOT_SET;
  AT_CHECK(num_interop_threads.compare_exchange_strong(no_value, nthreads),
      "Error: cannot set number of interop threads after parallel work "
      "has started or set_num_interop_threads called");
}

int get_num_interop_threads() {
  int nthreads = num_interop_threads.load();
  if (nthreads > 0) {
    return nthreads;
  } else if (nthreads == NOT_SET) {
    // return default value
    return TaskThreadPoolBase::defaultNumThreads();
  } else {
    return get_pool().size();
  }
}

void launch(const std::function<void()>& func) {
  get_pool().run(func);
}

} // namespace at
