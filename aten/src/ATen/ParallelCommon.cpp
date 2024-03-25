#include <ATen/Parallel.h>

#include <ATen/Config.h>
#include <ATen/PTThreadPool.h>
#include <ATen/Version.h>

#include <sstream>
#include <thread>

#if AT_MKL_ENABLED()
#include <mkl.h>
#endif

#ifdef _OPENMP
#include <omp.h>
#endif

namespace at {

namespace {

const char* get_env_var(
    const char* var_name, const char* def_value = nullptr) {
  const char* value = std::getenv(var_name);
  return value ? value : def_value;
}

size_t get_env_num_threads(const char* var_name, size_t def_value = 0) {
  try {
    if (auto* value = std::getenv(var_name)) {
      int nthreads = std::stoi(value);
      TORCH_CHECK(nthreads > 0);
      return nthreads;
    }
  } catch (const std::exception& e) {
    std::ostringstream oss;
    oss << "Invalid " << var_name << " variable value, " << e.what();
    TORCH_WARN(oss.str());
  }
  return def_value;
}

} // namespace

std::string get_parallel_info() {
  std::ostringstream ss;

  ss << "ATen/Parallel:\n\tat::get_num_threads() : "
     << at::get_num_threads() << '\n';
  ss << "\tat::get_num_interop_threads() : "
     << at::get_num_interop_threads() << '\n';

  ss << at::get_openmp_version() << '\n';
#ifdef _OPENMP
  ss << "\tomp_get_max_threads() : " << omp_get_max_threads() << '\n';
#endif

  ss << at::get_mkl_version() << '\n';
#if AT_MKL_ENABLED()
  ss << "\tmkl_get_max_threads() : " << mkl_get_max_threads() << '\n';
#endif

  ss << at::get_mkldnn_version() << '\n';

  ss << "std::thread::hardware_concurrency() : "
     << std::thread::hardware_concurrency() << '\n';

  ss << "Environment variables:" << '\n';
  ss << "\tOMP_NUM_THREADS : "
     << get_env_var("OMP_NUM_THREADS", "[not set]") << '\n';
  ss << "\tMKL_NUM_THREADS : "
     << get_env_var("MKL_NUM_THREADS", "[not set]") << '\n';

  ss << "ATen parallel backend: ";
  #if AT_PARALLEL_OPENMP
  ss << "OpenMP";
  #elif AT_PARALLEL_NATIVE
  ss << "native thread pool";
  #elif AT_PARALLEL_NATIVE_TBB
  ss << "native thread pool and TBB";
  #endif
  #ifdef C10_MOBILE
  ss << " [mobile]";
  #endif
  ss << '\n';

  #if AT_EXPERIMENTAL_SINGLE_THREAD_POOL
  ss << "Experimental: single thread pool" << std::endl;
  #endif

  return ss.str();
}

int intraop_default_num_threads() {
#ifdef C10_MOBILE
  // Intraop thread pool size should be determined by mobile cpuinfo.
  // We should hook up with the logic in caffe2/utils/threadpool if we ever need
  // call this API for mobile.
  TORCH_CHECK(false, "Undefined intraop_default_num_threads on mobile.");
#else
  size_t nthreads = get_env_num_threads("OMP_NUM_THREADS", 0);
  nthreads = get_env_num_threads("MKL_NUM_THREADS", nthreads);
  if (nthreads == 0) {
#if defined(FBCODE_CAFFE2) && defined(__aarch64__)
    nthreads = 1;
#else
    nthreads = TaskThreadPoolBase::defaultNumThreads();
#endif
  }
  return static_cast<int>(nthreads);
#endif
}

} // namespace at
