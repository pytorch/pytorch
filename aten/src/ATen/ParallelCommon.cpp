#include <ATen/Parallel.h>

#include <ATen/Config.h>
#include <ATen/Version.h>

#include <sstream>
#include <thread>

#ifdef TH_BLAS_MKL
#include <mkl.h>
#endif

#ifdef _OPENMP
#include <omp.h>
#endif

namespace at {

namespace {

const char* get_env_var(const char* var_name) {
  const char* value = std::getenv(var_name);
  return value ? value : "[not set]";
}

} // namespace

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

} // namespace at
