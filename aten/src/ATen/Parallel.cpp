#include <ATen/Parallel.h>

#include <atomic>

#ifdef TH_BLAS_MKL
#include <mkl.h>
#endif

namespace at {

namespace {
std::atomic<size_t> num_threads(1);
}

void init_num_threads() {
#if defined(_OPENMP) && defined(TH_BLAS_MKL)
  // If we are using MKL an OpenMP make sure the number of threads match.
  // Otherwise, MKL and our OpenMP-enabled functions will keep changing the
  // size of the OpenMP thread pool, resulting in worse performance (and memory
  // leaks in GCC 5.4)
  omp_set_num_threads(mkl_get_max_threads());
#endif
#if defined(_OPENMP)
num_threads.store(omp_get_max_threads());
#endif
}

void set_num_threads(size_t nthreads) {
  if (nthreads == 0) {
    return;
  }
#ifdef _OPENMP
  omp_set_num_threads(nthreads);
  num_threads.store(nthreads);
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

size_t get_num_threads() {
  return num_threads.load();
}

}
