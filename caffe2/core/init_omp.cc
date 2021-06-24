// NOLINTNEXTLINE(modernize-deprecated-headers)
#include <stdlib.h>

#include "caffe2/core/common.h"

#ifdef _OPENMP
#include "caffe2/core/common_omp.h"
#endif  // _OPENMP

#ifdef CAFFE2_USE_MKL
#include <mkl.h>
#endif // CAFFE2_USE_MKL

#include "caffe2/core/init.h"

// NOLINTNEXTLINE(cppcoreguidelines-avoid-non-const-global-variables)
C10_DEFINE_int(
    caffe2_omp_num_threads,
    0,
    "The number of openmp threads. 0 to use default value. "
    "Does not have effect if OpenMP is disabled.");
// NOLINTNEXTLINE(cppcoreguidelines-avoid-non-const-global-variables)
C10_DEFINE_int(
    caffe2_mkl_num_threads,
    0,
    "The number of mkl threads. 0 to use default value. If set, "
    "this overrides the caffe2_omp_num_threads flag if both are set. "
    "Does not have effect if MKL is not used.");

namespace caffe2 {

#ifdef _OPENMP
bool Caffe2SetOpenMPThreads(int*, char***) {
  if (!getenv("OMP_NUM_THREADS")) {
    // OMP_NUM_THREADS not passed explicitly, so *disable* OMP by
    // default. The user can use the CLI flag to override.
    VLOG(1) << "OMP_NUM_THREADS not passed, defaulting to 1 thread";
    omp_set_num_threads(1);
  }

  if (FLAGS_caffe2_omp_num_threads > 0) {
    VLOG(1) << "Setting omp_num_threads to " << FLAGS_caffe2_omp_num_threads;
    omp_set_num_threads(FLAGS_caffe2_omp_num_threads);
  }
  VLOG(1) << "Caffe2 running with " << omp_get_max_threads() << " OMP threads";
  return true;
}
REGISTER_CAFFE2_INIT_FUNCTION(Caffe2SetOpenMPThreads,
                              &Caffe2SetOpenMPThreads,
                              "Set OpenMP threads.");
#endif // _OPENMP

#ifdef CAFFE2_USE_MKL
bool Caffe2SetMKLThreads(int*, char***) {
  if (!getenv("MKL_NUM_THREADS")) {
    VLOG(1) << "MKL_NUM_THREADS not passed, defaulting to 1 thread";
    mkl_set_num_threads(1);
  }

  // If caffe2_omp_num_threads is set, we use that for MKL as well.
  if (FLAGS_caffe2_omp_num_threads > 0) {
    VLOG(1) << "Setting mkl_num_threads to " << FLAGS_caffe2_omp_num_threads
            << " as inherited from omp_num_threads.";
    mkl_set_num_threads(FLAGS_caffe2_omp_num_threads);
  }

  // Override omp_num_threads if mkl_num_threads is set.
  if (FLAGS_caffe2_mkl_num_threads > 0) {
    VLOG(1) << "Setting mkl_num_threads to " << FLAGS_caffe2_mkl_num_threads;
    mkl_set_num_threads(FLAGS_caffe2_mkl_num_threads);
  }
  VLOG(1) << "Caffe2 running with " << mkl_get_max_threads() << " MKL threads";
  return true;
}
REGISTER_CAFFE2_INIT_FUNCTION(
    Caffe2SetMKLThreads,
    &Caffe2SetMKLThreads,
    "Set MKL threads.");
#endif // CAFFE2_USE_MKL

}  // namespace caffe2
