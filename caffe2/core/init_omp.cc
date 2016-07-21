#ifdef _OPENMP
#include "omp.h"
#endif  // _OPENMP

#include "caffe2/core/init.h"

CAFFE2_DEFINE_int(
    caffe2_omp_num_threads, 0,
    "The number of openmp threads. 0 to use default value. "
    "Does not have effect if OpenMP is disabled.");

#ifdef _OPENMP
namespace caffe2 {
namespace {
bool Caffe2SetOpenMPThreads(int*, char***) {
  if (FLAGS_caffe2_omp_num_threads > 0) {
    LOG(INFO) << "Setting omp_num_threads to " << FLAGS_caffe2_omp_num_threads;
    omp_set_num_threads(FLAGS_caffe2_omp_num_threads);
  }
  return true;
}
REGISTER_CAFFE2_INIT_FUNCTION(Caffe2SetOpenMPThreads,
                              &Caffe2SetOpenMPThreads,
                              "Set OpenMP threads.");
}  // namespace
}  // namespace caffe2
#endif  // _OPENMP
