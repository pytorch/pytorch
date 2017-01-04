#ifndef CAFFE2_CORE_COMMON_OMP_H_
#define CAFFE2_CORE_COMMON_OMP_H_

#ifdef _OPENMP
#include <omp.h>
#endif  // _OPENMP

// Macro to abstract out basic omp parallel loops
#ifdef _OPENMP
#define CAFFE2_OMP_PARALLEL_FOR() \
  _Pragma( "omp parallel for" )
  // _Pragma( STRINGIFY( CONCATENATE( omp parallel for ) ) )
#else
// empty macro, do nothing
#define CAFFE2_OMP_PARALLEL_FOR()
#endif // _OPENMP

#endif  // CAFFE2_CORE_COMMON_OMP_H_
