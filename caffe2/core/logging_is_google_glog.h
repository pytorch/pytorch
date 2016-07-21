#ifndef CAFFE2_CORE_LOGGING_IS_GOOGLE_GLOG_H_
#define CAFFE2_CORE_LOGGING_IS_GOOGLE_GLOG_H_

#include <iomanip>  // because some of the caffe2 code uses e.g. std::setw
// Using google glog. For glog 0.3.2 versions, stl_logging.h needs to be before
// logging.h to actually use stl_logging. Because template magic.
// In addition, we do not do stl logging in .cu files because nvcc does not like
// it.
#ifndef __CUDACC__
#include <glog/stl_logging.h>
#endif  // __CUDACC__
#include <glog/logging.h>


#endif  // CAFFE2_CORE_LOGGING_IS_GOOGLE_GLOG_H_
