#include "caffe2/utils/mkl_utils.h"

#ifdef CAFFE2_HAS_MKL_DNN

namespace caffe2 {

CAFFE_KNOWN_TYPE(mkl::MKLMemory<float>);
CAFFE_KNOWN_TYPE(mkl::MKLMemory<double>);

} // namespace caffe2

#endif // CAFFE2_HAS_MKL_DNN
