#include "caffe2/mkl/mkl_utils.h"

#ifdef CAFFE2_HAS_MKL_DNN

CAFFE2_DEFINE_bool(
    caffe2_mkl_implicit_layout_change, false,
    "Controls the behavior when we call View() on an MKLMemory: if it is set "
    "true, then the View() function will actually change the underlying "
    "storage. If it is set false, an implicit copy is triggered but the "
    "original storage is not affected."
    );

namespace caffe2 {

CAFFE_KNOWN_TYPE(mkl::MKLMemory<float>);
CAFFE_KNOWN_TYPE(mkl::MKLMemory<double>);

} // namespace caffe2

#endif // CAFFE2_HAS_MKL_DNN
