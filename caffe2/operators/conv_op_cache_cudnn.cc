#include "caffe2/operators/conv_op_cache_cudnn.h"

#include <cudnn.h>

#include "caffe2/core/logging.h"
#include "caffe2/core/tensor.h"

namespace caffe2 {

template class AlgorithmsCache<cudnnConvolutionFwdAlgo_t>;
template class AlgorithmsCache<cudnnConvolutionBwdFilterAlgo_t>;
template class AlgorithmsCache<cudnnConvolutionBwdDataAlgo_t>;
template class AlgorithmsCache<int>; // For testing.
} // namespace caffe2
