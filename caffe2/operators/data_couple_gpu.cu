#include "caffe2/core/context_gpu.h"
#include "caffe2/operators/data_couple.h"

namespace caffe2 {
REGISTER_CUDA_OPERATOR(DataCouple, DataCoupleOp<CUDAContext>);
}
