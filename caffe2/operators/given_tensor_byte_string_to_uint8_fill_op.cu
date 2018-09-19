#include "caffe2/core/context_gpu.h"
#include "caffe2/operators/given_tensor_byte_string_to_uint8_fill_op.h"

namespace caffe2 {

REGISTER_CUDA_OPERATOR(
    GivenTensorByteStringToUInt8Fill,
    GivenTensorByteStringToUInt8FillOp<CUDAContext>);
}
