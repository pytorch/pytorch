#include "caffe2/core/context_gpu.h"
#include "caffe2/operators/elementwise_op.h"

namespace caffe2 {
namespace {

REGISTER_CUDA_OPERATOR(Add, AddOp<float, CUDAContext>);
REGISTER_CUDA_OPERATOR(Sub, SubOp<float, CUDAContext>);
REGISTER_CUDA_OPERATOR(Mul, MulOp<float, CUDAContext>);
REGISTER_CUDA_OPERATOR(Div, DivOp<float, CUDAContext>);

}  // namespace
}  // namespace caffe2
