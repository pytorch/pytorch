#include "caffe2/core/context_gpu.h"
#include "caffe2/operators/elementwise_op.h"

namespace caffe2 {
namespace {

REGISTER_CUDA_OPERATOR(Add, AddOp<CUDAContext>);
REGISTER_CUDA_OPERATOR(Sub, SubOp<CUDAContext>);
REGISTER_CUDA_OPERATOR(Mul, MulOp<CUDAContext>);
REGISTER_CUDA_OPERATOR(Div, DivOp<CUDAContext>);

}  // namespace
}  // namespace caffe2
