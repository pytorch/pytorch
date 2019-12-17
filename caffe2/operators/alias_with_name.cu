#include "caffe2/core/context_gpu.h"
#include "caffe2/operators/alias_with_name.h"

namespace caffe2 {

REGISTER_CUDA_OPERATOR(AliasWithName, AliasWithNameOp<CUDAContext>);

} // namespace caffe2

C10_EXPORT_CAFFE2_OP_TO_C10_CUDA(
    AliasWithName,
    caffe2::AliasWithNameOp<caffe2::CUDAContext>);
