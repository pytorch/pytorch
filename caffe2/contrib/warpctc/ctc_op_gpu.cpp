#include "caffe2/core/context_gpu.h"
#include "caffe2/core/operator.h"
#include "ctc_op.h"

namespace caffe2 {

namespace detail {
template <>
ctcComputeInfo workspaceInfo<CUDAContext>(const CUDAContext& context) {
  ctcComputeInfo result;
  result.loc = CTC_GPU;
  result.stream = context.cuda_stream();
  return result;
}
}

REGISTER_CUDA_OPERATOR(CTC, CTCOp<float, CUDAContext>);
}
