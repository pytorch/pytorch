#include "caffe2/operator/reduce_ops.h"

#include <algorithm>
#include <functional>
#include <vector>

#include "caffe2/core/common_gpu.h"
#include "caffe2/core/context_gpu.h"

namespace caffe2 {
REGISTER_CUDA_OPERATOR(
    ExpandNormal,
    ExpandOp<
        TensorType<std::int32_t, std::int64_t, float, double>,
        CUDAContext,
        NormalExpander<CUDAContext>>);

} // namespace caffe2
