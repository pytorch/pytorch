#include "caffe2/operators/expand_op.h"

#include <algorithm>
#include <functional>
#include <vector>

#include "caffe2/core/common_gpu.h"
#include "caffe2/core/context_gpu.h"

namespace caffe2 {
REGISTER_CUDA_OPERATOR(
    Expand,
    ExpandOp<
        TensorType<std::int32_t, std::int64_t, float, double>,
        CUDAContext,
        Expander<CUDAContext>>);

} // namespace caffe2
