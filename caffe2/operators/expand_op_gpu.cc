#include "caffe2/operators/expand_op.h"

#include <algorithm>
#include <functional>
#include <vector>

#include "caffe2/core/common_gpu.h"
#include "caffe2/core/context_gpu.h"

namespace caffe2 {

template <typename InputTypes, class Expander>
const Tensor<CPUContext> ExpandOp<InputTypes, CUDAContext, Expander>::GetShape() {
	const auto& shape_gpu = Input(1);
	Tensor<CPUContext> shape_cpu;
	shape_cpu.Resize(shape_gpu.size());
	shape_cpu.CopyFrom(shape_gpu, &context_);
	return shape_cpu;
}

REGISTER_CUDA_OPERATOR(
    Expand,
    ExpandOp<
        TensorTypes<std::int32_t, std::int64_t, float, double>,
        CUDAContext,
        Expander<CUDAContext>>);
REGISTER_CUDA_OPERATOR(
    ExpandGradient,
    ExpandGradientOp<
        TensorTypes<std::int32_t, std::int64_t, float, double>,
        CUDAContext,
        Expander<CUDAContext>>);
} // namespace caffe2
