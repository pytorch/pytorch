#include "caffe2/core/context_gpu.h"
#include "caffe2/operators/copy_op.h"

namespace caffe2 {

template <>
class CopyOnDeviceLikeOp<CUDAContext, CUDAContext, CUDAContext>
    : public Operator<CUDAContext> {
 public:
  template <class... Args>
  explicit CopyOnDeviceLikeOp(Args&&... args)
      : Operator<CUDAContext>(std::forward<Args>(args)...) {}
  USE_OPERATOR_FUNCTIONS(CUDAContext);

  bool RunOnDevice() override {
    auto& input = Input(0);
    auto* output = OperatorBase::Output<Tensor>(0, CUDA);
    CUDAContext context(GetGPUIDForPointer(Input(1).raw_data()));
    output->ResizeLike(input);
    context.template CopyItems<CUDAContext, CUDAContext>(
        input.meta(),
        input.numel(),
        input.raw_data(),
        output->raw_mutable_data(input.meta()));
    return true;
  }
};

// From CPU, copy it to whatever the current context
REGISTER_CUDA_OPERATOR(
    CopyFromCPUInput,
    CopyOp<CUDAContext, CUDAContext, CPUContext>);

// CopyGPUToCPU and CopyCPUToGPU should both be carried out in a cuda context,
// since gpu code will be involved.
REGISTER_CUDA_OPERATOR(
    CopyGPUToCPU,
    CopyOp<CUDAContext, CPUContext, CUDAContext>);
REGISTER_CUDA_OPERATOR(
    CopyCPUToGPU,
    CopyOp<CUDAContext, CUDAContext, CPUContext>);
// If we only specify Copy, we assume that it is a gpu to gpu copy - maybe
// involving different GPUs.
REGISTER_CUDA_OPERATOR(Copy, CopyOp<CUDAContext, CUDAContext, CUDAContext>);

REGISTER_CUDA_OPERATOR(
    CopyOnDeviceLike,
    CopyOnDeviceLikeOp<CUDAContext, CUDAContext, CUDAContext>);
} // namespace caffe2

using CopyGPUToCPU_CUDA = caffe2::
    CopyOp<caffe2::CUDAContext, caffe2::CPUContext, caffe2::CUDAContext>;
using CopyCPUToGPU_CUDA = caffe2::
    CopyOp<caffe2::CUDAContext, caffe2::CUDAContext, caffe2::CPUContext>;

C10_EXPORT_CAFFE2_OP_TO_C10_CUDA(CopyGPUToCPU, CopyGPUToCPU_CUDA);

C10_EXPORT_CAFFE2_OP_TO_C10_CPU_KERNEL_ONLY(CopyCPUToGPU, CopyCPUToGPU_CUDA);
