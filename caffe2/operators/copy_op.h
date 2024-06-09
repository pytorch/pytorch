#ifndef CAFFE2_OPERATORS_COPY_OP_H_
#define CAFFE2_OPERATORS_COPY_OP_H_

#include "caffe2/core/context.h"
#include "caffe2/core/export_caffe2_op_to_c10.h"
#include "caffe2/core/operator.h"

C10_DECLARE_EXPORT_CAFFE2_OP_TO_C10(CopyGPUToCPU)
C10_DECLARE_EXPORT_CAFFE2_OP_TO_C10(CopyCPUToGPU)

namespace caffe2 {

template <class Context, class DstContext, class SrcContext>
class CopyOp : public Operator<Context> {
 public:
  USE_OPERATOR_CONTEXT_FUNCTIONS;
  USE_SIMPLE_CTOR_DTOR(CopyOp)

  bool RunOnDevice() override {
    auto& input = this->template Input<Tensor>(0, SrcContext::GetDeviceType());
    auto* output =
        this->template Output<Tensor>(0, DstContext::GetDeviceType());
    output->ResizeLike(input);
    this->context_.template CopyItems<SrcContext, DstContext>(
        input.dtype(),
        input.numel(),
        input.raw_data(),
        output->raw_mutable_data(input.dtype()));
    return true;
  }
};

template <class Context, class DstContext, class SrcContext>
class CopyOnDeviceLikeOp : public CopyOp<Context, DstContext, SrcContext> {
 public:
  template <class... Args>
  explicit CopyOnDeviceLikeOp(Args&&... args)
      : CopyOp<Context, DstContext, SrcContext>(std::forward<Args>(args)...) {}
};

} // namespace caffe2

#endif // CAFFE2_OPERATORS_COPY_OP_H_
