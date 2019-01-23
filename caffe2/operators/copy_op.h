#ifndef CAFFE2_OPERATORS_COPY_OP_H_
#define CAFFE2_OPERATORS_COPY_OP_H_

#include "caffe2/core/context.h"
#include "caffe2/core/operator.h"

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
  CopyOnDeviceLikeOp(const OperatorDef& operator_def, Workspace* ws)
      : CopyOp<Context, DstContext, SrcContext>(operator_def, ws) {}
};

} // namespace caffe2

#endif // CAFFE2_OPERATORS_COPY_OP_H_
