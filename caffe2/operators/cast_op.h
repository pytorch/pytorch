#pragma once

#include "caffe2/core/context.h"
#include "caffe2/core/operator.h"
#include "caffe2/utils/cast.h"
#include "caffe2/utils/conversions.h"
#include "caffe2/utils/math.h"
#include "caffe2/core/logging.h"
#include "caffe2/core/types.h"

namespace caffe2 {

template <class Context>
class CastOp : public Operator<Context> {
 public:
  USE_OPERATOR_CONTEXT_FUNCTIONS;

  explicit CastOp(const OperatorDef& operator_def, Workspace* ws)
      : Operator<Context>(operator_def, ws) {
    const ArgumentHelper helper(operator_def);
    TensorProto_DataType to = cast::GetCastDataType(helper, "to");
    TensorProto_DataType from = cast::GetCastDataType(helper, "from_type");

    SetBody(to);
  }

  bool RunOnDevice() override {
    return (this->*body_)();
  }

  // Allow for Context-specific implementations
  void SetBody(TensorProto_DataType to);

  template <typename DstType>
  bool DoRunWithDstType();

  template <typename DstType, typename SrcType>
  bool DoRunWithType() {
    auto& input = Input(0);
    auto* output = Output(0);
    output->ResizeLike(input);
    const auto* data = input.template data<SrcType>();
    auto* out = output->template mutable_data<DstType>();
    auto N = input.size();
    for (int64_t i = 0; i < N; ++i) {
      out[i] = static_cast<DstType>(data[i]);
    }
    return true;
  }

 private:
  bool (CastOp::*body_)();
};

}  // namespace caffe2
