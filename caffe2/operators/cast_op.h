#pragma once

#include <c10/util/irange.h>
#include "caffe2/core/context.h"
#include "caffe2/core/logging.h"
#include "caffe2/core/operator.h"
#include "caffe2/core/types.h"
#include "caffe2/utils/cast.h"
#include "caffe2/utils/conversions.h"
#include "caffe2/utils/math.h"

namespace caffe2 {

template <class Context>
class CastOp : public Operator<Context> {
 public:
  USE_OPERATOR_CONTEXT_FUNCTIONS;

  explicit CastOp(const OperatorDef& operator_def, Workspace* ws)
      : Operator<Context>(operator_def, ws) {
    const ArgumentHelper helper(operator_def);
    TensorProto_DataType to = cast::GetCastDataType(helper, "to");

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
    for (const auto i : c10::irange(N)) {
      out[i] = static_cast<DstType>(data[i]);
    }
    return true;
  }

 private:
  bool (CastOp::*body_)();
};

} // namespace caffe2
