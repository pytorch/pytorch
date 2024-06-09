#include <caffe2/quantization/server/int8_gen_quant_params.h>
#include <caffe2/utils/proto_utils.h>
#include "freeze_quantization_params.h"

namespace caffe2 {
void freezeQuantizationParams(NetDef* net, Workspace* ws) {
  for (auto& op : *net->mutable_op()) {
    if ((op.type() == "Int8Quantize" && op.input_size() == 2) ||
        (op.type() == "Int8FC" && op.input_size() == 4)) {
      int lastPos = op.input_size() - 1;
      const auto& paramName = op.input(lastPos);
      auto* b = ws->GetBlob(paramName);
      if (!b) {
        LOG(WARNING)
            << "ParamBlob " << paramName
            << " does not exist in the workspace. Skip freezing current op.";
        continue;
      }
      if (!b->template IsType<caffe2::unique_ptr<Int8QuantParamsBlob>>()) {
        LOG(WARNING)
            << "ParamBlob " << paramName
            << " is not of caffe2::unique_ptr<Int8QuantParamsBlob> type. Skip freezing current op.";
        continue;
      }

      // Extract and set scale and zero point for the op
      const auto* param =
          b->template Get<caffe2::unique_ptr<Int8QuantParamsBlob>>().get();
      CAFFE_ENFORCE(param);
      const float scale = param->qparam.scale;
      const int zero_point = param->qparam.zero_point;
      bool argSet = false;
      for (auto& arg : *op.mutable_arg()) {
        if (arg.name() == "Y_scale") {
          arg.set_f(scale);
          argSet = true;
          break;
        }
      }
      if (!argSet) {
        op.add_arg()->CopyFrom(MakeArgument<float>("Y_scale", scale));
      }
      argSet = false;
      for (auto& arg : *op.mutable_arg()) {
        if (arg.name() == "Y_zero_point") {
          arg.set_i(zero_point);
          argSet = true;
          break;
        }
      }
      if (!argSet) {
        op.add_arg()->CopyFrom(MakeArgument<int>("Y_zero_point", zero_point));
      }

      // Remove last input of the op
      op.mutable_input()->RemoveLast();
    }
  }
}
} // namespace caffe2
