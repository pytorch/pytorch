#include "caffe2/operators/async_net_barrier_op.h"

namespace caffe2 {

namespace {
std::pair<std::vector<DeviceOption>, std::vector<DeviceOption>>
asyncBarrierOpDevInfer(const OperatorDef& def) {
  auto op_device =
      def.has_device_option() ? def.device_option() : DeviceOption();
  ArgumentHelper helper(def);
  auto cross_device = helper.GetSingleArgument<int>("cross_device", 0);
  std::vector<DeviceOption> opt;
  for (int i = 0; i < def.input().size(); ++i) {
    if (cross_device == 1) {
      DeviceOption dev;
      dev.set_device_type(op_device.device_type());
      dev.set_device_id(i);
      opt.push_back(dev);
    } else {
      opt.push_back(op_device);
    }
  }
  return std::make_pair(opt, opt);
}
}

OPERATOR_SCHEMA(AsyncNetBarrier)
    .NumInputs(1, INT_MAX)
    .NumOutputs(1, INT_MAX)
    .IdenticalTypeAndShape()
    .InputsCanCrossDevices()
    .AllowOneToOneInplace()
    .DeviceInferenceFunction(asyncBarrierOpDevInfer)
    .SetDoc(R"DOC(
This is a pretty much no-op operator, since it's only purposes is make sure that
async_scheduling will schedule certian operations earlier than others.

Exaple where this operator can work well - mixture of data-parallel and model-
parallel training, where one wants to force that all copies are started before
data-parallel part starts.
)DOC")
    .Arg(
        "cross_device",
        "Specifies either inputs should be across different devices in dev inference options");

SHOULD_NOT_DO_GRADIENT(AsyncNetBarrier);
REGISTER_CPU_OPERATOR(AsyncNetBarrier, AsyncNetBarrierOp<CPUContext>);


} // namespace caffe2
