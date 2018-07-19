#include "caffe2/core/net_simple_async.h"
#include "caffe2/core/net.h"

#include <iostream>
#include <set>
#include <unordered_map>
#include <unordered_set>

#include "caffe2/core/operator.h"
#include "caffe2/core/static_tracepoint.h"
#include "caffe2/core/timer.h"
#include "caffe2/proto/caffe2.pb.h"
#include "caffe2/utils/proto_utils.h"

namespace caffe2 {

AsyncSimpleNet::AsyncSimpleNet(
    const std::shared_ptr<const NetDef>& net_def,
    Workspace* ws)
    : NetBase(net_def, ws) {
  VLOG(1) << "Constructing AsyncSimpleNet " << net_def->name();
  const bool net_def_has_device_option = net_def->has_device_option();
  // Initialize the operators
  const DeviceOption* first_device_option = nullptr;
  const DeviceOption* current_device_option;
  for (int idx = 0; idx < net_def->op_size(); ++idx) {
    const auto& operator_def = net_def->op(idx);
    VLOG(1) << "Creating operator " << operator_def.name() << ": "
            << operator_def.type();
    std::unique_ptr<OperatorBase> op{nullptr};
    if (!operator_def.has_device_option() && net_def_has_device_option) {
      // In the case that the operator def does not specify a device option but
      // the net def has a default option, we copy the device option over to the
      // operator def.
      OperatorDef temp_def(operator_def);
      temp_def.mutable_device_option()->CopyFrom(net_def->device_option());
      op = CreateOperator(temp_def, ws, idx);
      current_device_option = &net_def->device_option();
    } else {
      op = CreateOperator(operator_def, ws, idx);
      op->set_debug_def(
          std::shared_ptr<const OperatorDef>{net_def, &(net_def->op(idx))});
      current_device_option = &operator_def.device_option();
    }
    if (!first_device_option) {
      first_device_option = current_device_option;
    } else {
      CAFFE_ENFORCE(
          IsSameDevice(*first_device_option, *current_device_option),
          "AsyncSimpleNet supports only single device networks");
    }
    operators_.emplace_back(std::move(op));
  }
  events_ = {&operators_.back()->event()};
}

bool AsyncSimpleNet::DoRunAsync() {
  StartAllObservers();

  VLOG(1) << "Running net " << name_;
  for (auto& op : operators_) {
    VLOG(1) << "Running operator " << op->debug_def().name() << "("
            << op->debug_def().type() << ").";
#ifdef CAFFE2_ENABLE_SDT
    const auto& op_name = op->debug_def().name().c_str();
    const auto& op_type = op->debug_def().type().c_str();
    auto* op_ptr = op.get();
    const auto& net_name = name_.c_str();
    CAFFE_SDT(operator_start, net_name, op_name, op_type, op_ptr);
#endif
    bool res = op->RunAsync();
#ifdef CAFFE2_ENABLE_SDT
    CAFFE_SDT(operator_done, net_name, op_name, op_type, op_ptr);
#endif
    if (!res) {
      LOG(ERROR) << "Operator failed: " << ProtoDebugString(op->debug_def());
      return false;
    }
  }
  StopAllObservers();
  return true;
}

REGISTER_NET(async_simple, AsyncSimpleNet);

} // namespace caffe2
