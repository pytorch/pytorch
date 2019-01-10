#pragma once

#include <ideep.hpp>
#include <caffe2/core/operator.h>
#include <caffe2/proto/caffe2.pb.h>

namespace caffe2 {

CAFFE_DECLARE_REGISTRY(
    IDEEPOperatorRegistry,
    OperatorBase,
    const OperatorDef&,
    Workspace*);

#define REGISTER_IDEEP_OPERATOR_CREATOR(key, ...) \
  CAFFE_REGISTER_CREATOR(IDEEPOperatorRegistry, key, __VA_ARGS__)
#define REGISTER_IDEEP_OPERATOR(name, ...) \
  CAFFE_REGISTER_CLASS(IDEEPOperatorRegistry, name, __VA_ARGS__)
#define REGISTER_IDEEP_OPERATOR_STR(str_name, ...) \
  CAFFE_REGISTER_TYPED_CLASS(IDEEPOperatorRegistry, str_name, __VA_ARGS__)

#define REGISTER_IDEEP_OPERATOR_WITH_ENGINE(name, engine, ...) \
  CAFFE_REGISTER_CLASS(IDEEPOperatorRegistry, name##_ENGINE_##engine, __VA_ARGS__)

// IDEEPOperator is the base scaffolding of the operators that uses IDEEP. It
// provides a few operators that are useful to IDEEP specific implementations.
class IDEEPOperator : public OperatorBase {
 public:
  explicit IDEEPOperator(const OperatorDef& operator_def, Workspace* ws)
      : OperatorBase(operator_def, ws),
        context_(operator_def.device_option()),
        order_(StringToStorageOrder(
            OperatorBase::GetSingleArgument<string>("order", "NCHW"))) {
    OPERATOR_NEEDS_FEATURE(
        order_ == StorageOrder::NCHW, "Unsupported storage order.");
  }
  virtual ~IDEEPOperator() {}

  inline const ideep::tensor& Input(int index) {
    return OperatorBase::template Input<ideep::tensor>(index);
  }
  inline ideep::tensor* Output(int index) {
    return OperatorBase::template Output<ideep::tensor>(index);
  }

  // The run function of Operator switches to the device, and then carries out
  // the actual computation with RunOnDevice(). You should implement RunOnDevice
  // instead of Run().
  bool Run(int /* unused */ /*stream_id*/) final {
    // Since IDEEP does not need to do SwithToDevice and
    // FinishDeviceComputation,
    // it is always just a re-route to RunOnDevice().
    try {
      return RunOnDevice();
    } catch (EnforceNotMet& err) {
      err.AppendMessage(getErrorMsg());
      throw;
    } catch (ideep::error& e) {
      VLOG(1) << "IDEEP error:" << e.message; 
      throw;
    }
  }

  // Waits for a previous event. Note that to properly wait and run
  // asynchronously, WaitEvent, RunAsync and Record should all be executed
  // on the same CPU thread.
  void WaitEvent(const Event& ev, int /* unused */) final {
    context_.WaitEvent(ev);
  }

  void WaitEvents(const std::vector<const Event*>& events, int /* unused */)
      final {
    for (const auto& ev : events) {
      context_.WaitEvent(*ev);
    }
  }

  void RecordEvent(const char* err_msg = nullptr) final {
    if (event_) {
      context_.Record(event_.get(), err_msg);
    }
  }

  virtual bool RunOnDevice() = 0;

 protected:
  std::string getErrorMsg() {
    if (has_debug_def()) {
      return "Error from operator: " + ProtoDebugString(debug_def());
    } else {
      return "Error from operator: no op def";
    }
  }

  IDEEPContext context_;
  StorageOrder order_;
};

#define USE_IDEEP_OPERATOR_FUNCTIONS()                                         \
  USE_OPERATOR_BASE_FUNCTIONS;                                                 \
  /* using override */ using IDEEPOperator::Input;                             \
  /* using override */ using IDEEPOperator::Output;                            \
  /* using override */ using IDEEPOperator::order_;                            \
  /* using override */ using IDEEPOperator::context_;

#define USE_SIMPLE_IDEEP_CTOR_DTOR(name)                                       \
  name(const OperatorDef& operator_def, Workspace* ws)                         \
      : IDEEPOperator(operator_def, ws) {}                                     \
  virtual ~name() {}

} // namespace caffe2
