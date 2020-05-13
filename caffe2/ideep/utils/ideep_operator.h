#pragma once

#include <ideep.hpp>
#include <caffe2/core/operator.h>
#include <caffe2/proto/caffe2_pb.h>

namespace caffe2 {

C10_DECLARE_REGISTRY(
    IDEEPOperatorRegistry,
    OperatorBase,
    const OperatorDef&,
    Workspace*);

#define REGISTER_IDEEP_OPERATOR_CREATOR(key, ...) \
  C10_REGISTER_CREATOR(IDEEPOperatorRegistry, key, __VA_ARGS__)
#define REGISTER_IDEEP_OPERATOR(name, ...) \
  C10_REGISTER_CLASS(IDEEPOperatorRegistry, name, __VA_ARGS__)
#define REGISTER_IDEEP_OPERATOR_WITH_ENGINE(name, engine, ...) \
  C10_REGISTER_CLASS(IDEEPOperatorRegistry, name##_ENGINE_##engine, __VA_ARGS__)
#define REGISTER_IDEEP_OPERATOR_STR(str_name, ...) \
  C10_REGISTER_TYPED_CLASS(IDEEPOperatorRegistry, str_name, __VA_ARGS__)
#define REGISTER_IDEEP_COMPARE_OPERATOR(Op)                    \
  REGISTER_IDEEP_OPERATOR(                                     \
      Op,                                                      \
      IDEEPFallbackOp<BinaryElementwiseOp<                     \
          TensorTypes<bool, int32_t, int64_t, float, double>,  \
          CPUContext,                                          \
          Op##Functor<CPUContext>,                             \
          FixedType<bool>>>)


// IDEEPOperator is the base scaffolding of the operators that uses IDEEP. It
// provides a few operators that are useful to IDEEP specific implementations.
class IDEEPOperator : public OperatorBase {
 public:
  explicit IDEEPOperator(const OperatorDef& operator_def, Workspace* ws)
      : OperatorBase(operator_def, ws),
        context_(operator_def.device_option()),
        order_(StringToStorageOrder(
            OperatorBase::GetSingleArgument<string>("order", "NCHW"))) {
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
      StartAllObservers();
      bool result =  RunOnDevice();
      StopAllObservers();
      return result;
    } catch (EnforceNotMet& err) {
      TORCH_RETHROW(err, getErrorMsg());
    } catch (ideep::error& e) {
      LOG(ERROR) << "IDEEP error:" << e.message;
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

// Convert zero_point scales to min_max scales
// NOTE:
//  The scales in operator is saved in FBGEMM format,
//  while FBGEMM scales are the reciprocals of MKL-DNN scales.
//  This function is provided to convert scales from FBGEMM to MKL-DNN
inline ideep::scale_t ConvertScales(
    const std::vector<float> scales_z) {
  ideep::scale_t scales (scales_z);
  for (auto it = scales.begin(); it != scales.end(); it++) {
    *it = 1.0f / *it;
  }
  return scales;
}

inline ideep::tensor::dims CanonicalDims(
    ideep::tensor::dims adims, int32_t axis) {
  CAFFE_ENFORCE(axis < (int32_t)adims.size(), "Invalid axis!");
  CAFFE_ENFORCE(axis > (int32_t)-adims.size(), "Invalid axis!");
  if (adims.size() == 2 || axis == 1)
    return adims;
  if (axis < 0) {
    axis += (int32_t)adims.size();
  }

  auto dim0 = std::accumulate(adims.begin(), adims.begin() + axis, 1,
                              std::multiplies<ideep::tensor::dim_t>());
  auto dim1 = std::accumulate(adims.begin() + axis, adims.end(), 1,
                              std::multiplies<ideep::tensor::dim_t>());
  return ideep::tensor::dims({dim0, dim1});
}

} // namespace caffe2
