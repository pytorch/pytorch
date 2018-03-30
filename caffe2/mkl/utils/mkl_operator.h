#ifndef CAFFE2_UTILS_MKL_OPERATOR_H_
#define CAFFE2_UTILS_MKL_OPERATOR_H_

#include "caffe2/core/operator.h"
#include "caffe2/mkl/utils/mkl_dnn_cppwrapper.h"
#include "caffe2/mkl/utils/mkl_memory.h"
#include "caffe2/proto/caffe2.pb.h"

CAFFE2_DECLARE_bool(caffe2_mkl_memonger_in_use);

namespace caffe2 {

CAFFE_DECLARE_REGISTRY(
    MKLOperatorRegistry,
    OperatorBase,
    const OperatorDef&,
    Workspace*);
#define REGISTER_MKL_OPERATOR_CREATOR(key, ...) \
  CAFFE_REGISTER_CREATOR(MKLOperatorRegistry, key, __VA_ARGS__)
#define REGISTER_MKL_OPERATOR(name, ...) \
  CAFFE_REGISTER_CLASS(MKLOperatorRegistry, name, __VA_ARGS__)
#define REGISTER_MKL_OPERATOR_STR(str_name, ...) \
  CAFFE_REGISTER_TYPED_CLASS(MKLOperatorRegistry, str_name, __VA_ARGS__)

#define REGISTER_MKL_OPERATOR_WITH_ENGINE(name, engine, ...) \
  CAFFE_REGISTER_CLASS(MKLOperatorRegistry, name##_ENGINE_##engine, __VA_ARGS__)

namespace mkl {
// MKLOperator is the base scaffolding of the operators that uses MKLDNN. It
// provides a few operators that are useful to MKLDNN specific implementations.
template <typename T>
class MKLOperator : public OperatorBase {
 public:
  explicit MKLOperator(const OperatorDef& operator_def, Workspace* ws)
      : OperatorBase(operator_def, ws),
        context_(operator_def.device_option()) {}
  virtual ~MKLOperator() {}

  inline const MKLMemory<T>& Input(int idx) {
    return OperatorBase::template Input<MKLMemory<T>>(idx);
  }
  inline MKLMemory<T>* Output(int idx) {
    return OperatorBase::template Output<MKLMemory<T>>(idx);
  }

  // The run function of Operator switches to the device, and then carries out
  // the actual computation with RunOnDevice(). You should implement RunOnDevice
  // instead of Run().
  bool Run(int /* unused */ /*stream_id*/) final {
    // Since MKLDNN does not need to do SwithToDevice and
    // FinishDeviceComputation,
    // it is always just a re-route to RunOnDevice().
    try {
      return RunOnDevice();
    } catch (EnforceNotMet& err) {
      err.AppendMessage(getErrorMsg());
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

  inline void ExecutePrimitive() {
    MKLDNN_SAFE_CALL(mkl::dnnExecute<T>(primitive_, resources_));
  }

 protected:
  std::string getErrorMsg() {
    if (has_debug_def()) {
      return "Error from operator: " + ProtoDebugString(debug_def());
    } else {
      return "Error from operator: no op def";
    }
  }

  MKLContext context_;
  // The primitive used in the operator.
  PrimitiveWrapper<T> primitive_;
  // Size cache for all the input sizes.
  vector<vector<TIndex>> input_size_cache_;
  // An internal MKLMemory buffer. This is usually handy when we have a
  // single output from the operator. If your operator has multiple outputs
  // then you should allocate your own buffer.
  MKLMemory<T> buffer_;
  // The resources vector that we will need to use;
  void* resources_[dnnResourceNumber];
};
} // namespace mkl

#define USE_MKLOPERATOR_FUNCTIONS(T)                            \
  USE_OPERATOR_BASE_FUNCTIONS;                                  \
  /* using override */ using MKLOperator<T>::Input;             \
  /* using override */ using MKLOperator<T>::Output;            \
  /* using override */ using MKLOperator<T>::ExecutePrimitive;  \
  /* using override */ using MKLOperator<T>::primitive_;        \
  /* using override */ using MKLOperator<T>::input_size_cache_; \
  /* using override */ using MKLOperator<T>::buffer_;           \
  /* using override */ using MKLOperator<T>::resources_

#define USE_SIMPLE_MKL_CTOR_DTOR(name, T)              \
  name(const OperatorDef& operator_def, Workspace* ws) \
      : MKLOperator<T>(operator_def, ws) {}            \
  virtual ~name() {}

} // namespace caffe2

#endif // CAFFE2_UTILS_MKL_OPERATOR_H_
