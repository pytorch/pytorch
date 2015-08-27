#ifndef CAFFE2_CORE_OPERATOR_H_
#define CAFFE2_CORE_OPERATOR_H_

#include <climits>
#include <cstddef>
#include <typeinfo>
#include <vector>

#include "caffe2/core/blob.h"
#include "caffe2/core/common.h"
#include "caffe2/core/net.h"
#include "caffe2/core/registry.h"
#include "caffe2/core/workspace.h"
#include "caffe2/proto/caffe2.pb.h"

namespace caffe2 {

class OperatorBase {
 public:
  // The constructor of the operator. Note that you should not do any
  // custom initializations in the constructor; instead, do those in the
  // SetUp() function.
  explicit OperatorBase(const OperatorDef& operator_def, Workspace* ws);
  virtual ~OperatorBase() {}

  // Verify return true if an operator is set up correctly. This cannot be
  // implemented in the constructor, because there will be calls to overridden
  // functions.
  virtual bool Verify();

  // Parameter getters. You can use these to get the arguments that you want.
  bool HasArgument(const string& name) { return (arg_map_.count(name) > 0); }

  // Functions that deal with arguments. Basically, this allows us to map an
  // argument mane to a specific type of argument that we are trying to access.
  template <typename T>
  T GetSingleArgument(const string& name, const T& default_value);
  template <typename T>
  vector<T> GetRepeatedArgument(const string& name);

  template <typename MessageType>
  MessageType GetMessageArgument(const string& name) {
    CHECK(arg_map_.count(name)) << "Cannot find parameter named " << name;
    MessageType message;
    if (arg_map_[name]->has_s()) {
      CHECK(message.ParseFromString(arg_map_[name]->s()))
          << "Faild to parse content from the string";
    } else {
      VLOG(1) << "Return empty message for parameter " << name;
    }
    return message;
  }
  template <typename MessageType>
  vector<MessageType> GetRepeatedMessageArgument(const string& name) {
    CHECK(arg_map_.count(name)) << "Cannot find parameter named " << name;
    vector<MessageType> messages(arg_map_[name]->strings_size());
    for (int i = 0; i < messages.size(); ++i) {
      CHECK(messages[i].ParseFromString(arg_map_[name]->strings(i)))
          << "Faild to parse content from the string";
    }
    return messages;
  }

  // Get the inputs and outputs as specific types.
  template <typename T>
  inline const T& Input(int idx) {
    DCHECK_LT(idx, inputs_.size());
    return inputs_.at(idx)->template Get<T>();
  }

  template <typename T>
  inline T* Output(int idx) {
    DCHECK_LT(idx, outputs_.size());
    return outputs_.at(idx)->template GetMutable<T>();
  }

  template <typename T>
  inline bool InputIsType(int idx) {
    return inputs_.at(idx)->template IsType<T>();
  }

  template <typename T>
  inline bool OutputIsType(int idx) {
    return outputs_.at(idx)->template IsType<T>();
  }

  inline int InputSize() { return inputs_.size(); }
  inline int OutputSize() { return outputs_.size(); }
  inline const vector<const Blob*>& Inputs() const { return inputs_; }
  inline const vector<Blob*>& Outputs() { return outputs_; }

  virtual bool Run() { NOT_IMPLEMENTED; return false; }

  inline const OperatorDef& def() { return operator_def_; }

 protected:
  // Do not manually override these functions. Instead, use INPUT_OUTPUT_STATS
  // macro below.
  virtual int MinInput() { return 0; }
  virtual int MaxInput() { return INT_MAX; }
  virtual int MinOutput() { return 0; }
  virtual int MaxOutput() { return INT_MAX; }

 private:
  CaffeMap<string, const Argument*> arg_map_;
  OperatorDef operator_def_;
  vector<const Blob*> inputs_;
  vector<Blob*> outputs_;

  DISABLE_COPY_AND_ASSIGN(OperatorBase);
};

// If your operator does not need any specialized contructor or destructor,
// you can simply use this to save two lines of code.
#define USE_SIMPLE_BASE_CTOR_DTOR(name)                                        \
  name(const OperatorDef& operator_def, Workspace* ws)                         \
      : OperatorBase(operator_def, ws) {}                                      \
  virtual ~name() {}

// INPUT_OUTPUT_STATS gives the statistics of the input and output that are
// legal. If the max input/output is not limited, you can specify INT_MAX.
// TODO(Yangqing): If necessary, add ability to specify that n_input = n_output.
#define INPUT_OUTPUT_STATS(min_input, max_input, min_output, max_output)       \
 protected:                                                                    \
  int MinInput() override { return min_input; }                                \
  int MaxInput() override { return max_input; }                                \
  int MinOutput() override { return min_output; }                              \
  int MaxOutput() override { return max_output; }

// INPUT_TAGS and OUTPUT_TAGS are optional features to name the indices of the
// operator's inputs and outputs, in order to avoid confusion. For example, for
// a fully convolution layer that has input, weight and bias, you can define its
// input tags as:
//     INPUT_TAGS(INPUT, WEIGHT, BIAS);
// And in the code, instead of doing
//     auto& weight = Input(1);
// you can now do
//     auto& weight = Input(WEIGHT);
// to make it more clear.
#define INPUT_TAGS(first_input, ...)                                           \
  enum _InputTags { first_input = 0, __VA_ARGS__ }
#define OUTPUT_TAGS(first_input, ...)                                          \
  enum _OutputTags { first_input = 0, __VA_ARGS__ }


// Operator is the class that you usually want to derive, if your operator will
// run on different devices. You should then implement the RunOnDevice()
// function.
template <typename dtype, class DeviceContext>
class Operator : public OperatorBase {
 public:
  // The constructor of the operator. Note that you should not do any
  // custom initializations in the constructor; instead, do those in the
  // SetUp() function.
  explicit Operator(const OperatorDef& operator_def, Workspace* ws)
      : OperatorBase(operator_def, ws),
        device_context_(operator_def.device_option()) {
    // In the constructor, we switch to the device so that the child class
    // constructors will run on that device.
    device_context_.SwitchToDevice();
  }
  virtual ~Operator() {}

  inline const Tensor<dtype, DeviceContext>& Input(int idx) {
    return OperatorBase::template Input<Tensor<dtype, DeviceContext> >(idx); }
  inline Tensor<dtype, DeviceContext>* Output(int idx) {
    return OperatorBase::template Output<Tensor<dtype, DeviceContext> >(idx);
  }

  // The run function of Operator switches to the device, and then carries out
  // the actual computation with RunOnDevice(). You should implement RunOnDevice
  // instead of Run().
  bool Run() {
    device_context_.SwitchToDevice();
    bool result = RunOnDevice();
    result &= device_context_.FinishDeviceComputation();
    return result;
  }

  virtual bool RunOnDevice() = 0;

 protected:
  DeviceContext device_context_;
  DISABLE_COPY_AND_ASSIGN(Operator);
};

#define USE_OPERATOR_BASE_FUNCTIONS                                            \
  using OperatorBase::GetSingleArgument;                                       \
  using OperatorBase::GetRepeatedArgument;                                     \
  using OperatorBase::def;                                                     \
  using OperatorBase::InputIsType;                                             \
  using OperatorBase::InputSize;                                               \
  using OperatorBase::OutputSize;                                              \
  using Operator<dtype, DeviceContext>::device_context_;                       \
  using Operator<dtype, DeviceContext>::Input;                                 \
  using Operator<dtype, DeviceContext>::Output

#define USE_SIMPLE_CTOR_DTOR(name)                                             \
  name(const OperatorDef& operator_def, Workspace* ws)                         \
      : Operator<dtype, DeviceContext>(operator_def, ws) {}                    \
  virtual ~name() {}

// The operator registry. Since we are not expecting a great number of devices,
// we will simply have an if-then type command and allocate the actual
// generation to device-specific registerers.
// Note that although we have CUDA and CUDNN here, the registerers themselves do
// not depend on specific cuda or cudnn libraries. This means that we will be
// able to compile it even when there is no cuda available - we simply do not
// link any cuda or cudnn operators.
DECLARE_REGISTRY(CPUOperatorRegistry, OperatorBase,
                 const OperatorDef&, Workspace*);
#define REGISTER_CPU_OPERATOR_CREATOR(key, ...) \
  REGISTER_CREATOR(CPUOperatorRegistry, key, __VA_ARGS__)
#define REGISTER_CPU_OPERATOR(name, ...) \
  REGISTER_CLASS(CPUOperatorRegistry, name, __VA_ARGS__)

DECLARE_REGISTRY(CUDAOperatorRegistry, OperatorBase,
                 const OperatorDef&, Workspace*);
#define REGISTER_CUDA_OPERATOR_CREATOR(key, ...) \
  REGISTER_CREATOR(CUDAOperatorRegistry, key, __VA_ARGS__)
#define REGISTER_CUDA_OPERATOR(name, ...) \
  REGISTER_CLASS(CUDAOperatorRegistry, name, __VA_ARGS__)

DECLARE_REGISTRY(CUDNNOperatorRegistry, OperatorBase,
                 const OperatorDef&, Workspace*);
#define REGISTER_CUDNN_OPERATOR_CREATOR(key, ...) \
  REGISTER_CREATOR(CUDNNOperatorRegistry, key, __VA_ARGS__)
#define REGISTER_CUDNN_OPERATOR(name, ...) \
  REGISTER_CLASS(CUDNNOperatorRegistry, name, __VA_ARGS__)

// Creates an operator with the given operator definition.
OperatorBase* CreateOperator(const OperatorDef& operator_def, Workspace* ws);

}  // namespace caffe2

#endif  // CAFFE2_CORE_OPERATOR_H_
