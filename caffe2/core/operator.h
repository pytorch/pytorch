#ifndef CAFFE2_CORE_OPERATOR_H_
#define CAFFE2_CORE_OPERATOR_H_

#include <climits>
#include <cstddef>
#include <typeinfo>
#include <vector>

#include "caffe2/core/blob.h"
#include "caffe2/core/common.h"
#include "caffe2/core/net.h"
#include "caffe2/core/operator_gradient.h"
#include "caffe2/core/registry.h"
#include "caffe2/core/tensor.h"
#include "caffe2/core/workspace.h"
#include "caffe2/utils/proto_utils.h"
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
    CAFFE_CHECK(arg_map_.count(name)) << "Cannot find parameter named " << name;
    MessageType message;
    if (arg_map_[name]->has_s()) {
      CAFFE_CHECK(message.ParseFromString(arg_map_[name]->s()))
          << "Faild to parse content from the string";
    } else {
      CAFFE_VLOG(1) << "Return empty message for parameter " << name;
    }
    return message;
  }
  template <typename MessageType>
  vector<MessageType> GetRepeatedMessageArgument(const string& name) {
    CAFFE_CHECK(arg_map_.count(name)) << "Cannot find parameter named " << name;
    vector<MessageType> messages(arg_map_[name]->strings_size());
    for (int i = 0; i < messages.size(); ++i) {
      CAFFE_CHECK(messages[i].ParseFromString(arg_map_[name]->strings(i)))
          << "Faild to parse content from the string";
    }
    return messages;
  }

  // Get the inputs and outputs as specific types.
  template <typename T>
  inline const T& Input(int idx) {
    CAFFE_DCHECK_LT(idx, inputs_.size());
    return inputs_.at(idx)->template Get<T>();
  }

  template <typename T>
  inline T* Output(int idx) {
    CAFFE_DCHECK_LT(idx, outputs_.size());
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
  // Specify the minimum and maximum number of inputs and outputs.
  // Do not manually override these functions. Instead, use INPUT_OUTPUT_STATS
  // macro below.
  virtual int MinInput() const { return 0; }
  virtual int MaxInput() const { return INT_MAX; }
  virtual int MinOutput() const { return 0; }
  virtual int MaxOutput() const { return INT_MAX; }
  // Specify whether in-place computation is allowed for a given pair of input
  // index and output index. In-place computations are opt-in, meaning that an
  // operator has to explicitly specify that it allows in-place computation.
  // Otherwise, input and output MUST be different.
  // Do not manually override this function if your operator does very simple
  // in-place opt-ins, such as allowing input 0 and output 0 to be inplace.
  // Use OP_IN_PLACE_ALLOWED({0, 0}) macro below.
  virtual bool InplaceAllowed(const int input_id, const int output_id) const {
    return false;
  }

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

// OP_SINGLE_ARG provides a shorter initialization choice for initialization of
// member variables for the class constructors.
#define OP_SINGLE_ARG(type, name, variable, default)                           \
  variable(OperatorBase::GetSingleArgument<type>(name, (default)))

// INPUT_OUTPUT_STATS gives the statistics of the input and output that are
// legal. If the max input/output is not limited, you can specify INT_MAX.
// TODO(Yangqing): If necessary, add ability to specify that n_input = n_output.
#define INPUT_OUTPUT_STATS(min_input, max_input, min_output, max_output)       \
 protected:                                                                    \
  int MinInput() const override { return min_input; }                          \
  int MaxInput() const override { return max_input; }                          \
  int MinOutput() const override { return min_output; }                        \
  int MaxOutput() const override { return max_output; }

// Note that this implementation uses vector so it likely won't work well for
// very large operators, but we should be fine since the InplaceAllowed function
// should be very sparse.
#define IN_PLACE_ALLOWED(...)                                                  \
 protected:                                                                    \
  bool InplaceAllowed(                                                         \
      const int input_id, const int output_id) const override {                \
    const vector<std::pair<int, int> > kVec{__VA_ARGS__};                      \
    auto p = std::make_pair(input_id, output_id);                              \
    return (std::find(kVec.begin(), kVec.end(), p) != kVec.end());             \
  }

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
template <class Context>
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

  inline const Tensor<Context>& Input(int idx) {
    return OperatorBase::template Input<Tensor<Context> >(idx); }
  inline Tensor<Context>* Output(int idx) {
    return OperatorBase::template Output<Tensor<Context> >(idx);
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
  Context device_context_;
  DISABLE_COPY_AND_ASSIGN(Operator);
};

#define USE_OPERATOR_BASE_FUNCTIONS                                            \
  using OperatorBase::HasArgument;                                             \
  using OperatorBase::GetSingleArgument;                                       \
  using OperatorBase::GetRepeatedArgument;                                     \
  using OperatorBase::def;                                                     \
  using OperatorBase::InputIsType;                                             \
  using OperatorBase::InputSize;                                               \
  using OperatorBase::OutputSize

#define USE_OPERATOR_FUNCTIONS(context)                                        \
  USE_OPERATOR_BASE_FUNCTIONS;                                                 \
  using Operator<context>::device_context_;                                    \
  using Operator<context>::Input;                                              \
  using Operator<context>::Output

#define USE_OPERATOR_CONTEXT_FUNCTIONS USE_OPERATOR_FUNCTIONS(Context)

#define USE_SIMPLE_CTOR_DTOR(name)                                             \
  name(const OperatorDef& operator_def, Workspace* ws)                         \
      : Operator<Context>(operator_def, ws) {}                                 \
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

#define REGISTER_CPU_OPERATOR_WITH_ENGINE(name, engine, ...) \
  REGISTER_CLASS(CPUOperatorRegistry, name##:##engine, __VA_ARGS__)

DECLARE_REGISTRY(CUDAOperatorRegistry, OperatorBase,
                 const OperatorDef&, Workspace*);
#define REGISTER_CUDA_OPERATOR_CREATOR(key, ...) \
  REGISTER_CREATOR(CUDAOperatorRegistry, key, __VA_ARGS__)
#define REGISTER_CUDA_OPERATOR(name, ...) \
  REGISTER_CLASS(CUDAOperatorRegistry, name, __VA_ARGS__)

#define REGISTER_CUDA_OPERATOR_WITH_ENGINE(name, engine, ...) \
  REGISTER_CLASS(CUDAOperatorRegistry, name##_ENGINE_##engine, __VA_ARGS__)

// Macros for cudnn since we use it often
#define REGISTER_CUDNN_OPERATOR(name, ...) \
  REGISTER_CUDA_OPERATOR_WITH_ENGINE(name, CUDNN, __VA_ARGS__)

// Creates an operator with the given operator definition.
OperatorBase* CreateOperator(const OperatorDef& operator_def, Workspace* ws);

}  // namespace caffe2

#endif  // CAFFE2_CORE_OPERATOR_H_
