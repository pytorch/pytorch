#ifndef CAFFE2_CORE_OPERATOR_H_
#define CAFFE2_CORE_OPERATOR_H_

#include <array>
#include <climits>
#include <cstddef>
#include <exception>
#include <typeinfo>
#include <vector>

#include "caffe2/core/blob.h"
#include "caffe2/core/common.h"
#include "caffe2/core/net.h"
#include "caffe2/core/observer.h"
#include "caffe2/core/operator_gradient.h"
#include "caffe2/core/operator_schema.h"
#include "caffe2/core/registry.h"
#include "caffe2/core/tensor.h"
#include "caffe2/core/workspace.h"
#include "caffe2/proto/caffe2.pb.h"
#include "caffe2/utils/proto_utils.h"

namespace caffe2 {

class OperatorBase {
 public:
  explicit OperatorBase(const OperatorDef& operator_def, Workspace* ws);
  virtual ~OperatorBase() noexcept {}

  /** @brief Checks if the operator has an argument of the given name.
   */
  inline bool HasArgument(const string& name) const {
    return arg_helper_.HasArgument(name);
  }

  // Functions that deal with arguments. Basically, this allows us to map an
  // argument name to a specific type of argument that we are trying to access.
  template <typename T>
  inline T GetSingleArgument(const string& name, const T& default_value) const {
    return arg_helper_.template GetSingleArgument<T>(name, default_value);
  }
  template <typename T>
  inline bool HasSingleArgumentOfType(const string& name) const {
    return arg_helper_.template HasSingleArgumentOfType<T>(name);
  }
  template <typename T>
  inline vector<T> GetRepeatedArgument(
      const string& name,
      const vector<T>& default_value = {}) const {
    return arg_helper_.template GetRepeatedArgument<T>(name, default_value);
  }

  // Get the inputs and outputs as specific types.
  template <typename T>
  inline const T& Input(int idx) {
    DCHECK_LT(idx, inputs_.size());
    try {
      return inputs_.at(idx)->template Get<T>();
    } catch (::caffe2::EnforceNotMet& enf) {
      enf.AppendMessage(".\nOffending Blob name: ");
      enf.AppendMessage(operator_def_.input(idx));
      enf.AppendMessage(".\n");
      throw enf;
    }
  }

  template <typename T>
  inline T* Output(int idx) {
    return outputs_.at(idx)->template GetMutable<T>();
  }

  inline const Blob& InputBlob(int idx) {
    return *inputs_.at(idx);
  }

  inline Blob* OutputBlob(int idx) {
    return outputs_.at(idx);
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

  virtual bool Run(int /* unused */ stream_id = 0) {
    CAFFE_NOT_IMPLEMENTED;
  }

  virtual bool RunAsync(int /* unused */ stream_id = 0) {
    return Run(stream_id);
  }

  virtual void AddRelatedBlobInfo(EnforceNotMet* err) {
    bool found_input;
    if (err->caller() != nullptr) {
      for (int i = 0; i < inputs_.size(); i++) {
        if (inputs_[i]->GetRaw() == err->caller()) {
          found_input = true;
          err->AppendMessage("\n** while accessing input: " + def().input(i));
          break;
        }
      }
      for (int i = 0; i < outputs_.size(); i++) {
        if (outputs_[i]->GetRaw() == err->caller()) {
          if (found_input) {
            err->AppendMessage("\n OR ");
          }
          err->AppendMessage("\n** while accessing output: " + def().output(i));
          break;
        }
      }
    }
  }

  inline const OperatorDef& def() const {
    return operator_def_;
  }
  inline const ArgumentHelper& arg_helper() const {
    return arg_helper_;
  }

  void SetObserver(ObserverBase<OperatorBase>* observer) {
    observer_ = observer;
  }

  void RemoveObserver() {
    observer_ = nullptr;
  }

  void RecordLastFailedOpNetPosition() {
    if (net_position_ != kNoNetPositionSet) {
      VLOG(1) << "Operator with id " << net_position_ << " failed";
      operator_ws_->last_failed_op_net_position = net_position_;
    } else {
      VLOG(1) << "Failed operator doesn't have id set";
    }
  }

  void set_net_position(int idx) {
    net_position_ = idx;
  }

 public:
  static constexpr int kNoNetPositionSet = -1;

 protected:
  ObserverBase<OperatorBase>* observer_ = nullptr;
  Workspace* operator_ws_;

 private:
  OperatorDef operator_def_;
  ArgumentHelper arg_helper_;
  vector<const Blob*> inputs_;
  vector<Blob*> outputs_;

  int net_position_{kNoNetPositionSet};

  DISABLE_COPY_AND_ASSIGN(OperatorBase);
};

// If your operator does not need any specialized contructor or destructor,
// you can simply use this to save two lines of code.
#define USE_SIMPLE_BASE_CTOR_DTOR(name)                                        \
  name(const OperatorDef& operator_def, Workspace* ws)                         \
      : OperatorBase(operator_def, ws) {}                                      \
  virtual ~name() noexcept {}

// OP_SINGLE_ARG provides a shorter initialization choice for initialization of
// member variables for the class constructors.
#define OP_SINGLE_ARG(type, name, variable, default)                           \
  variable(OperatorBase::GetSingleArgument<type>(name, (default)))

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
  explicit Operator(const OperatorDef& operator_def, Workspace* ws)
      : OperatorBase(operator_def, ws),
        context_(operator_def.device_option()) {
    // In the constructor, we switch to the device so that the child class
    // constructors will run on that device.
    context_.SwitchToDevice(0);
  }
  ~Operator() noexcept override {}

  inline const Tensor<Context>& Input(int idx) {
    return OperatorBase::template Input<Tensor<Context> >(idx); }
  inline Tensor<Context>* Output(int idx) {
    return OperatorBase::template Output<Tensor<Context>>(idx);
  }

  // The run function of Operator switches to the device, and then carries out
  // the actual computation with RunOnDevice(). You should implement RunOnDevice
  // instead of Run().
  bool Run(int stream_id = 0) final {
    try {
      if (observer_) {
        observer_->Start();
      }
      context_.SwitchToDevice(stream_id);
      bool started = RunOnDevice();
      bool finished = context_.FinishDeviceComputation();
      auto result = started && finished;
      if (!result) {
        this->RecordLastFailedOpNetPosition();
      }
      if (!finished) {
        // FinishDeviceComputation() returning error basically means that there
        // is something wrong with the device (like CUDA) that usually cannot be
        // recovered, so we should log FATAL.
        LOG(FATAL) << "Computation on device returned error in operator\n"
                   << ProtoDebugString(this->def());
      }
      if (observer_) {
        observer_->Stop();
      }
      return result;
    } catch (EnforceNotMet& err) {
      err.AppendMessage("Error from operator: \n" + ProtoDebugString(def()));
      AddRelatedBlobInfo(&err);
      this->RecordLastFailedOpNetPosition();
      throw;
    } catch (...) {
      this->RecordLastFailedOpNetPosition();
      throw;
    }
  }

  bool RunAsync(int stream_id = 0) final {
    try {
      context_.SwitchToDevice(stream_id);
      auto result = RunOnDevice();
      if (!result) {
        this->RecordLastFailedOpNetPosition();
      }
      return result;
    } catch (EnforceNotMet& err) {
      err.AppendMessage("Error from operator: \n" + ProtoDebugString(def()));
      AddRelatedBlobInfo(&err);
      this->RecordLastFailedOpNetPosition();
      throw;
    } catch (...) {
      this->RecordLastFailedOpNetPosition();
      throw;
    }
  }

  virtual bool RunOnDevice() = 0;

 protected:
  Context context_;
};

#define USE_OPERATOR_BASE_FUNCTIONS                                 \
  /* using override */ using OperatorBase::HasArgument;             \
  /* using override */ using OperatorBase::GetSingleArgument;       \
  /* using override */ using OperatorBase::HasSingleArgumentOfType; \
  /* using override */ using OperatorBase::GetRepeatedArgument;     \
  /* using override */ using OperatorBase::def;                     \
  /* using override */ using OperatorBase::InputIsType;             \
  /* using override */ using OperatorBase::InputSize;               \
  /* using override */ using OperatorBase::OutputSize

#define USE_OPERATOR_FUNCTIONS(context)                   \
  USE_OPERATOR_BASE_FUNCTIONS;                            \
  /* using override */ using Operator<context>::context_; \
  /* using override */ using Operator<context>::Input;    \
  /* using override */ using Operator<context>::Output

#define USE_OPERATOR_CONTEXT_FUNCTIONS USE_OPERATOR_FUNCTIONS(Context)

#define USE_SIMPLE_CTOR_DTOR(name)                                             \
  name(const OperatorDef& operator_def, Workspace* ws)                         \
      : Operator<Context>(operator_def, ws) {}                                 \
  virtual ~name() noexcept {}

// Helpers to implement runtime op polymorphism. Often it's convenient to make
// an op work on different input types (e.g. i32 vs i64 indices) or special-case
// it for particular input size (e.g. ScatterWeightedSum for block size of 1
// doesn't need to call Eigen).
//
// DispatchHelper provides compile-time generation of nested "if" statements,
// e.g. `DispatchHelper<FixedValues<1, 4>>::call(this, block_size);`
// unrolls into:
//   if (block_size == 1) {
//     return DoRunWithValue<1>();
//   } else if (block_size = 4) {
//     return DoRunWithValue<4>();
//   } else {
//     return DoRunWithValue<-1>();
//   }`
//
// DoRunWithValue implementation can use template arguments to do "if"
// statements
// or proxy to functions in math.h which often provide fixed size
// implementation.
//
// Similarly `TensorTypes<int32_t, int64_t>(this, Input(0))` provides branching
// based on type of the first input and calls DoRunWithType.
//
// Note, that the same instance of Op class is used as the method, not class is
// templated. We might consider adding static class-level polymorphism later.
//
// Convenient macro USE_DISPATCH_HELPER is provided for declaring friendship in
// case DoRunWithValue or DoRunWithType are declared non-public.

#define USE_DISPATCH_HELPER                           \
  template <typename FirstArg, typename... ExtraArgs> \
  friend struct DispatchHelper

template <int... Values>
struct FixedValues {};

template <typename... Types>
struct TensorTypes {};

// Special tag that can be listed in TensorTypes to denote that a special
// implementation in 'RunWithOtherType' needs to be called instead of failing
// Obviously this needs to be the last item in lists, e.g.
// TensorTypes<float, double, GenericTensorImplementation>
struct GenericTensorImplementation {};

// Same as TensorTypes but call DoRunWithType2
template <typename... Types>
struct TensorTypes2 {};

template <typename Sizes, typename... ExtraArgs>
struct DispatchHelper;

template <int FirstVal, int... Values, typename... ExtraArgs>
struct DispatchHelper<FixedValues<FirstVal, Values...>, ExtraArgs...> {
  template <typename Op>
  static bool call(Op* op, int value) {
    if (FirstVal == value) {
      return op->template DoRunWithValue<ExtraArgs..., FirstVal>();
    }
    return DispatchHelper<FixedValues<Values...>, ExtraArgs...>::template call<
        Op>(op, value);
  }
};

template <typename... ExtraArgs>
struct DispatchHelper<FixedValues<>, ExtraArgs...> {
  template <typename Op>
  static bool call(Op* op, TIndex size) {
    return op->template DoRunWithValue<ExtraArgs..., -1>();
  }
};

#define CAFFE2_DEFINE_TENSOR_TYPES_DISPATCHER(                                 \
    TensorTypes, DoRunWithType, DoRunWithOtherType)                            \
  template <typename FirstType, typename... Types, typename... ExtraArgs>      \
  struct DispatchHelper<TensorTypes<FirstType, Types...>, ExtraArgs...> {      \
    template <typename Op>                                                     \
    static bool call(Op* op, const TypeMeta& meta) {                           \
      static_assert(                                                           \
          !std::is_same<GenericTensorImplementation, FirstType>::value,        \
          "GenericTensorImplementation must be the last in TensorTypes list"); \
      if (meta.Match<FirstType>()) {                                           \
        return op->template DoRunWithType<ExtraArgs..., FirstType>();          \
      }                                                                        \
      return DispatchHelper<TensorTypes<Types...>, ExtraArgs...>::             \
          template call<Op>(op, meta);                                         \
    }                                                                          \
    template <typename Op, typename Context>                                   \
    static bool call(Op* op, const Tensor<Context>& tensor) {                  \
      return call<Op>(op, tensor.meta());                                      \
    }                                                                          \
    template <typename Op>                                                     \
    static bool call(Op* op, const Blob& blob) {                               \
      return call<Op>(op, blob.meta());                                        \
    }                                                                          \
  };                                                                           \
                                                                               \
  template <typename... ExtraArgs>                                             \
  struct DispatchHelper<TensorTypes<>, ExtraArgs...> {                         \
    template <typename Op>                                                     \
    static bool call(Op* /* unused */, const TypeMeta& meta) {                 \
      CAFFE_THROW("Unsupported type of tensor: ", meta.name());                \
    }                                                                          \
    template <typename Op, typename Context>                                   \
    static bool call(Op* op, const Tensor<Context>& tensor) {                  \
      return call<Op>(op, tensor.meta());                                      \
    }                                                                          \
    template <typename Op>                                                     \
    static bool call(Op* op, const Blob& blob) {                               \
      return call<Op>(op, blob.meta());                                        \
    }                                                                          \
  };                                                                           \
                                                                               \
  template <typename... ExtraArgs>                                             \
  struct DispatchHelper<                                                       \
      TensorTypes<GenericTensorImplementation>,                                \
      ExtraArgs...> {                                                          \
    template <typename Op>                                                     \
    static bool call(Op* op, const TypeMeta& meta) {                           \
      return op->template DoRunWithOtherType<ExtraArgs...>();                  \
    }                                                                          \
    template <typename Op, typename Context>                                   \
    static bool call(Op* op, const Tensor<Context>& tensor) {                  \
      return call<Op>(op, tensor.meta());                                      \
    }                                                                          \
    template <typename Op>                                                     \
    static bool call(Op* op, const Blob& blob) {                               \
      return call<Op>(op, blob.meta());                                        \
    }                                                                          \
  };
CAFFE2_DEFINE_TENSOR_TYPES_DISPATCHER(
    TensorTypes,
    DoRunWithType,
    DoRunWithOtherType)
CAFFE2_DEFINE_TENSOR_TYPES_DISPATCHER(
    TensorTypes2,
    DoRunWithType2,
    DoRunWithOtherType2)
#undef CAFFE2_DEFINE_TENSOR_TYPES_DISPATCHER

// The device type registry. This works in two phases:
// (1) gDeviceTypeRegistry() maps the device types values to the actual operator
//     registry function.
// (2) Then, one can call the operator registry function to further create the
//     operators.
typedef Registry<std::string, OperatorBase, const OperatorDef&, Workspace*>
    OperatorRegistry;
typedef Registry<std::string, OperatorBase, const OperatorDef&, Workspace*>* (
    *RegistryFunction)();
std::map<int32_t, OperatorRegistry*>* gDeviceTypeRegistry();

struct DeviceTypeRegisterer {
  explicit DeviceTypeRegisterer(int32_t type, RegistryFunction func) {
    if (gDeviceTypeRegistry()->count(type)) {
      std::cerr << "Device type " << type
                << "registered twice. This should not happen. Did you have "
                   "duplicated numbers assigned to different devices?";
      std::exit(1);
    }
    // Calling the registry function to get the actual registry pointer.
    gDeviceTypeRegistry()->emplace(type, func());
  }
};

#define CAFFE_REGISTER_DEVICE_TYPE(type, registry_function) \
  namespace {                                               \
  static DeviceTypeRegisterer CAFFE_ANONYMOUS_VARIABLE(     \
      DeviceType)(type, &registry_function);                \
  }

// The operator registry. Since we are not expecting a great number of devices,
// we will simply have an if-then type command and allocate the actual
// generation to device-specific registerers.
// Note that although we have CUDA and CUDNN here, the registerers themselves do
// not depend on specific cuda or cudnn libraries. This means that we will be
// able to compile it even when there is no cuda available - we simply do not
// link any cuda or cudnn operators.
CAFFE_DECLARE_REGISTRY(
    CPUOperatorRegistry,
    OperatorBase,
    const OperatorDef&,
    Workspace*);
#define REGISTER_CPU_OPERATOR_CREATOR(key, ...) \
  CAFFE_REGISTER_CREATOR(CPUOperatorRegistry, key, __VA_ARGS__)
#define REGISTER_CPU_OPERATOR(name, ...) \
  CAFFE_REGISTER_CLASS(CPUOperatorRegistry, name, __VA_ARGS__)
#define REGISTER_CPU_OPERATOR_STR(str_name, ...) \
  CAFFE_REGISTER_TYPED_CLASS(CPUOperatorRegistry, str_name, __VA_ARGS__)

#define REGISTER_CPU_OPERATOR_WITH_ENGINE(name, engine, ...) \
  CAFFE_REGISTER_CLASS(CPUOperatorRegistry, name##_ENGINE_##engine, __VA_ARGS__)

CAFFE_DECLARE_REGISTRY(
    CUDAOperatorRegistry,
    OperatorBase,
    const OperatorDef&,
    Workspace*);
#define REGISTER_CUDA_OPERATOR_CREATOR(key, ...) \
  CAFFE_REGISTER_CREATOR(CUDAOperatorRegistry, key, __VA_ARGS__)
#define REGISTER_CUDA_OPERATOR(name, ...) \
  CAFFE_REGISTER_CLASS(CUDAOperatorRegistry, name, __VA_ARGS__)
#define REGISTER_CUDA_OPERATOR_STR(str_name, ...) \
  CAFFE_REGISTER_TYPED_CLASS(CUDAOperatorRegistry, str_name, __VA_ARGS__)

#define REGISTER_CUDA_OPERATOR_WITH_ENGINE(name, engine, ...) \
  CAFFE_REGISTER_CLASS(                                       \
      CUDAOperatorRegistry, name##_ENGINE_##engine, __VA_ARGS__)

// Macros for cudnn since we use it often
#define REGISTER_CUDNN_OPERATOR(name, ...) \
  REGISTER_CUDA_OPERATOR_WITH_ENGINE(name, CUDNN, __VA_ARGS__)

// StaticLinkingProtector is a helper class that ensures that the Caffe2
// library is linked correctly with whole archives (in the case of static
// linking). What happens is that when CreateOperator is called for the first
// time, it instantiates an OperatorLinkingProtector object to check if the
// operator registry is empty. If it is empty, this means that we are not
// properly linking the library.
//
// You should not need to use this class.
struct StaticLinkingProtector {
  StaticLinkingProtector() {
    const int registered_ops = CPUOperatorRegistry()->Keys().size();
    // Note: this is a check failure instead of an exception, because if
    // the linking is wrong, Caffe2 won't be able to run properly anyway,
    // so it's better to fail loud.
    // If Caffe2 is properly linked with whole archive, there should be more
    // than zero registered ops.
    if (registered_ops == 0) {
      LOG(FATAL) <<
        "You might have made a build error: the Caffe2 library does not seem "
        "to be linked with whole-static library option. To do so, use "
        "-Wl,-force_load (clang) or -Wl,--whole-archive (gcc) to link the "
        "Caffe2 library.";
    }
  }
};

// An exception that can be thrown by an operator constructor that notifies
// that it does not support the given setting. This can be usually used for
// specific engines that only implement a subset of the features required by
// the original operator schema.
// TODO(jiayq): make more feature-complete exception message.
class UnsupportedOperatorFeature : public std::exception {
 public:
  UnsupportedOperatorFeature(const string& msg) : msg_(msg) {}
  const char* what() const noexcept override {
    return msg_.c_str();
  }

 private:
  string msg_;
};

// A helper macro that should ONLY be used in the operator constructor to check
// if needed features are met. If not, throws the UnsupportedOperatorFeature
// exception with the given message.
#define OPERATOR_NEEDS_FEATURE(condition, ...)                           \
  if (!(condition)) {                                                    \
    throw UnsupportedOperatorFeature(::caffe2::MakeString(__VA_ARGS__)); \
  }

// Creates an operator with the given operator definition.
// Throws on error and never returns nullptr
unique_ptr<OperatorBase> CreateOperator(
    const OperatorDef& operator_def,
    Workspace* ws,
    int net_position = OperatorBase::kNoNetPositionSet);

TensorShapes InferBlobShapesAndTypesFromWorkspace(
    Workspace* ws,
    const vector<std::unique_ptr<NetDef>>& nets);

TensorShapes InferBlobShapesAndTypesFromMap(
    const CaffeMap<std::string, std::vector<TIndex>>& blob_dimensions,
    const vector<std::unique_ptr<NetDef>>& nets);

}  // namespace caffe2

#endif  // CAFFE2_CORE_OPERATOR_H_
