#ifndef CAFFE2_CORE_OPERATOR_H_
#define CAFFE2_CORE_OPERATOR_H_

#include <array>
#include <climits>
#include <cstddef>
#include <exception>
#include <set>
#include <typeinfo>
#include <vector>

#include "c10/macros/Macros.h"
#include "c10/util/Registry.h"
#include "caffe2/core/blob.h"
#include "caffe2/core/common.h"
#include "caffe2/core/net.h"
#include "caffe2/core/observer.h"
#include "caffe2/core/operator_gradient.h"
#include "caffe2/core/operator_schema.h"
#include "caffe2/core/tensor.h"
#include "caffe2/core/types.h"
#include "caffe2/core/workspace.h"
#include "caffe2/proto/caffe2_pb.h"
#include "caffe2/utils/proto_utils.h"

namespace caffe2 {

class CAFFE2_API OperatorBase;
typedef ObserverBase<OperatorBase> OperatorObserver;

class CAFFE2_API OperatorBase : public Observable<OperatorBase> {
 public:
  explicit OperatorBase(const OperatorDef& operator_def, Workspace* ws);
  virtual ~OperatorBase() noexcept {}

  /** @brief Checks if the operator has an argument of the given name.
   */
  inline bool HasArgument(const string& name) const {
    CAFFE_ENFORCE(operator_def_, "operator_def was null!");
    return ArgumentHelper::HasArgument(*operator_def_, name);
  }

  // Functions that deal with arguments. Basically, this allows us to map an
  // argument name to a specific type of argument that we are trying to access.
  template <typename T>
  inline T GetSingleArgument(const string& name, const T& default_value) const {
    CAFFE_ENFORCE(operator_def_, "operator_def was null!");
    return ArgumentHelper::GetSingleArgument<OperatorDef, T>(
        *operator_def_, name, default_value);
  }
  template <typename T>
  inline bool HasSingleArgumentOfType(const string& name) const {
    CAFFE_ENFORCE(operator_def_, "operator_def was null!");
    return ArgumentHelper::HasSingleArgumentOfType<OperatorDef, T>(
        *operator_def_, name);
  }
  template <typename T>
  inline vector<T> GetRepeatedArgument(
      const string& name,
      const vector<T>& default_value = {}) const {
    CAFFE_ENFORCE(operator_def_, "operator_def was null!");
    return ArgumentHelper::GetRepeatedArgument<OperatorDef, T>(
        *operator_def_, name, default_value);
  }

  // Get the inputs and outputs as specific types.
  template <typename T>
  inline const T& Input(int idx) {
    static_assert(
        !std::is_same<T, Tensor>::value,
        "You should use Input<Tensor>(int, DeviceType) for "
        "Tensor.");
    DCHECK_LT(idx, inputs_.size());
    try {
      return inputs_.at(idx)->template Get<T>();
    } catch (::caffe2::EnforceNotMet& enf) {
      if (has_debug_def()) {
        enf.AppendMessage(".\nOffending Blob name: ");
        enf.AppendMessage(debug_def().input(idx));
        enf.AppendMessage(".\n");
      }
      throw enf;
    }
  }

  // TODO(jerryzh): Remove template
  // and the type argument?
  // This is to keep the API changes minimal and make refactoring
  // a bit easier
  template <typename T>
  inline const T& Input(int idx, DeviceType type) {
    static_assert(
        std::is_same<T, Tensor>::value,
        "Input(int, DeviceType) is only available for Tensor");
    DCHECK_LT(idx, inputs_.size());
    try {
      // TODO(jerryzh): We'll need to check device type in Get<T>() later
      // Get<T>() -> Get<T>(type)
      const auto& tensor = inputs_.at(idx)->template Get<T>();
      return tensor;
    } catch (::caffe2::EnforceNotMet& enf) {
      if (has_debug_def()) {
        enf.AppendMessage(".\nOffending Blob name: ");
        enf.AppendMessage(debug_def().input(idx));
        enf.AppendMessage(".\n");
      }
      throw enf;
    }
  }

  template <typename T>
  inline T* Output(int idx) {
    static_assert(
        !std::is_same<T, Tensor>::value,
        "You should use Output<Tensor>(int, DeviceType) for "
        "Tensor.");
    return outputs_.at(idx)->template GetMutable<T>();
  }

  // TODO(jerryzh): Remove this template
  template <typename T>
  inline T* Output(int idx, DeviceType type) {
    static_assert(
        std::is_same<T, Tensor>::value,
        "Output(int, DeviceType) is only available for Tensor");
    // When you get a Tensor here it is not fully initialized
    return BlobGetMutableTensor(outputs_.at(idx), type);
  }

  inline Tensor*
  OutputTensor(int idx, at::IntList dims, at::TensorOptions options) {
    CAFFE_ENFORCE_WITH_CALLER(
        options.device_opt() != c10::nullopt,
        "device must be provided in option.");
    return BlobGetMutableTensor(outputs_.at(idx), dims, options);
  }

  // Get output Tensor of the operator and CopyFrom the given Tensor
  Tensor* OutputTensorCopyFrom(
      int idx,
      at::TensorOptions options,
      const Tensor& src,
      BaseContext* context = nullptr) {
    Tensor* t = Output<Tensor>(idx, options.device().type());
    // TODO:
    // We plan to use the following:
    // Tensor* t = OutputTensor(idx, src.sizes(), src.options()+options);
    // that is overwrite options of src Tensor
    CAFFE_ENFORCE(
        !t->dtype_initialized() || t->dtype() == src.dtype(),
        "We don't allow a change of data type in OutputTensor");
    t->CopyFrom(src, context);
    return t;
  }

  template <typename T>
  inline T* Output(int idx, T* allocated) {
    outputs_.at(idx)->Reset(allocated);
    return allocated;
  }

  inline const Blob& InputBlob(int idx) {
    return *inputs_.at(idx);
  }

  inline Blob* OutputBlob(int idx) {
    return outputs_.at(idx);
  }

  // Check whether output j is an alias of input i by comparing Blob pointers,
  // note this does not check if the two Blobs points to the same Tensor, or if
  // the Tensor pointers point to the same TensorImpl, or if the Storages alias
  inline bool IsInputOutputAlias(int i, int j) {
    return inputs_.at(i) == outputs_.at(j);
  }

  template <typename T>
  inline bool InputIsType(int idx) {
    static_assert(
        !std::is_same<T, Tensor>::value,
        "You should use InputIsTensorType(int, DeviceType) for "
        "Tensor.");
    return inputs_.at(idx)->template IsType<T>();
  }

  inline bool InputIsTensorType(int idx, DeviceType device_type) {
    return BlobIsTensorType(*inputs_.at(idx), device_type);
  }

  template <typename T>
  inline bool OutputIsType(int idx) {
    static_assert(
        !std::is_same<T, Tensor>::value,
        "You should use OutputIsTensorType(int, DeviceType) for "
        "Tensor.");
    return outputs_.at(idx)->template IsType<T>();
  }

  inline bool OutputIsTensorType(int idx, DeviceType type) {
    return BlobIsTensorType(*outputs_.at(idx), type);
  }

  inline int InputSize() const {
    return inputs_.size();
  }
  inline int OutputSize() const {
    return outputs_.size();
  }
  inline const vector<const Blob*>& Inputs() const { return inputs_; }
  inline const vector<Blob*>& Outputs() { return outputs_; }
  vector<TensorShape> InputTensorShapes() const;

  virtual void WaitEvent(const Event& ev, int /*stream_id */ = -1) {
    ev.Finish();
  }

  inline void Wait(const OperatorBase& other, int stream_id = -1) {
    if (!other.IsEventDisabled()) {
      WaitEvent(other.event(), stream_id);
    }
  }

  virtual void WaitEvents(
      const std::vector<const Event*>& events,
      int /*stream_id*/ = -1) {
    for (const auto& ev : events) {
      ev->Finish();
    }
  }

  virtual void Finish() {
    if (event_) {
      event_->Finish();
    }
  }

  virtual bool Run(int /* unused */ /*stream_id*/ = 0) {
    CAFFE_NOT_IMPLEMENTED;
  }

  virtual bool HasAsyncPart() const {
    return false;
  }

  virtual bool SupportsAsyncScheduling() const {
    return false;
  }

  // RunAsync, if implemenented by the specific operators, will schedule the
  // computation on the corresponding context and record the event in its
  // event_ member object. If the specific operator does not support RunAsync,
  // it will simply be synchronous as a fallback.
  virtual bool RunAsync(int stream_id = 0) {
    try {
      auto result = Run(stream_id);
      if (result) {
        if (HasAsyncPart()) {
          RecordEvent();
        } else {
          SetEventFinished();
        }
      } else {
        SetEventFinished(getErrorMsg().c_str());
      }
      return result;
    } catch (EnforceNotMet& err) {
      SetEventFinishedWithException(err.what());
      throw;
    } catch (const std::exception& err) {
      SetEventFinishedWithException(err.what());
      throw;
    } catch (...) {
      SetEventFinishedWithException(getErrorMsg().c_str());
      throw;
    }
  }

  virtual void AddRelatedBlobInfo(EnforceNotMet* err) {
    if (!has_debug_def()) {
      return;
    }

    bool found_input;
    if (err->caller() != nullptr) {
      for (size_t i = 0; i < inputs_.size(); i++) {
        if (inputs_[i]->GetRaw() == err->caller()) {
          found_input = true;
          err->AppendMessage(
              "\n** while accessing input: " + debug_def().input(i));
          break;
        }
      }
      for (size_t i = 0; i < outputs_.size(); i++) {
        if (outputs_[i]->GetRaw() == err->caller()) {
          if (found_input) {
            err->AppendMessage("\n OR ");
          }
          err->AppendMessage(
              "\n** while accessing output: " + debug_def().output(i));
          break;
        }
      }
    }
  }

  inline const OperatorDef& debug_def() const {
    CAFFE_ENFORCE(has_debug_def(), "operator_def was null!");
    return *operator_def_;
  }

  inline void set_debug_def(
      const std::shared_ptr<const OperatorDef>& operator_def) {
    operator_def_ = operator_def;
  }

  inline bool has_debug_def() const {
    return operator_def_ != nullptr;
  }

 public:
  void RecordLastFailedOpNetPosition() {
    if (net_position_ != kNoNetPositionSet) {
      VLOG(1) << "Operator with id " << net_position_ << " failed";
      operator_ws_->last_failed_op_net_position = net_position_;
    } else {
      VLOG(1) << "Failed operator doesn't have id set";
    }
  }

  int net_position() const {
    return net_position_;
  }

  void set_net_position(int idx) {
    net_position_ = idx;
  }

  const DeviceOption& device_option() const {
    return device_option_;
  }

  const Event& event() const {
    CAFFE_ENFORCE(event_, "Event is disabled");
    return *event_;
  }

  Event& event() {
    CAFFE_ENFORCE(event_, "Event is disabled");
    return *event_;
  }

  void ResetEvent() {
    if (event_) {
      event_->Reset();
    }
  }

  void DisableEvent() {
    event_ = nullptr;
  }

  bool IsEventDisabled() const {
    return !event_;
  }

  // Internal API invoked by observers. Normal callers shouldn't invoke it.
  virtual void SyncDeviceBarrierForObservers() {
    CAFFE_NOT_IMPLEMENTED;
  }

  // Checks whether stream is ready to execute new computation,
  // used in stream allocation optimization to skip stream that is currently
  // busy. Depends on context and operator's device, returns true by default
  virtual bool IsStreamFree(int /* unused */) const {
    return true;
  }

  const std::string& type() const {
    return type_;
  }

  void annotate_engine(const std::string& engine) {
    engine_ = engine;
  }

  const std::string& engine() const {
    return engine_;
  }

  void SetExecutorHelper(ExecutorHelper* helper) {
    helper_ = helper;
  }

  ExecutorHelper* GetExecutorHelper() const {
    return helper_;
  }

 public:
  static const int kNoNetPositionSet = -1;

 private:
  Workspace* operator_ws_;
  std::shared_ptr<const OperatorDef> operator_def_;
  DeviceOption device_option_;
  std::string engine_;
  std::string type_;
  vector<const Blob*> inputs_;
  vector<Blob*> outputs_;

  int net_position_{kNoNetPositionSet};

  ExecutorHelper* helper_ = nullptr;

 protected:
  virtual void RecordEvent(const char* /*err_msg*/ = nullptr) {
    CAFFE_NOT_IMPLEMENTED;
  }

  void SetEventFinished(const char* err_msg = nullptr) {
    if (event_) {
      event_->SetFinished(err_msg);
    }
  }

  void SetEventFinishedWithException(const char* err_msg = nullptr) {
    if (event_) {
      event_->SetFinishedWithException(err_msg);
    }
  }

  std::string getErrorMsg() {
    if (has_debug_def()) {
      return "Error from operator: " + ProtoDebugString(debug_def());
    } else {
      return "Error from operator: no op def";
    }
  }

  // An event used by asynchronous execution.
  std::unique_ptr<Event> event_;

  C10_DISABLE_COPY_AND_ASSIGN(OperatorBase);
};

// If your operator does not need any specialized contructor or destructor,
// you can simply use this to save two lines of code.
#define USE_SIMPLE_BASE_CTOR_DTOR(name)                                        \
  name(const OperatorDef& operator_def, Workspace* ws)                         \
      : OperatorBase(operator_def, ws) {}                                      \
  virtual ~name() noexcept {}

// OP_SINGLE_ARG provides a shorter initialization choice for initialization of
// member variables for the class constructors.
// This is a workaround for CUDA9.2 and GCC7
#if defined(CUDART_VERSION) && CUDART_VERSION >= 9020 && __GNUC__ >= 7
#define OP_SINGLE_ARG(type, name, variable, default)                           \
  variable(this->template GetSingleArgument<type>(name, (default)))
#else
#define OP_SINGLE_ARG(type, name, variable, default)                           \
  variable(OperatorBase::GetSingleArgument<type>(name, (default)))
#endif

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
      : OperatorBase(operator_def, ws), context_(operator_def.device_option()) {
    // In the constructor, we switch to the device so that the child class
    // constructors will run on that device.
    context_.SwitchToDevice(0);
  }
  ~Operator() noexcept override {}

  inline const Tensor& Input(
      int idx,
      DeviceType type = Context::GetDeviceType()) {
    return OperatorBase::template Input<Tensor>(idx, type);
  }

  inline Tensor* Output(int idx, at::IntList dims, at::TensorOptions options) {
    if (options.device_opt() == c10::nullopt) {
      return OperatorBase::OutputTensor(
          idx, dims, at::TensorOptions(options).device(context_.device()));
    }
    return OperatorBase::OutputTensor(idx, dims, options);
  }

  inline Tensor* Output(int idx, DeviceType type = Context::GetDeviceType()) {
    return OperatorBase::template Output<Tensor>(idx, type);
  }

  void WaitEvent(const Event& ev, int stream_id = -1) final {
    if (stream_id >= 0) {
      context_.SwitchToDevice(stream_id);
    }
    context_.WaitEvent(ev);
  }

  void WaitEvents(const std::vector<const Event*>& events, int stream_id = -1)
      final {
    if (stream_id >= 0) {
      context_.SwitchToDevice(stream_id);
    }
    for (const auto& ev : events) {
      context_.WaitEvent(*ev);
    }
  }

  // The run function of Operator switches to the device, and then carries out
  // the actual computation with RunOnDevice(). You should implement RunOnDevice
  // instead of Run().
  // Note: Run does not update operator's event and can be used only with
  // non-async executors that do not rely on events
  bool Run(int stream_id = 0) final {
    try {
      StartAllObservers();

      context_.SwitchToDevice(stream_id);
      bool result = RunOnDevice();
      if (!result) {
        this->RecordLastFailedOpNetPosition();
      }
      context_.FinishDeviceComputation(); // throws on error

      StopAllObservers();

      return result;
    } catch (EnforceNotMet& err) {
      if (has_debug_def()) {
        err.AppendMessage(
            "Error from operator: \n" + ProtoDebugString(debug_def()));
        AddRelatedBlobInfo(&err);
      }
      this->RecordLastFailedOpNetPosition();
      StopAllObservers();
      throw;
    } catch (...) {
      this->RecordLastFailedOpNetPosition();
      StopAllObservers();
      throw;
    }
  }

  bool RunAsync(int stream_id = 0) final {
    try {
      StartAllObservers();

      context_.SwitchToDevice(stream_id);
      auto result = RunOnDevice();
      if (result) {
        if (HasAsyncPart()) {
          RecordEvent();
        } else {
          // Manually set CPU operator's event status to finished,
          // unless this is an async CPU operator
          SetEventFinished();
        }
      } else {
        SetEventFinished(getErrorMsg().c_str());
        this->RecordLastFailedOpNetPosition();
      }

      StopAllObservers();

      return result;
    } catch (EnforceNotMet& err) {
      if (has_debug_def()) {
        err.AppendMessage(
            "Error from operator: \n" + ProtoDebugString(debug_def()));
        AddRelatedBlobInfo(&err);
      }
      SetEventFinishedWithException(err.what());
      this->RecordLastFailedOpNetPosition();
      StopAllObservers();
      throw;
    } catch (const std::exception& err) {
      SetEventFinishedWithException(err.what());
      this->RecordLastFailedOpNetPosition();
      StopAllObservers();
      throw;
    } catch (...) {
      SetEventFinishedWithException(getErrorMsg().c_str());
      this->RecordLastFailedOpNetPosition();
      StopAllObservers();
      throw;
    }
  }

  bool IsStreamFree(int stream_id) const override {
    return context_.IsStreamFree(device_option(), stream_id);
  }

  virtual bool RunOnDevice() = 0;

  // Returns whether operator has async on device part.
  // CUDA operators by default have async parts, CPU operators by default
  // don't have async parts and are finished after RunOnDevice call.
  // Events of operators that don't have async parts are automatically set
  // to finished state by RunAsync.
  // Defaulting to the value from context (true for CUDA, false for CPU).
  // Override in case of async CPU operators
  // Async CPU operators are expected to catch all exceptions in async parts
  // and set Event to finished/failed state with Event::SetFinished or
  // SetFinishedWithException call.
  bool HasAsyncPart() const override {
    return context_.HasAsyncPartDefault();
  }

  // Returns whether operator's RunOnDevice schedules async on device part and
  // can be run without waiting for parent operator's async part to be finished
  // on the same device.
  // Note: when true, RunOnDevice must not access the content of the input blobs
  // as they might not be computed yet
  // Note: when true, operator's device needs to support async scheduling:
  //  - supports concept of streams: async ops scheduled on the same stream are
  //    guaranteed to be executed in the same order they were scheduled
  //  - provides non-blocking cross device/cross stream synchronization
  //    primitives
  //
  // By default, assuming an op with an async part can be scheduled
  // asynchronously if device supports async scheduling
  bool SupportsAsyncScheduling() const override {
    return HasAsyncPart() && context_.SupportsAsyncScheduling();
  }

  void SyncDeviceBarrierForObservers() override {
    context_.FinishDeviceComputation();
  }

  const Context* getContext() const {
    return &context_;
  }

 protected:
  void RecordEvent(const char* err_msg = nullptr) final {
    if (event_) {
      context_.Record(event_.get(), err_msg);
    }
  }

  Context context_;
};

#define USE_OPERATOR_BASE_FUNCTIONS                                 \
  /* using override */ using OperatorBase::HasArgument;             \
  /* using override */ using OperatorBase::GetSingleArgument;       \
  /* using override */ using OperatorBase::HasSingleArgumentOfType; \
  /* using override */ using OperatorBase::GetRepeatedArgument;     \
  /* using override */ using OperatorBase::InputIsType;             \
  /* using override */ using OperatorBase::InputSize;               \
  /* using override */ using OperatorBase::Output;                  \
  /* using override */ using OperatorBase::Input;                   \
  /* using override */ using OperatorBase::OutputSize;              \
  /* using override */ using OperatorBase::IsInputOutputAlias

#define USE_OPERATOR_FUNCTIONS(context)                    \
  USE_OPERATOR_BASE_FUNCTIONS;                             \
  /* using override */ using Operator<context>::context_;  \
  /* using override */ using Operator<context>::Input;     \
  /* using override */ using Operator<context>::InputBlob; \
  /* using override */ using Operator<context>::Output;    \
  /* using override */ using Operator<context>::OutputBlob

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
  static bool call(Op* op, int64_t /*size*/) {
    return op->template DoRunWithValue<ExtraArgs..., -1>();
  }
};

#define C10_DEFINE_TENSOR_TYPES_DISPATCHER(                                    \
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
    template <typename Op>                                                     \
    static bool call(Op* op, const Tensor& tensor) {                           \
      return call<Op>(op, tensor.dtype());                                     \
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
    template <typename Op>                                                     \
    static bool call(Op* op, const Tensor& tensor) {                           \
      return call<Op>(op, tensor.dtype());                                     \
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
    static bool call(Op* op, const TypeMeta&) {                                \
      return op->template DoRunWithOtherType<ExtraArgs...>();                  \
    }                                                                          \
    template <typename Op>                                                     \
    static bool call(Op* op, const Tensor& tensor) {                           \
      return call<Op>(op, tensor.dtype());                                     \
    }                                                                          \
    template <typename Op>                                                     \
    static bool call(Op* op, const Blob& blob) {                               \
      return call<Op>(op, blob.meta());                                        \
    }                                                                          \
  };
C10_DEFINE_TENSOR_TYPES_DISPATCHER(
    TensorTypes,
    DoRunWithType,
    DoRunWithOtherType)
C10_DEFINE_TENSOR_TYPES_DISPATCHER(
    TensorTypes2,
    DoRunWithType2,
    DoRunWithOtherType2)
#undef C10_DEFINE_TENSOR_TYPES_DISPATCHER

// The device type registry. This works in two phases:
// (1) gDeviceTypeRegistry() maps the device types values to the actual operator
//     registry function.
// (2) Then, one can call the operator registry function to further create the
//     operators.
typedef c10::Registry<
    std::string,
    std::unique_ptr<OperatorBase>,
    const OperatorDef&,
    Workspace*>
    OperatorRegistry;
typedef c10::Registry<
    std::string,
    std::unique_ptr<OperatorBase>,
    const OperatorDef&,
    Workspace*>* (*RegistryFunction)();
CAFFE2_API std::map<DeviceType, OperatorRegistry*>* gDeviceTypeRegistry();

struct CAFFE2_API DeviceTypeRegisterer {
  explicit DeviceTypeRegisterer(DeviceType type, RegistryFunction func) {
    if (gDeviceTypeRegistry()->count(type)) {
      std::cerr << "Device type " << DeviceTypeName(type)
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
  static DeviceTypeRegisterer C10_ANONYMOUS_VARIABLE(       \
      DeviceType)(type, &registry_function);                \
  }

// The operator registry. Since we are not expecting a great number of devices,
// we will simply have an if-then type command and allocate the actual
// generation to device-specific registerers.
// Note that although we have CUDA and CUDNN here, the registerers themselves do
// not depend on specific cuda or cudnn libraries. This means that we will be
// able to compile it even when there is no cuda available - we simply do not
// link any cuda or cudnn operators.
C10_DECLARE_REGISTRY(
    CPUOperatorRegistry,
    OperatorBase,
    const OperatorDef&,
    Workspace*);
#define REGISTER_CPU_OPERATOR_CREATOR(key, ...) \
  C10_REGISTER_CREATOR(CPUOperatorRegistry, key, __VA_ARGS__)
#define REGISTER_CPU_OPERATOR(name, ...)                           \
  C10_IMPORT void CAFFE2_PLEASE_ADD_OPERATOR_SCHEMA_FOR_##name();  \
  static void CAFFE2_UNUSED CAFFE_ANONYMOUS_VARIABLE_CPU##name() { \
    CAFFE2_PLEASE_ADD_OPERATOR_SCHEMA_FOR_##name();                \
  }                                                                \
  C10_REGISTER_CLASS(CPUOperatorRegistry, name, __VA_ARGS__)
#define REGISTER_CPU_OPERATOR_STR(str_name, ...) \
  C10_REGISTER_TYPED_CLASS(CPUOperatorRegistry, str_name, __VA_ARGS__)

#define REGISTER_CPU_OPERATOR_WITH_ENGINE(name, engine, ...) \
  C10_REGISTER_CLASS(CPUOperatorRegistry, name##_ENGINE_##engine, __VA_ARGS__)

// Use these macros to register gradient operators.  They can be automatically
// excluded from builds that don't need them (e.g., mobile).
#ifdef CAFFE2_NO_GRADIENT_OPS
#define REGISTER_CPU_GRADIENT_OPERATOR(...) /* No gradients. */
#else
#define REGISTER_CPU_GRADIENT_OPERATOR(...) \
  MACRO_EXPAND(REGISTER_CPU_OPERATOR(__VA_ARGS__))
#endif

C10_DECLARE_REGISTRY(
    CUDAOperatorRegistry,
    OperatorBase,
    const OperatorDef&,
    Workspace*);
#define REGISTER_CUDA_OPERATOR_CREATOR(key, ...) \
  C10_REGISTER_CREATOR(CUDAOperatorRegistry, key, __VA_ARGS__)
#define REGISTER_CUDA_OPERATOR(name, ...)                           \
  C10_IMPORT void CAFFE2_PLEASE_ADD_OPERATOR_SCHEMA_FOR_##name();   \
  static void CAFFE2_UNUSED CAFFE_ANONYMOUS_VARIABLE_CUDA##name() { \
    CAFFE2_PLEASE_ADD_OPERATOR_SCHEMA_FOR_##name();                 \
  }                                                                 \
  C10_REGISTER_CLASS(CUDAOperatorRegistry, name, __VA_ARGS__)
#define REGISTER_CUDA_OPERATOR_STR(str_name, ...) \
  C10_REGISTER_TYPED_CLASS(CUDAOperatorRegistry, str_name, __VA_ARGS__)

#define REGISTER_CUDA_OPERATOR_WITH_ENGINE(name, engine, ...) \
  C10_REGISTER_CLASS(CUDAOperatorRegistry, name##_ENGINE_##engine, __VA_ARGS__)

// Macros for cudnn since we use it often
#define REGISTER_CUDNN_OPERATOR(name, ...) \
  REGISTER_CUDA_OPERATOR_WITH_ENGINE(name, CUDNN, __VA_ARGS__)

// Macros for HIP operators
C10_DECLARE_REGISTRY(
    HIPOperatorRegistry,
    OperatorBase,
    const OperatorDef&,
    Workspace*);
#define REGISTER_HIP_OPERATOR_CREATOR(key, ...) \
  C10_REGISTER_CREATOR(HIPOperatorRegistry, key, __VA_ARGS__)
#define REGISTER_HIP_OPERATOR(name, ...)                           \
  C10_IMPORT void CAFFE2_PLEASE_ADD_OPERATOR_SCHEMA_FOR_##name();  \
  static void CAFFE2_UNUSED CAFFE_ANONYMOUS_VARIABLE_HIP##name() { \
    CAFFE2_PLEASE_ADD_OPERATOR_SCHEMA_FOR_##name();                \
  }                                                                \
  C10_REGISTER_CLASS(HIPOperatorRegistry, name, __VA_ARGS__)
#define REGISTER_HIP_OPERATOR_STR(str_name, ...) \
  C10_REGISTER_TYPED_CLASS(HIPOperatorRegistry, str_name, __VA_ARGS__)

#define REGISTER_HIP_OPERATOR_WITH_ENGINE(name, engine, ...) \
  C10_REGISTER_CLASS(HIPOperatorRegistry, name##_ENGINE_##engine, __VA_ARGS__)

#define REGISTER_MIOPEN_OPERATOR(name, ...) \
  REGISTER_HIP_OPERATOR_WITH_ENGINE(name, MIOPEN, __VA_ARGS__) \
  REGISTER_HIP_OPERATOR_WITH_ENGINE(name, CUDNN, __VA_ARGS__) // Make CUDNN an alias of MIOPEN for HIP ops

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
class CAFFE2_API UnsupportedOperatorFeature : public std::exception {
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
#define OPERATOR_NEEDS_FEATURE(condition, ...)                 \
  if (!(condition)) {                                          \
    throw UnsupportedOperatorFeature(::c10::str(__VA_ARGS__)); \
  }

// Creates an operator with the given operator definition.
// Throws on error and never returns nullptr
CAFFE2_API unique_ptr<OperatorBase> CreateOperator(
    const OperatorDef& operator_def,
    Workspace* ws,
    int net_position = OperatorBase::kNoNetPositionSet);

CAFFE2_API const std::string OpRegistryKey(
    const std::string& op_type,
    const std::string& engine = "");

// User can set the preferred engines as a list of engine names, in
// descending order of preference.
using EnginePrefType = std::vector<std::string>;
// {device_type -> {operator_name -> EnginePrefType}}
using PerOpEnginePrefType =
    CaffeMap<DeviceType, CaffeMap<std::string, EnginePrefType>>;
// {device_type -> EnginePrefType}
using GlobalEnginePrefType = CaffeMap<DeviceType, EnginePrefType>;
CAFFE2_API void SetPerOpEnginePref(const PerOpEnginePrefType& per_op_engine_pref);
CAFFE2_API void SetGlobalEnginePref(const GlobalEnginePrefType& global_engine_pref);
CAFFE2_API void SetEnginePref(
    const PerOpEnginePrefType& per_op_engine_pref,
    const GlobalEnginePrefType& global_engine_pref);
CAFFE2_API void SetOpEnginePref(
    const std::string& op_type,
    const CaffeMap<DeviceType, EnginePrefType>& op_pref);

CAFFE2_API TensorShape GetTensorShapeOfBlob(const Blob* b);

CAFFE2_API TensorShapes InferBlobShapesAndTypes(
    CaffeMap<string, TensorShape>& blob_desc,
    const vector<NetDef*>& nets);

CAFFE2_API TensorShapes InferBlobShapesAndTypesFromWorkspace(
    Workspace* ws,
    const vector<NetDef*>& nets);

CAFFE2_API TensorShapes InferBlobShapesAndTypesFromMap(
    const CaffeMap<std::string, std::vector<int64_t>>& blob_dimensions,
    const vector<NetDef*>& nets);

CAFFE2_API TensorShapes InferBlobShapesAndTypesFromMap(
    const CaffeMap<std::string, std::vector<int64_t>>& blob_dimensions,
    const CaffeMap<std::string, TensorProto_DataType>& blob_types,
    const vector<NetDef*>& nets);

CAFFE2_API std::map<string, std::pair<DeviceOption, DeviceOption>> ValidateTensorDevices(
    OperatorBase& op,
    const OperatorDef& op_def);

// Get a set of registered operator names
CAFFE2_API std::set<std::string> GetRegisteredOperators();

// Operator logging capabilities
CAFFE2_API void SetOperatorLogger(std::function<void(const OperatorDef&)> tracer);
std::function<void(const OperatorDef&)> GetOperatorLogger();

}  // namespace caffe2

#endif  // CAFFE2_CORE_OPERATOR_H_
