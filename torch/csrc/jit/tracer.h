#pragma once

#include <ATen/core/stack.h>
#include <c10/util/Exception.h>
#include <torch/csrc/WindowsTorchApiMacro.h>
#include <ATen/core/jit_type.h>
#include <ATen/core/Dimname.h>
#include <ATen/core/EnableNamedTensor.h>

#include <torch/csrc/utils/variadic.h>

#include <cstdint>
#include <iostream>
#include <memory>
#include <mutex>
#include <unordered_map>
#include <vector>

namespace torch {
namespace jit {
struct Node;
struct Value;
struct Graph;

namespace script {
  struct Module;
}

namespace tracer {

using ::c10::ivalue::Shared;

using ::c10::IValue;
using ::c10::ivalue::Future;

using ::c10::ivalue::ConstantString;
using ::c10::TupleTypePtr;
using ::c10::TupleType;
using ::c10::ArrayRef;

using torch::autograd::Variable;
using variable_list = std::vector<Variable>;

struct TORCH_API TracingState
    : public std::enable_shared_from_this<TracingState> {
  TracingState();
  ~TracingState();

  std::shared_ptr<Graph> graph;
  bool warn = true;
  bool force_outplace = false;
  std::function<std::string(const Variable& var)> lookup_var_name_fn =
      [](const Variable& var) { return ""; };

  void enterFrame() {
    env_stack.emplace_back();
  }

  void leaveFrame() {
    env_stack.pop_back();
  }

  void setValue(const IValue& v, Value* value);
  void delValue(const IValue& var);
  Value* getValue(const IValue& var);
  Value* getOutput(const IValue& var, size_t i);
  bool hasValue(const IValue& var) const;

private:
  using WeakIValue = at::WeakIValue;

  struct WeakIValueHasher {
    size_t operator()(const WeakIValue& t) const {
      return t.hash();
    }
  };

  struct WeakIValueEq {
    bool operator()(const WeakIValue& t1, const WeakIValue& t2) const {
      return t1.isSameIdentity(t2);
    }
  };

  using Frame = std::unordered_map<WeakIValue, Value*, WeakIValueHasher, WeakIValueEq>;
  std::vector<Frame> env_stack;

};

// This is meant to be used as a thread local place, where we can store extra
// info that gets lost when we call into ATen from Python bindings. One example
// for when this happens is when we get an IntArrayRef argument with e.g. sizes for
// view. When tracing, those might be tensors, which let us encode extra data
// dependencies, but once they get to the ATen call where we actually have the
// tracing logic, they get converted into a raw IntArrayRef, and we loose all
// information. To prevent this, we temporarily stash it in here.
struct ArgumentStash {
  struct IntArrayRefTrace : std::vector<Value*> {
    IntArrayRefTrace(int size) : std::vector<Value*>(size, nullptr) {}
  };

  static bool empty() {
    return stash.intlists.empty();
  }

  TORCH_API static void stashIntArrayRefElem(
      const std::string& arg_name,
      size_t size,
      size_t idx,
      const Variable& var);

  static bool hasIntArrayRef(const std::string& arg_name) {
    return stash.intlists.count(arg_name) > 0;
  }

  static IntArrayRefTrace popIntArrayRef(const std::string& arg_name) {
    auto info = std::move(stash.intlists.at(arg_name));
    stash.intlists.erase(arg_name);
    return info;
  }

  // Value stashing: Use these methods to stash arguments which correspond
  // to regular Value*'s in the graph. i.e. they don't require special
  // handling like in the case of IntArrayRefs
  TORCH_API static void stashValue(
      const std::string& arg_name,
      size_t idx,
      const Variable& var,
      const c10::TypePtr& type = nullptr);

  static bool hasValue(const std::string& arg_name) {
    return stash.values.count(arg_name) > 0;
  }

  static Value* popValue(const std::string& arg_name) {
    auto info = stash.values.at(arg_name);
    stash.values.erase(arg_name);
    return info;
  }

 private:
  static thread_local ArgumentStash stash;
  std::unordered_map<std::string, IntArrayRefTrace> intlists;
  std::unordered_map<std::string, Value*> values;
};

// Retrieve or set the current tracing state. Returns a nullptr if tracing is
// disabled.
TORCH_API const std::shared_ptr<TracingState>& getTracingState();
TORCH_API void setTracingState(std::shared_ptr<TracingState> state);

inline bool isTracing() {
  return static_cast<bool>(getTracingState());
}

using warn_fn_type = void (*)(const std::string& msg);
TORCH_API extern const char* WARN_PYTHON_DATAFLOW;
TORCH_API extern const char* WARN_CONSTRUCTOR;
TORCH_API extern const char* WARN_RESIZE;
TORCH_API void _do_warn(const char* _reason, const char* _kind);
inline void warn(const char* _reason, const char* _kind = nullptr) {
  if (const auto& state = getTracingState()) {
    if (!state->warn)
      return;
    _do_warn(_reason, _kind);
  }
}
TORCH_API void setWarn(warn_fn_type fn);

struct TORCH_API NoWarn {
  NoWarn() : state(getTracingState()) {
    if (state) {
      prev = state->warn;
      state->warn = false;
    }
  }
  ~NoWarn() {
    if (state) {
      state->warn = prev;
    }
  }
  std::shared_ptr<TracingState> state;
  bool prev;
};

struct WithNestedTracingFrame {
  WithNestedTracingFrame() {
    getTracingState()->enterFrame();
  }

  ~WithNestedTracingFrame() {
    getTracingState()->leaveFrame();
  }
};
TORCH_API void recordSourceLocation(Node* n);
TORCH_API void setRecordSourceLocation(void (*v)(Node*));

// Having finished adding a new 'node' to the graph IR 'setValueTrace'
// associates this node with an output variable, so that further operations
// involving this variable know which node in the IR to reference.
TORCH_API void setValueTrace(const IValue& v, Value* value);

TORCH_API void delValueTrace(const IValue& var);

TORCH_API std::function<void()> pauseTracing();

TORCH_API Value* getValueTrace(const IValue& var);

TORCH_API std::pair<std::shared_ptr<TracingState>, Stack> trace(
    Stack inputs,
    const std::function<Stack(Stack)>& traced_fn,
    std::function<std::string(const Variable&)> var_name_lookup_fn,
    bool force_outplace = false,
    script::Module* self = nullptr);

TORCH_API void abandon();

// NB: those serve both as an intermediate steps in addInputs below,
// as well as the overloads that terminate template recursion
TORCH_API void addInputs(Node* n, const char* name, int64_t value);
TORCH_API void addInputs(
    Node* n,
    const char* name,
    c10::optional<int64_t> value);
TORCH_API void addInputs(Node* n, const char* name, bool value);
TORCH_API void addInputs(
    Node* n,
    const char* name,
    const c10::optional<bool>& value);
TORCH_API void addInputs(Node* n, const char* name, double value);
TORCH_API void addInputs(Node* n, const char* name, const at::Scalar& value);
TORCH_API void addInputs(
    Node* n,
    const char* name,
    const c10::optional<at::Scalar>& value);
TORCH_API void addInputs(Node* n, const char* name, const at::Tensor& value);
TORCH_API void addInputs(Node* n, const char* name, at::IntArrayRef value);
TORCH_API void addInputs(
    Node* n,
    const char* name,
    at::TensorList value,
    bool allow_undefined = false);
TORCH_API void addInputs(
    Node* n,
    const char* name,
    const ArrayRef<double>& value);
TORCH_API void addInputs(
    Node* n,
    const char* name,
    const std::vector<double>& value);
TORCH_API void addInputs(Node* n, const char* name, const std::string& value);
TORCH_API void addInputs(
    Node* n,
    const char* name,
    const at::TensorOptions& value);
TORCH_API void addInputs(Node* n, const char* name, at::Device value);
TORCH_API void addInputs(Node* n, const char* name, at::Layout value);
TORCH_API void addInputs(Node* n, const char* name, at::ScalarType value);
TORCH_API void addInputs(
    Node* n,
    const char* name,
    const c10::optional<at::ScalarType>& value);
TORCH_API void addInputs(Node* n, const char* name, at::MemoryFormat value);
#ifdef BUILD_NAMEDTENSOR
TORCH_API void addInputs(Node* n, const char* name, c10::optional<at::DimnameList> value);
#endif
TORCH_API void addInputs(
    Node* n,
    const char* name,
    const c10::optional<at::MemoryFormat>& value);
TORCH_API void addInputs(Node* n, const char* name, at::Generator* value);
TORCH_API void addInputs(
    Node* n,
    const char* name,
    const c10::optional<at::Device>& value);
TORCH_API void addInputs(
    Node* n,
    const char* name,
    const c10::optional<at::Layout>& value);
template <typename T>
TORCH_API void addInputs(
    Node* n,
    const char* name,
    const std::vector<T>& value);

template <typename K, typename V>
TORCH_API void addInputs(
    Node* n,
    const char* name,
    const std::unordered_map<K, V>& value);

template <typename T>
void addInputs(Node* n, const char* name, const std::vector<T>& value) {
  AT_ERROR("Tracing a list of arbitrary type is currently not supported!");
}
template <typename K, typename V>
void addInputs(
    Node* n,
    const char* name,
    const std::unordered_map<K, V>& value) {
  AT_ERROR("Tracing a dict of arbitrary types is currently not supported!");
}

template <size_t N>
void addInputs(Node* n, const char* name, std::array<bool, N> value) {
  throw std::runtime_error(
      "Found an unsupported argument type in the JIT tracer. File a bug report.");
}

TORCH_API void ensureUniqueIfOutOfPlaced(
    const char* name,
    const at::Tensor& tensor);

template <
    typename T,
    typename = torch::enable_if_t<
        (!std::is_convertible<torch::decay_t<T>, at::TensorList>::value &&
         !std::is_convertible<torch::decay_t<T>, at::Tensor>::value)>>
void addOutput(Node* node, T&&) {
  AT_ERROR(
      "Found an unsupported argument type ",
      c10::demangle_type<T>(),
      " in the JIT tracer. File a bug report.");
}
TORCH_API void addOutput(Node* node, const at::Tensor& tensor);
TORCH_API void setOutput(Value* value, const at::Tensor& output);
TORCH_API void addOutput(Node* node, const std::vector<at::Tensor>& list);

TORCH_API autograd::Variable getSizeOf(
    const autograd::Variable& var,
    int64_t dim);

} // namespace tracer
} // namespace jit
} // namespace torch
