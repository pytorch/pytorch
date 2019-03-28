#pragma once

#include <ATen/core/functional.h>
#include <ATen/core/jit_type.h>
#include <ATen/core/stack.h>
#include <c10/util/Exception.h>
#include <torch/csrc/WindowsTorchApiMacro.h>
#include <torch/csrc/autograd/function_hook.h>
#include <torch/csrc/autograd/variable.h>
#include <torch/csrc/jit/constants.h>
#include <torch/csrc/utils/variadic.h>

#include <ATen/Backtrace.h>

#include <cstdint>
#include <iostream>
#include <memory>
#include <mutex>
#include <unordered_map>
#include <vector>

namespace torch {
namespace jit {
namespace tracer {

using torch::autograd::Variable;
using variable_list = std::vector<Variable>;

struct TORCH_API TracingState
    : public std::enable_shared_from_this<TracingState> {
  TracingState();
  ~TracingState();

  using WeakTensor = at::WeakTensor;

  struct WeakTensorHasher {
    size_t operator()(const WeakTensor& t) const {
      return std::hash<void*>()(t.unsafeGetTensorImpl());
    }
  };

  struct WeakTensorEq {
    bool operator()(const WeakTensor& t1, const WeakTensor& t2) const {
      return t1.is_same(t2);
    }
  };

  struct TracingEnvironmentFrame {
    std::unordered_map<WeakTensor, Value*, WeakTensorHasher, WeakTensorEq>
        value_map;
    // TODO weak refcount
    std::unordered_map<c10::intrusive_ptr<c10::ivalue::Future>, Value*>
        future_map;
  };

  using TracingEnvironmentStack = std::vector<TracingEnvironmentFrame>;

  TracingEnvironmentStack env_stack;
  std::shared_ptr<Graph> graph;
  bool warn = true;
  bool force_outplace = false;
  std::function<std::string(const Variable& var)> lookup_var_name_fn =
      [](const Variable& var) { return ""; };
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
TORCH_API extern const char* LEGACY_CONSTRUCTOR;
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
    getTracingState()->env_stack.emplace_back();
  }

  ~WithNestedTracingFrame() {
    getTracingState()->env_stack.pop_back();
  }
};

} // namespace tracer
} // namespace jit
} // namespace torch
