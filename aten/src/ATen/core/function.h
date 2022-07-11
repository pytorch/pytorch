#pragma once

#include <ATen/core/function_schema.h>
#include <ATen/core/ivalue.h>
#include <ATen/core/qualified_name.h>
#include <c10/util/Exception.h>
#include <c10/util/FunctionRef.h>

namespace c10 {
struct FunctionSchema;
};

namespace at {
TORCH_API void launch(std::function<void()> func);
}

namespace torch {
namespace jit {

struct Graph;
struct Code;

namespace mobile {
struct Code;
}

using Stack = std::vector<at::IValue>;
using Kwargs = std::unordered_map<std::string, at::IValue>;
struct RecursiveMethodCallError : public std::exception {};
using TaskLauncher = std::function<void(std::function<void()>)>;

TORCH_API void preoptimizeGraph(std::shared_ptr<Graph>& graph, bool disable_autocast=false);

// A Function is a pure Graph with no implicit `self` object bound.
// It contains schema information and the executor that manages the
// execution of the function. Method is a wrapper around an
// underlying Function that also provides a `self` object.
struct TORCH_API Function {
  virtual c10::string_view doc_string() const {
    static constexpr c10::string_view no_doc_string = "";
    return no_doc_string;
  }

  virtual bool isGraphFunction() const {
    return false;
  }

  virtual void run(Stack& stack) = 0;

  virtual c10::intrusive_ptr<c10::ivalue::Future> runAsync(
      Stack& /*stack*/,
      TaskLauncher taskLauncher = at::launch) {
    (void)taskLauncher; // Suppress unused variable warning
    TORCH_INTERNAL_ASSERT_DEBUG_ONLY(false);
    return {};
  }

  at::IValue operator()(
    Stack stack,
    const Kwargs& kwargs = Kwargs()) {
    getSchema().checkAndNormalizeInputs(stack, kwargs);
    run(stack);
    return stack.front();
  }

  virtual const c10::QualifiedName& qualname() const = 0;

  const std::string& name() const {
    return qualname().name();
  }

  // if this isn't yet defined, run its method_creator function
  virtual void ensure_defined() = 0;

  virtual const c10::FunctionSchema& getSchema() const = 0;

  virtual size_t num_inputs() const = 0;

  virtual Function& setSchema(c10::FunctionSchema schema) = 0;

  // call() defines how different interpreter implementations interacts with
  // Function objects. Basically interpreters need to provide a callback to
  // communicate to Functions what to do if provided a Code object.
  // Alternatively we could design the signature to return an optional Code
  // object, but that requires special handling the null case in interpreter
  // and the fallback behavior is not well defined by interpreter but rather
  // Function themselves, so a callback approach is more reasonable than
  // returning values.
  // If call() returns true, then callback completes successfully, otherwise
  // call() returns false.

  // Overload for server interpreter, a bailout size is needed for graph executor.
  virtual bool call(Stack&, c10::optional<size_t>, c10::function_ref<void(const Code&)>) {
    TORCH_INTERNAL_ASSERT_DEBUG_ONLY(false);
    return false;
  }

  // Overload for mobile interpreter.
  virtual bool call(Stack&, c10::function_ref<void(const mobile::Code&)>) {
    TORCH_INTERNAL_ASSERT_DEBUG_ONLY(false);
    return false;
  }

  virtual ~Function() {}
};
} // namespace jit
} // namespace torch
