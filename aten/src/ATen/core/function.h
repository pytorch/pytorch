#pragma once
#include <ATen/core/function_schema.h>
#include <ATen/core/ivalue.h>
#include <ATen/core/qualified_name.h>
#include <mutex>

namespace c10 {
struct FunctionSchema;
};

namespace torch {
namespace jit {

struct Graph;
struct GraphExecutor;

using Stack = std::vector<at::IValue>;
using Kwargs = std::unordered_map<std::string, at::IValue>;
struct RecursiveMethodCallError : public std::exception {};

TORCH_API void preoptimizeGraph(std::shared_ptr<Graph>& graph);

// A Function is a pure Graph with no implicit `self` object bound.
// It contains schema information, and the executor that manages the
// execution of the function. Method is a wrapper around a
// underlying Function that also provides a `self` object.
struct TORCH_API Function {
  virtual bool isGraphFunction() const = 0;

  virtual void run(Stack& stack) = 0;

  virtual void run(Stack&& stack) = 0;

  virtual c10::intrusive_ptr<c10::ivalue::Future> runAsync(Stack& stack) = 0;

  virtual at::IValue operator()(
      std::vector<at::IValue> stack,
      const Kwargs& kwargs = Kwargs()) = 0;

  virtual const c10::QualifiedName& qualname() const = 0;

  virtual const std::string& name() const = 0;

  // if this isn't yet defined, run its method_creator function
  virtual void ensure_defined() = 0;

  virtual std::shared_ptr<Graph> graph() const = 0;

  virtual std::shared_ptr<Graph> optimized_graph() const = 0;

  virtual void clear_execution_info() = 0;

  virtual GraphExecutor& get_executor() = 0;

  virtual const c10::FunctionSchema& getSchema() const = 0;

  virtual size_t num_inputs() const = 0;

  virtual void check_single_output() = 0;

  virtual std::string pretty_print_schema() const = 0;

  virtual Function& setSchema(c10::FunctionSchema schema) = 0;

  virtual ~Function() {}
};
} // namespace jit
} // namespace torch
