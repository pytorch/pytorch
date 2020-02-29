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
// execution of the function. script::Method is a wrapper around a
// underlying Function that also provides a `self` object.
struct TORCH_API Function {
  virtual void run(Stack& stack) = 0;

  virtual void run(Stack&& stack) = 0;

  virtual at::IValue operator()(
      std::vector<at::IValue> stack,
      const Kwargs& kwargs = Kwargs()) = 0;

  virtual const c10::QualifiedName& qualname() const = 0;

  virtual const std::string& name() const = 0;

  // if this isn't yet defined, run its method_creator function
  virtual void ensure_defined() = 0;

  virtual std::shared_ptr<Graph> graph() const = 0;

  virtual std::shared_ptr<Graph> optimized_graph() const = 0;

  virtual GraphExecutor& get_executor() = 0;

  const c10::FunctionSchema& getSchema() const;

  size_t num_inputs() const;

  void check_single_output();

  std::string pretty_print_schema() const;

  Function& setSchema(c10::FunctionSchema schema);

  virtual ~Function() {}

 private:
  // if absent, then we generate a default schema based on the graph
  // mutable because getSchema caches the default schema if one is requested
  // before a call to setSchema
  mutable std::unique_ptr<c10::FunctionSchema> schema_;
};
} // namespace jit
} // namespace torch
