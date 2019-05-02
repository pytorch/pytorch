#pragma once
#include <c10/util/Exception.h>
#include <torch/csrc/jit/graph_executor.h>
#include <torch/csrc/jit/ir.h>
#include <torch/csrc/jit/source_range.h>

#include <torch/csrc/WindowsTorchApiMacro.h>
#include <torch/csrc/utils/memory.h>

#include <ATen/core/function_schema.h>
#include <c10/util/ArrayRef.h>
#include <c10/util/Optional.h>

#include <functional>
#include <memory>
#include <mutex>
#include <ostream>
#include <string>
#include <unordered_map>
#include <vector>

namespace torch {
namespace jit {

namespace script {

struct Def;
struct SugaredValue;
struct Function;
struct Resolver;

using ResolverPtr = std::shared_ptr<Resolver>;
using Self = std::function<std::shared_ptr<SugaredValue>(Value*)>;
using Kwargs = std::unordered_map<std::string, IValue>;

// A Function is a pure Graph with no implicit `self` object bound.
// It contains schema information, and the executor that manages the
// execution of the function. script::Method is a wrapper around a
// underlying Function that also provides a `self` object.
struct TORCH_API Function {
  Function(
      std::string name,
      bool optimize,
      std::shared_ptr<Graph> graph,
      std::function<void(Function&)> function_creator)
      : name_(std::move(name)),
        graph_(std::move(graph)),
        optimize_(optimize),
        function_creator_(std::move(function_creator)) {}

  void run(Stack& stack) {
    get_executor().run(stack);
  }

  void run(Stack&& stack) {
    run(stack);
  }

  IValue operator()(
      std::vector<IValue> stack,
      const Kwargs& kwargs = Kwargs()) {
    getSchema().checkAndNormalizeInputs(stack, kwargs);
    run(stack);
    return stack.front();
  }

  std::shared_ptr<Graph> graph() const {
    return graph_;
  }

  const std::string& name() const {
    return name_;
  }

  // if this isn't yet defined, run its method_creator function
  void ensure_defined();

  size_t num_inputs() const {
    return graph()->inputs().size();
  }

  Function& setSchema(FunctionSchema schema) {
    schema_ = make_unique<FunctionSchema>(std::move(schema));
    return *this;
  }

  const FunctionSchema& getSchema() const {
    if (schema_ == nullptr) {
      schema_ = make_unique<FunctionSchema>(defaultSchemaFor(*this));
    }
    return *schema_;
  }

  std::string pretty_print_schema() const {
    AT_ASSERT(schema_);
    std::stringstream ss;
    ss << *schema_;
    return ss.str();
  }

  GraphExecutorState getDebugState() {
    return get_executor().getDebugState();
  }

  bool is_optimized() const {
    return optimize_;
  }

  void check_single_output() {
    AT_CHECK(
        graph()->outputs().size() == 1,
        "Method (but not graphs in general) require a single output. Use None/Tuple for 0 or 2+ outputs");
  }

  GraphExecutor& get_executor() {
    std::call_once(executor_init_, [&] {
      check_single_output();
      executor_ = GraphExecutor(graph(), optimize_);
    });
    return executor_;
  }

  // returns nullptr and fills in failure_messages if the callee does not
  // match the functions schema

  // TODO: defined in module.cpp, move to compilation_unit.cpp
  Value* try_emit_call(
      Graph& graph,
      const SourceRange& loc,
      c10::optional<NamedValue> self,
      ArrayRef<NamedValue> args,
      ArrayRef<NamedValue> kwargs,
      std::stringstream& failure_messages,
      bool conv_tensors_to_nums);

  Value* emit_call(
      Graph& graph,
      const SourceRange& loc,
      ArrayRef<NamedValue> args,
      ArrayRef<NamedValue> kwargs);

 private:
  static FunctionSchema defaultSchemaFor(const Function& function) {
    std::vector<Argument> args;
    std::vector<Argument> returns;
    Graph& g = *function.graph();
    size_t num_inputs = function.num_inputs();
    for (size_t i = 0; i < num_inputs; ++i) {
      const Value* v = g.inputs().at(i);
      std::string name = v->hasUniqueName() ? v->uniqueNameBase()
                                            : ("argument_" + std::to_string(i));
      args.emplace_back(std::move(name), unshapedType(g.inputs()[i]->type()));
    }
    for (size_t i = 0; i < g.outputs().size(); ++i) {
      returns.emplace_back("", unshapedType(g.outputs()[i]->type()));
    }
    return {function.name(), "", std::move(args), std::move(returns)};
  }

  std::string name_;
  std::shared_ptr<Graph> graph_; // for debugging and for inlining
  bool optimize_;

  GraphExecutor executor_; // for execution

  std::once_flag executor_init_;

  // an optional function that actually creates the method when
  // emit_call_to(this,...) is first called. this is used by the compiler so
  // that it can construct methods out of order
  std::function<void(Function&)> function_creator_;

  // if absent, then we generate a default schema based on the graph
  // mutable because getSchema caches the default schema if one is requested
  // before a call to setSchema
  mutable std::unique_ptr<FunctionSchema> schema_;
};

// A CompilationUnit is a list of named script::Functions
// with helper methods to iterate the list, or invoke the function.
// Classes have a CompilationUnit holding the class methods
// and Modules also have a CompilationUnit holding the Functions that
// are used to implement their Methods

struct TORCH_API CompilationUnit {
  std::shared_ptr<Function> find_function(const std::string& name) const {
    auto it = dict_.find(name);
    if (it == dict_.end())
      return nullptr;
    return functions_[it->second];
  }

  Function& get_function(const std::string& name) const {
    if (auto r = find_function(name))
      return *r;
    AT_ERROR("attempted to get undefined function ", name);
  }

  void set_optimized(bool o) {
    optimized_ = o;
  }

  bool is_optimized() const {
    return optimized_;
  }

  // for historic reasons, these are defined in compiler.cpp
  void define(
      const std::vector<Def>& definitions,
      const std::vector<ResolverPtr>& resolvers, /* determines how we handle free
                                                 variables in each definition*/
      // if non-null, the first argument to each def, is bound to this value
      const Self& self);

  // same as above but parse the definitions from source
  void define(
      const std::string& source,
      const ResolverPtr& resolver,
      const Self& self);

  std::shared_ptr<Function> create_function(std::string name, std::shared_ptr<Graph> graph) {
    auto fn = std::make_shared<Function>(
        std::move(name), is_optimized(), std::move(graph), nullptr);
    register_function(fn);
    return fn;
  }

  const std::vector<std::shared_ptr<Function>>& get_functions() const {
    return functions_;
  }

  /// Run a method from this compilation.
  ///
  /// For example:
  /// @code
  ///   IValue output = module->run("relu_script", a, b);
  /// @endcode
  ///
  /// To get a compile a module from a source string, see torch::jit::compile
  ///
  /// @param method_name The name of the method to run
  /// @param args Arguments to be passed to the method
  /// @return An IValue containing the return value (or values if it is a tuple)
  /// from the method
  template <typename... Types>
  IValue run_method(const std::string& method_name, Types&&... args) {
    return get_function(method_name)({IValue(std::forward<Types>(args))...});
  }

  void drop_all_functions() {
    dict_.clear();
    functions_.clear();
  }

 private:
  Function& register_function(std::shared_ptr<Function> fn) {
    AT_CHECK(
        0 == dict_.count(fn->name()),
        "method '",
        fn->name(),
        "' already defined.");
    functions_.emplace_back(std::move(fn));
    dict_[functions_.back()->name()] = functions_.size() - 1;
    return *functions_.back();
  }
  std::vector<std::shared_ptr<Function>> functions_;
  // for fast lookup
  std::unordered_map<std::string, size_t> dict_;
  bool optimized_ = true;
};

} // namespace script
} // namespace jit
} // namespace torch
