#pragma once
#include <c10/util/Exception.h>
#include <torch/csrc/jit/function.h>
#include <torch/csrc/jit/graph_executor.h>
#include <torch/csrc/jit/ir.h>
#include <torch/csrc/jit/source_range.h>

#include <torch/csrc/WindowsTorchApiMacro.h>
#include <torch/csrc/utils/memory.h>

#include <ATen/core/function_schema.h>
#include <ATen/core/qualified_name.h>
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
struct Resolver;

using ResolverPtr = std::shared_ptr<Resolver>;
using Self = std::function<std::shared_ptr<SugaredValue>(Value*)>;

// A CompilationUnit is a list of named Functions
// with helper methods to iterate the list, or invoke the function.
// Classes have a CompilationUnit holding the class methods
// and Modules also have a CompilationUnit holding the Functions that
// are used to implement their Methods

struct TORCH_API CompilationUnit {
  // constructor that takes a set of functions to compile using the native
  // resolver
  explicit CompilationUnit(const std::string& source);
  CompilationUnit() = default;

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
      const std::vector<ResolverPtr>&
          resolvers, /* determines how we handle free
                     variables in each definition*/
      // if non-null, the first argument to each def, is bound to this value
      const Self& self);

  // same as above but parse the definitions from source
  void define(
      const std::string& source,
      const ResolverPtr& resolver,
      const Self& self);

  std::shared_ptr<Function> create_function(
      std::string name,
      std::shared_ptr<Graph> graph) {
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

  /**
   * Register a class as being owned by this compilation unit.
   */
  void register_class(ClassTypePtr classType) {
    classes_.push_back(std::move(classType));
  };

  ClassTypePtr get_class(const c10::QualifiedName& name) const {
    for (const auto& cls : classes_) {
      if (cls->qualname() == name.qualifiedName()) {
        return cls;
      }
    }
    return nullptr;
  }

  /**
   * Python compilation unit methods
   *
   * Right now there is a single compilation unit that owns all ScriptClasses
   * defined in Python. Below are accessors methods for it.
   */
  static const CompilationUnit& _get_python_cu_const() {
    return _get_python_cu();
  }
  static CompilationUnit& _get_python_cu() {
    static CompilationUnit pyCu;
    return pyCu;
  }
  // For testing: clear all Python-defined classes to ensure that unit tests
  // have isolation.
  static void _clear_python_cu() {
    _get_python_cu().classes_.clear();
  }

 private:
  Function& register_function(std::shared_ptr<Function> fn) {
    TORCH_CHECK(
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

  // [class owernship] Right now there aree two relationships between classes
  // and compilation units:
  // 1. Classes have compilation units internally that hold their methods.
  // 2. On load, the TypePtrs of any imported classes are owned by the main
  // module's compilation unit.
  std::vector<ClassTypePtr> classes_;
};

} // namespace script
} // namespace jit
} // namespace torch
