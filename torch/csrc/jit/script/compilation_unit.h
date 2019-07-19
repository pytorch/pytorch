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
struct Self {
  virtual ~Self() {}
  virtual std::shared_ptr<SugaredValue> makeSugared(Value* v) const = 0;
  virtual ClassTypePtr getClassType() const = 0;
};

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

  CompilationUnit& operator=(CompilationUnit&&) = default;
  CompilationUnit(CompilationUnit&&) = default;
  CompilationUnit& operator=(const CompilationUnit&) = delete;
  CompilationUnit(const CompilationUnit&) = delete;

  Function* find_function(const c10::QualifiedName& name) const {
    auto it = dict_.find(name);
    if (it == dict_.end()) {
      return nullptr;
    }
    return functions_[it->second].get();
  }

  Function& get_function(const c10::QualifiedName& name) const {
    if (auto r = find_function(name)) {
      return *r;
    }
    TORCH_CHECK(false, "attempted to get undefined function ", name.name());
  }

  void set_optimized(bool o) {
    optimized_ = o;
  }

  bool is_optimized() const {
    return optimized_;
  }

  // for historic reasons, these are defined in compiler.cpp
  // Returns the list of Function's just defined.
  std::vector<Function*> define(
      const c10::optional<c10::QualifiedName>& prefix,
      const std::vector<Def>& definitions,
      const std::vector<ResolverPtr>&
          resolvers, /* determines how we handle free
                     variables in each definition*/
      // if non-null, the first argument to each def, is bound to this value
      const Self* self);

  // same as above but parse the definitions from source
  // Returns the list of Function's just defined.
  std::vector<Function*> define(
      // prefix namespace to put all the defined functions into
      const c10::optional<c10::QualifiedName>& prefix,
      const std::string& source,
      const ResolverPtr& resolver,
      const Self* self);

  Function* create_function(
      c10::QualifiedName name,
      std::shared_ptr<Graph> graph) {
    auto fn = torch::make_unique<Function>(
        std::move(name), is_optimized(), std::move(graph), nullptr);
    auto ret = fn.get();
    register_function(std::move(fn));
    return ret;
  }

  std::vector<Function*> get_functions() const {
    return fmap(functions_, [](const std::unique_ptr<Function>& fn) {
      return fn.get();
    });
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
  IValue run_method(const c10::QualifiedName& method_name, Types&&... args) {
    return get_function(method_name)({IValue(std::forward<Types>(args))...});
  }

  void drop_all_functions() {
    dict_.clear();
    functions_.clear();
  }

  /**
   * Register a class as being owned by this compilation unit.
   */
  void register_class(c10::NamedTypePtr classType) {
    classes_.push_back(std::move(classType));
  };

  c10::ClassTypePtr get_class(const c10::QualifiedName& name) const {
    for (const auto& cls : classes_) {
      if (cls->qualname() == name.qualifiedName()) {
        return cls->expect<ClassType>();
      }
    }
    return nullptr;
  }

  c10::TupleTypePtr get_named_tuple(const c10::QualifiedName& name) const {
    for (const auto& cls : classes_) {
      if (cls->qualname() == name.qualifiedName()) {
        return cls->expect<TupleType>();
      }
    }
    return nullptr;
  }

  c10::NamedTypePtr get_type(const c10::QualifiedName& name) const {
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
  static std::shared_ptr<CompilationUnit> _get_python_cu_const() {
    return _get_python_cu();
  }
  static std::shared_ptr<CompilationUnit> _get_python_cu() {
    static auto pyCu = std::make_shared<CompilationUnit>();
    return pyCu;
  }
  // For testing: clear all Python-defined classes to ensure that unit tests
  // have isolation.
  static void _clear_python_cu() {
    _get_python_cu()->classes_.clear();
    _get_python_cu()->functions_.clear();
    _get_python_cu()->dict_.clear();
  }

 private:
  std::unique_ptr<Function> define(
      const c10::optional<c10::QualifiedName>& prefix,
      const Def& def,
      const ResolverPtr& resolver,
      const Self* self,
      const std::unordered_map<std::string, Function*>& function_table) const;

  Function& register_function(std::unique_ptr<Function> fn) {
    TORCH_CHECK(
        0 == dict_.count(fn->qualname()),
        "method '",
        fn->qualname().qualifiedName(),
        "' already defined.");
    functions_.emplace_back(std::move(fn));
    dict_[functions_.back()->qualname()] = functions_.size() - 1;
    return *functions_.back();
  }
  std::vector<std::unique_ptr<Function>> functions_;
  // for fast lookup
  std::unordered_map<c10::QualifiedName, size_t> dict_;
  bool optimized_ = true;

  // [class owernship] Right now there aree two relationships between classes
  // and compilation units:
  // 1. Classes have compilation units internally that hold their methods.
  // 2. On load, the TypePtrs of any imported classes are owned by the main
  // module's compilation unit.
  std::vector<c10::NamedTypePtr> classes_;
};

} // namespace script

// An owning pointer to a Function. Just a pair of a raw Function ptr and it's
// owning CU. We need this because pybind requires a ref-counted way to refer to
// Functions.
struct StrongFunctionPtr {
  StrongFunctionPtr(
      std::shared_ptr<script::CompilationUnit> cu,
      Function* function)
      : cu_(std::move(cu)), function_(function) {
    TORCH_INTERNAL_ASSERT(cu_);
    TORCH_INTERNAL_ASSERT(function_);
  }
  std::shared_ptr<script::CompilationUnit> cu_;
  Function* function_;
};
} // namespace jit
} // namespace torch
