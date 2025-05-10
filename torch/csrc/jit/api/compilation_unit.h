#pragma once
#include <ATen/core/function.h>
#include <c10/util/Exception.h>
#include <torch/csrc/jit/api/function_impl.h>
#include <torch/csrc/jit/frontend/name_mangler.h>
#include <torch/csrc/jit/frontend/source_range.h>
#include <torch/csrc/jit/ir/ir.h>
#include <torch/csrc/jit/runtime/graph_executor.h>

#include <torch/csrc/Export.h>

#include <ATen/core/function_schema.h>
#include <ATen/core/qualified_name.h>
#include <c10/util/ArrayRef.h>
#include <optional>

#include <functional>
#include <memory>
#include <mutex>
#include <ostream>
#include <string>
#include <unordered_map>
#include <vector>

namespace torch::jit {

struct Def;
struct Property;
struct ClassDef;
struct SugaredValue;
struct Resolver;

using ResolverPtr = std::shared_ptr<Resolver>;
struct Self {
  virtual ~Self() = default;
  virtual std::shared_ptr<SugaredValue> makeSugared(Value* v) const = 0;
  virtual ClassTypePtr getClassType() const = 0;
};

// A CompilationUnit is a list of named Functions
// with helper methods to iterate the list or invoke the function.
// Classes have a CompilationUnit holding the class methods,
// and Modules have a CompilationUnit holding the Functions that
// are used to implement their Methods

struct TORCH_API CompilationUnit {
  enum class FunctionType { Method, Hook, PreHook };
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
    TORCH_WARN(
        "CompilationUnit::set_optimized() is deprecated and has no effect. "
        "Please use setGraphExecutorOptimize()");
  }

  bool is_optimized() const {
    TORCH_WARN(
        "CompilationUnit::is_optimized() is deprecated and always returns true. "
        "Please use getGraphExecutorOptimize()");
    return true;
  }

  // for historic reasons, these are defined in ir_emitter.cpp
  // Returns the list of Functions just defined.
  std::vector<Function*> define(
      const std::optional<c10::QualifiedName>& prefix,
      const std::vector<Property>& properties,
      const std::vector<ResolverPtr>& propResolvers,
      const std::vector<Def>& definitions,
      const std::vector<ResolverPtr>&
          defResolvers, /* determines how we handle free
                     variables in each definition*/
      // if non-null, the first argument to each def, is bound to this value
      const Self* self,
      // see [name mangling]
      bool shouldMangle = false,
      std::optional<size_t> operator_set_version = std::nullopt);

  void define_hooks(
      const std::optional<c10::QualifiedName>& prefix,
      const std::vector<Def>& hookDefs,
      const std::vector<ResolverPtr>& hookResolvers,
      const std::vector<Def>& preHookDefs,
      const std::vector<ResolverPtr>& preHookResolvers,
      const Self* self,
      bool shouldMangle = false);

  // same as above but parse the definitions from source
  // Returns the list of Functions just defined.
  std::vector<Function*> define(
      // prefix namespace to put all the defined functions into
      const std::optional<c10::QualifiedName>& prefix,
      const std::string& source,
      const ResolverPtr& resolver,
      const Self* self);

  void define_interface(
      const c10::QualifiedName& qualifiedName,
      const ClassDef& classDef,
      ResolverPtr rcb,
      bool is_module = false);

  Function* create_function(
      c10::QualifiedName name,
      std::shared_ptr<Graph> graph,
      bool shouldMangle = false) {
    if (shouldMangle) {
      name = mangle(name);
    }
    auto fn = std::make_unique<GraphFunction>(
        std::move(name), std::move(graph), nullptr);
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
  void register_type(c10::NamedTypePtr namedType) {
    // TODO: class types cannot be redefined because we have no way right now
    // of invalidating their methods. NamedTuples are fine though, since they
    // don't have methods.
    TORCH_CHECK(
        0 == classDict_.count(*namedType->name()),
        "class '",
        namedType->name()->qualifiedName(),
        "' already defined.");
    classes_.push_back(std::move(namedType));
    classDict_[*classes_.back()->name()] = classes_.size() - 1;
  }

  c10::ClassTypePtr get_class(const c10::QualifiedName& name) const {
    auto type = get_type(name);
    if (!type) {
      return nullptr;
    }
    return type->cast<c10::ClassType>();
  }

  c10::InterfaceTypePtr get_interface(const c10::QualifiedName& name) const {
    auto type = get_type(name);
    if (!type) {
      return nullptr;
    }
    return type->cast<c10::InterfaceType>();
  }

  c10::TupleTypePtr get_named_tuple(const c10::QualifiedName& name) const {
    for (const auto& cls : classes_) {
      if (cls->name()->qualifiedName() == name.qualifiedName()) {
        return cls->expect<TupleType>();
      }
    }
    return nullptr;
  }

  c10::NamedTypePtr get_type(const c10::QualifiedName& name) const {
    auto it = classDict_.find(name);
    if (it == classDict_.end()) {
      return nullptr;
    }
    return classes_[it->second];
  }

  // For testing: clear all Python-defined classes to ensure that unit tests
  // have isolation.
  void _clear_python_cu() {
    // Delete all the associated class methods
    for (const auto& type : classes_) {
      if (auto cls = type->cast<ClassType>()) {
        for (auto method : cls->methods()) {
          // Tombstone the method in the compilation unit.
          // Don't erase because the dict_
          auto it = dict_.find(method->qualname());
          if (it != dict_.end()) {
            functions_[it->second] = nullptr;
            // Erase in our big lookup table
            dict_.erase(it);
          }
        }
        // Classes can have multiple pointers to the same hook,
        // need to make sure to not delete it twice
        std::unordered_set<Function*> hooks_to_delete;
        for (const auto& hook : cls->getForwardHooks()) {
          hooks_to_delete.insert(hook);
        }
        for (const auto& pre_hook : cls->getForwardPreHooks()) {
          hooks_to_delete.insert(pre_hook);
        }
        for (const auto& hook : hooks_to_delete) {
          // Tombstone the hook in the compilation unit.
          auto it = dict_.find(hook->qualname());
          if (it != dict_.end()) {
            functions_[it->second] = nullptr;
            // Erase in our big lookup table
            dict_.erase(it);
          }
        }
      }
    }
    classes_.clear();
    classDict_.clear();
  }

  // [Internal Only] Remove method.
  // Note Used for freezing.
  void unsafeRemoveMethod(const c10::QualifiedName& method_name) {
    auto it = dict_.find(method_name);
    TORCH_CHECK(
        it != dict_.end(),
        "method '",
        method_name.qualifiedName(),
        "' does not exist.");
    functions_[it->second] = nullptr;
    dict_.erase(it);
  }

  // [name mangling] All code objects must have a unique qualified name in a
  // CompilationUnit. In Python, sometimes functions won't have unique qualified
  // name (for example, nested functions). So we mangle Python functions to
  // ensure that they are uniquely named.
  //
  // We also use mangling to distinguish different Module instances. Since each
  // Module is a singleton class instance, different instances of the same
  // Python Module will have different types but the same qualified name.
  c10::QualifiedName mangle(const c10::QualifiedName& name) const {
    auto mangled = name;
    while (get_type(mangled) || find_function(mangled)) {
      mangled = mangler_.mangle(mangled);
    }
    return mangled;
  }

 private:
  std::unique_ptr<Function> define(
      const std::optional<c10::QualifiedName>& prefix,
      const Def& def,
      const ResolverPtr& resolver,
      const Self* self,
      const std::unordered_map<std::string, Function*>& function_table,
      bool shouldMangle = false,
      FunctionType type = FunctionType::Method,
      std::optional<size_t> version = std::nullopt) const;

  // Define a property on \p self.
  struct PropertyPair;
  PropertyPair define_property(
      const std::optional<c10::QualifiedName>& prefix,
      const Property& prop,
      const ResolverPtr& resolver,
      const Self* self,
      const std::unordered_map<std::string, Function*>& function_table,
      bool shouldMangle = false) const;

  Function& register_function(std::unique_ptr<Function> fn) {
    TORCH_CHECK(
        0 == dict_.count(fn->qualname().qualifiedName()),
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
  std::unordered_map<c10::QualifiedName, size_t> classDict_;

  // [class ownership] Right now there are two relationships between classes
  // and compilation units:
  // 1. Classes have compilation units internally that hold their methods.
  // 2. On load, the TypePtrs of any imported classes are owned by the main
  // module's compilation unit.
  std::vector<c10::NamedTypePtr> classes_;

  mutable NameMangler mangler_;
};

// An owning pointer to a Function. Just a pair of a raw Function ptr and it's
// owning CU. We need this because pybind requires a ref-counted way to refer to
// Functions.
struct StrongFunctionPtr {
  StrongFunctionPtr(std::shared_ptr<CompilationUnit> cu, Function* function)
      : cu_(std::move(cu)), function_(function) {
    TORCH_INTERNAL_ASSERT(cu_);
    TORCH_INTERNAL_ASSERT(function_);
  }
  std::shared_ptr<CompilationUnit> cu_;
  Function* function_;
};

namespace script {
// We once had a `script::` namespace that was deleted. This is for backcompat
// of the public API; new code should not use this type alias.
using CompilationUnit = ::torch::jit::CompilationUnit;
} // namespace script
} // namespace torch::jit
