#pragma once
#include <functional>
#include <memory>
#include <string>

#include <torch/csrc/jit/ir.h>
#include <torch/csrc/jit/script/error_report.h>
#include <torch/csrc/jit/script/module.h>
#include <torch/csrc/jit/script/sugared_value.h>
#include <torch/csrc/jit/script/tree_views.h>

namespace torch {
namespace jit {
namespace script {

using Resolver = std::function<std::shared_ptr<
    SugaredValue>(const std::string& name, Method& m, const SourceRange& loc)>;

inline std::shared_ptr<SugaredValue> nativeResolver(
    const std::string& name,
    Method& m,
    const SourceRange& loc) {
  if (name == "torch") {
    return std::make_shared<BuiltinModule>("aten");
  }
  return nullptr;
}

// Represents the `self` argument to a method. This wrapper class is necessary
// because sometimes `self` sometimes is first class and sometimes not.
//
// `self` is first class when it refers to a ClassType. It will be bound as a
// graph input argument.
// `self` is sugared when it refers to a ModuleValue.
class Self {
 public:
  explicit Self(std::shared_ptr<SugaredValue> sugared)
      : sugared_(std::move(sugared)) {}
  explicit Self(ClassTypePtr type) : firstClass_(std::move(type)) {}

  ClassTypePtr asFirstClass() const {
    return firstClass_;
  }
  std::shared_ptr<SugaredValue> asSugared() const {
    return sugared_;
  }

 private:
  // Used when `self` is not first-class and so we don't represent it in the
  // graph. This is only ModuleValue.
  std::shared_ptr<SugaredValue> sugared_ = nullptr;
  // Used when `self` is a first-class type
  ClassTypePtr firstClass_ = nullptr;
};

TORCH_API void defineMethodsInModule(
    const std::shared_ptr<Module>& m,
    const std::vector<Def>& definitions,
    const std::vector<Resolver>& resolvers, /* determines how we handle free
                                               variables in each definition*/
    // if non-null, the first argument to each def, is bound to this value
    const c10::optional<Self>& self,
    const c10::optional<std::string> class_namespace = c10::nullopt);

// same as above but parse the definitions from source
TORCH_API void defineMethodsInModule(
    const std::shared_ptr<Module>& m,
    const std::string& source,
    const Resolver& resolver,
    const c10::optional<Self>& self,
    const c10::optional<std::string> class_namespace = c10::nullopt);

TORCH_API void lambdaLiftFork(Node* fork_node);

} // namespace script
} // namespace jit
} // namespace torch
