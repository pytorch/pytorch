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

TORCH_API void defineMethodsInModule(
    const std::shared_ptr<Module>& m,
    const std::vector<Def>& definitions,
    const std::vector<Resolver>& resolvers, /* determines how we handle free
                                               variables in each definition*/
    const std::shared_ptr<SugaredValue>&
        self /* if non-null, the first argument to each def, is bound to this
                value */
);

// same as above but parse the definitions from source
TORCH_API void defineMethodsInModule(
    const std::shared_ptr<Module>& m,
    const std::string& source,
    const Resolver& resolver,
    const std::shared_ptr<SugaredValue>& self);

TORCH_API void lambdaLiftFork(Node* fork_node);

} // namespace script
} // namespace jit
} // namespace torch
