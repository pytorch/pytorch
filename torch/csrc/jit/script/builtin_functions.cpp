#include "torch/csrc/jit/script/builtin_functions.h"
#include "torch/csrc/api/include/torch/jit.h"
#include "torch/csrc/jit/code_template.h"

namespace torch { namespace jit { namespace script {

auto scalar_operators_source = CodeTemplate(
R"SCRIPT(
def mul(a : ${Scalar}, b : Tensor) -> Tensor:
  return b * a
def add(a : ${Scalar}, b : Tensor) -> Tensor:
  return b + a
def ne(a : ${Scalar}, b : Tensor) -> Tensor:
  return b != a
def eq(a : ${Scalar}, b : Tensor) -> Tensor:
  return b == a
def lt(a : ${Scalar}, b : Tensor) -> Tensor:
  return b > a
def le(a : ${Scalar}, b : Tensor) -> Tensor:
  return b >= a
def gt(a : ${Scalar}, b : Tensor) -> Tensor:
  return b < a
def ge(a : ${Scalar}, b : Tensor) -> Tensor:
  return b <= a
def sub(a : ${Scalar}, b : Tensor) -> Tensor:
  return torch.neg(b) + a
def div(a : ${Scalar}, b : Tensor) -> Tensor:
  return torch.reciprocal(b) * a
)SCRIPT");

struct BuiltinFunctionRegistry {

  const std::vector<Method*>& getAllBuiltinFunctionsFor(Symbol name) {
    const static std::vector<Method*> empty;
    // when initializing the builtin function library, we will re-enter
    // getAllBuiltinFunctionsFor since it is called in the compiler to
    // lookup builtins and initializing the builtin functions calls the compiler.
    // To avoid deadlocking, we use a recursive mutex (same thread can re-lock,
    // the mutex without waiting), and report no loaded builtins during init.
    std::lock_guard<std::recursive_mutex> guard(mutex);
    if(state == INTIIALIZING) {
      return empty;
    } else if (state == UNINITIALIZED) {
      state = INTIIALIZING;
      loadBuiltinFunctions();
      state = INITIALIZED;
    }
    JIT_ASSERT(state == INITIALIZED);
    auto it = builtins_by_name.find(name);
    if(it == builtins_by_name.end())
      return empty;
    return it->second;
  }
private:
  void loadSource(const std::string& source) {
    auto module = std::make_shared<script::Module>();
    defineMethodsInModule(
        *module, source, script::nativeResolver, /*self=*/nullptr);
    modules.push_back(module);
    for (auto& method : module->get_methods()) {
      builtins_by_name[Symbol::fromQualString("aten::" + method.key)].push_back(
          method.value.get());
    }
  }
  void loadBuiltinFunctions() {
    for(auto scalar : {"float", "int"}) {
      TemplateEnv env;
      env.s("Scalar", scalar);
      loadSource(scalar_operators_source.format(env));
    }
  }
  enum {UNINITIALIZED, INTIIALIZING, INITIALIZED} state = UNINITIALIZED;
  std::recursive_mutex mutex;
  std::vector<std::shared_ptr<Module>> modules;
  std::unordered_map<Symbol, std::vector<Method*>> builtins_by_name;
};

TORCH_API const std::vector<Method*>& getAllBuiltinFunctionsFor(Symbol name) {
  static BuiltinFunctionRegistry registry;
  return registry.getAllBuiltinFunctionsFor(name);
}

}}}
