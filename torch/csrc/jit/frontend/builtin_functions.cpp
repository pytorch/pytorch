#include <torch/csrc/api/include/torch/jit.h>
#include <torch/csrc/jit/frontend/code_template.h>
#include <torch/csrc/jit/frontend/builtin_functions.h>
#include <torch/csrc/jit/frontend/resolver.h>

namespace torch {
namespace jit {

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

auto _ntuple_ops = CodeTemplate(
    R"SCRIPT(
def _${name}(x: BroadcastingList${Length}[${Scalar}]) -> List[${Scalar}]:
  return x
)SCRIPT");

auto floordiv = CodeTemplate(
    R"SCRIPT(
def floordiv(self : Tensor, other : ${Rhs_Type}) -> Tensor:
  return torch.floor_divide(self, other)
)SCRIPT");

auto tensor_properties =
    R"SCRIPT(
def ndim(a : Tensor) -> int:
  return a.dim()
def T(a : Tensor) -> Tensor:
  return a.numpy_T()
)SCRIPT";

struct BuiltinFunctionRegistry {
  const std::vector<Function*>& getAllBuiltinFunctionsFor(
      Symbol name) {
    const static std::vector<Function*> empty;
    // when initializing the builtin function library, we will re-enter
    // getAllBuiltinFunctionsFor since it is called in the compiler to
    // lookup builtins and initializing the builtin functions calls the
    // compiler. To avoid deadlocking, we use a recursive mutex (same thread can
    // re-lock, the mutex without waiting), and report no loaded builtins during
    // init.
    std::lock_guard<std::recursive_mutex> guard(mutex);
    if (state == INTIIALIZING) {
      return empty;
    } else if (state == UNINITIALIZED) {
      state = INTIIALIZING;
      loadBuiltinFunctions();
      state = INITIALIZED;
    }
    AT_ASSERT(state == INITIALIZED);
    auto it = builtins_by_name_.find(name);
    if (it == builtins_by_name_.end())
      return empty;
    return it->second;
  }

 private:
  void loadSource(const std::string& source, const std::string& the_namespace) {
    std::shared_ptr<CompilationUnit> cu = std::make_shared<CompilationUnit>();
    modules.emplace_back(cu);
    cu->define(
        c10::nullopt, source, nativeResolver(), /*self=*/nullptr);
    for (auto& method : cu->get_functions()) {
      builtins_by_name_[Symbol::fromQualString(
                            the_namespace + "::" + method->name())]
          .push_back(method);
    }
  }
  void loadBuiltinFunctions() {
    for (auto scalar : {"float", "int"}) {
      TemplateEnv env;
      env.s("Scalar", scalar);
      loadSource(scalar_operators_source.format(env), "aten");
    }

    using str_pair = std::pair<std::string, std::string>;
    const std::vector<str_pair> name_len = {
        str_pair("single", "1"),
        str_pair("pair", "2"),
        str_pair("triple", "3"),
        str_pair("quadruple", "4"),
    };
    for (const auto scalar : {"float", "int"}) {
      for (const auto& pair : name_len) {
        TemplateEnv env;
        env.s("Scalar", scalar);
        env.s("name", pair.first);
        env.s("Length", pair.second);
        loadSource(_ntuple_ops.format(env), "aten");
      }
    }
    for (auto rhs : {"number", "Tensor"}) {
      TemplateEnv env;
      env.s("Rhs_Type", rhs);
      loadSource(floordiv.format(env), "aten");
    }

    // These are under `prim` instead of `aten` since they exist to bind certain
    // tensor property getters to correpsonding methods
    loadSource(tensor_properties, "prim");
  }
  enum { UNINITIALIZED, INTIIALIZING, INITIALIZED } state = UNINITIALIZED;
  std::recursive_mutex mutex;
  std::vector<std::shared_ptr<CompilationUnit>> modules;
  std::unordered_map<Symbol, std::vector<Function*>>
      builtins_by_name_;
};

const std::vector<Function*>& getAllBuiltinFunctionsFor(
    Symbol name) {
  static BuiltinFunctionRegistry registry;
  return registry.getAllBuiltinFunctionsFor(name);
}

} // namespace jit
} // namespace torch
