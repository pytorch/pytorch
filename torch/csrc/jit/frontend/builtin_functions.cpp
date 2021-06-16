#include <torch/csrc/jit/frontend/builtin_functions.h>

#include <torch/csrc/api/include/torch/jit.h>
#include <torch/csrc/jit/frontend/code_template.h>
#include <torch/csrc/jit/frontend/resolver.h>

namespace torch {
namespace jit {

// NOLINTNEXTLINE(cppcoreguidelines-avoid-non-const-global-variables)
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

// NOLINTNEXTLINE(cppcoreguidelines-avoid-non-const-global-variables)
auto _ntuple_ops = CodeTemplate(
    R"SCRIPT(
def _${name}(x: BroadcastingList${Length}[${Scalar}]) -> List[${Scalar}]:
  return x
)SCRIPT");

// NOLINTNEXTLINE(cppcoreguidelines-avoid-non-const-global-variables)
auto floordiv = CodeTemplate(
    R"SCRIPT(
def floordiv(self : Tensor, other : ${Rhs_Type}) -> Tensor:
  return torch.floor_divide(self, other)
)SCRIPT");

// NOLINTNEXTLINE(cppcoreguidelines-avoid-non-const-global-variables)
auto tensor_properties =
    R"SCRIPT(
def ndim(a : Tensor) -> int:
  return a.dim()
def T(a : Tensor) -> Tensor:
  return a.numpy_T()
def shape(a : Tensor) -> List[int]:
  return a.size()
)SCRIPT";

// _assert_int_or_pair is only here for backwards-compatibility with the
// aten::_assert_int_or_pair op which was removed once we were able to compile
// torch.nn.functional.assert_int_or_pair
// list_with_default also needs to be here for BC
// NOLINTNEXTLINE(cppcoreguidelines-avoid-non-const-global-variables)
auto aten_ops =
    R"SCRIPT(
def _assert_int_or_pair(vals: List[int], name: str, message: str):
  pass
def list_with_default(out_size: List[int], defaults: List[int]):
  assert len(defaults) > len(out_size)
  return out_size
def _assert(condition : bool, message : str):
  assert condition, message
def type(self: Tensor, dtype: int, non_blocking: bool=False, copy: bool=False) -> Tensor:
  return self.to(dtype, non_blocking, copy)
)SCRIPT";

// an additional overload for Tensor variant of _assert
const auto aten_ops_additional =
    R"SCRIPT(
def _assert(condition : Tensor, message : str):
  assert bool(condition), message
def __contains__(self: str, key: str):
    return self.find(key, 0, len(self)) != -1
)SCRIPT";

// Implementations of historic symbol behaviors are defined here
// See note [Versioned Symbols]

// This builtin is for testing
// NOLINTNEXTLINE(cppcoreguidelines-avoid-non-const-global-variables)
auto _test_serialization_subcmul = R"SCRIPT(
def _test_serialization_subcmul_0_2(self: Tensor, other:Tensor, alpha: number=2) -> Tensor:
  return other - (self * alpha)
)SCRIPT";

// Division versioned symbols, for Torchscript programs serialized when
// division on integer tensors was floor division, not true division.

// Tensor x Tensor
// NOTE: testing for the tensors being float tensors is sufficient here,
// because the Torchscript versions this fix applies to (0 through 3)
// did not support complex tensors.
// NOLINTNEXTLINE(cppcoreguidelines-avoid-non-const-global-variables)
auto div_tensor = R"SCRIPT(
def div_0_3(self: Tensor, other: Tensor) -> Tensor:
  if (self.is_floating_point() or other.is_floating_point()):
    return self.true_divide(other)
  return self.divide(other, rounding_mode='trunc')
)SCRIPT";

// Tensor x Scalar
// NOLINTNEXTLINE(cppcoreguidelines-avoid-non-const-global-variables)
auto div_tensor_scalar = R"SCRIPT(
def div_0_3(self: Tensor, other: number) -> Tensor:
  if (self.is_floating_point() or isinstance(other, float)):
    return self.true_divide(other)
  return self.divide(other, rounding_mode='trunc')
)SCRIPT";

// Scalar x Scalar
// NOLINTNEXTLINE(cppcoreguidelines-avoid-non-const-global-variables)
auto div_scalar_scalar = R"SCRIPT(
def div_0_3(self: number, other: number) -> number:
  return self / other
)SCRIPT";

// Tensor x Tensor with out kwarg
// NOTE: the JIT doesn't support Tensor x Scalar with the out kwarg
// NOLINTNEXTLINE(cppcoreguidelines-avoid-non-const-global-variables)
auto div_tensor_out = R"SCRIPT(
def div_0_3(self: Tensor, other: Tensor, *, out: Tensor) -> Tensor:
  if (self.is_floating_point() or other.is_floating_point() or out.is_floating_point()):
    return self.true_divide(other, out=out)
  return self.divide(other, rounding_mode='trunc', out=out)
)SCRIPT";

// Tensor x Tensor inplace
// NOLINTNEXTLINE(cppcoreguidelines-avoid-non-const-global-variables)
auto div__tensor = R"SCRIPT(
def div__0_3(self: Tensor, other: Tensor) -> Tensor:
  if (self.is_floating_point() or other.is_floating_point()):
    return self.true_divide_(other)
  return self.divide_(other, rounding_mode='trunc')
)SCRIPT";

// Tensor x Scalar inplace
// NOLINTNEXTLINE(cppcoreguidelines-avoid-non-const-global-variables)
auto div__scalar = R"SCRIPT(
def div__0_3(self: Tensor, other: number) -> Tensor:
  if (self.is_floating_point() or isinstance(other, float)):
    return self.true_divide_(other)
  return self.divide_(other, rounding_mode='trunc')
)SCRIPT";

// NOTE: torch.full would historically infer a float dtype for bool and
//   integral fill values.
// NOTE: Torchscript does not currently support complex values
// NOTE: Torchscript does not currently support named tensors, although
//   torch.full does have a named tensor variant
// NOLINTNEXTLINE(cppcoreguidelines-avoid-non-const-global-variables)
auto full = R"SCRIPT(
def full_0_4(size:List[int], fill_value:number, *, dtype:Optional[int]=None,
             layout:Optional[int]=None, device:Optional[Device]=None,
             pin_memory:Optional[bool]=None) -> Tensor:
  if dtype is None:
    fill_value = float(fill_value)

  return torch.full(size, fill_value, dtype=dtype, layout=layout, device=device, pin_memory=pin_memory)
)SCRIPT";

// NOTE: the out variant of full works the same, but must be overridden
//   since the other variant of full is overridden
// NOLINTNEXTLINE(cppcoreguidelines-avoid-non-const-global-variables)
auto full_out = R"SCRIPT(
def full_0_4(size:List[int], fill_value:number, *, out:Tensor) -> Tensor:
  return torch.full(size, fill_value, out=out)
)SCRIPT";

struct BuiltinFunctionRegistry {
  const std::vector<Function*>& getAllBuiltinFunctionsFor(Symbol name) {
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
    cu->define(c10::nullopt, source, nativeResolver(), /*self=*/nullptr);
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

    loadSource(aten_ops, "aten");
    loadSource(aten_ops_additional, "aten");

    // Loads functions implementing historic behavior, see note [Versioned
    // Symbols]
    // Note: these functions go into the "upgraders" namespace
    loadSource(_test_serialization_subcmul, "upgraders");
    loadSource(div_tensor, "upgraders");
    loadSource(div_tensor_scalar, "upgraders");
    loadSource(div_scalar_scalar, "upgraders");
    loadSource(div__tensor, "upgraders");
    loadSource(div_tensor_out, "upgraders");
    loadSource(div__scalar, "upgraders");
    loadSource(full, "upgraders");
    loadSource(full_out, "upgraders");

    // These are under `prim` instead of `aten` since they exist to bind certain
    // tensor property getters to correpsonding methods
    loadSource(tensor_properties, "prim");
  }
  enum { UNINITIALIZED, INTIIALIZING, INITIALIZED } state = UNINITIALIZED;
  std::recursive_mutex mutex;
  std::vector<std::shared_ptr<CompilationUnit>> modules;
  std::unordered_map<Symbol, std::vector<Function*>> builtins_by_name_;
};

const std::vector<Function*>& getAllBuiltinFunctionsFor(Symbol name) {
  static BuiltinFunctionRegistry registry;
  return registry.getAllBuiltinFunctionsFor(name);
}

} // namespace jit
} // namespace torch
