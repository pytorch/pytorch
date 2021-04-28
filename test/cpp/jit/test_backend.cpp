#include <gtest/gtest.h>
#include <test/cpp/jit/test_utils.h>
#include <torch/csrc/jit/api/module.h>
#include <torch/csrc/jit/backends/backend_detail.h>
#include <torch/csrc/jit/mobile/import.h>
#include <torch/torch.h>

// Tests go in torch::jit
namespace torch {
namespace jit {
TEST(BackendTest, ToBackend) {
  Module m("m");
  m.define(R"(
    def forward(self, x, h):
        return self.accum(x, h), self.sub_accum(x, h)

    def accum(self, x, h):
        return x + h

    def sub_accum(self, x, h):
        return x - h
  )");

  std::vector<IValue> inputs;
  inputs.emplace_back(2.0 * torch::ones({}));
  inputs.emplace_back(1.0 * torch::ones({}));
  auto ref = m.forward(inputs).toTuple()->elements();

  c10::Dict<IValue, IValue> compile_spec(StringType::get(), AnyType::get());
  c10::Dict<IValue, IValue> fake_dict(StringType::get(), AnyType::get());
  fake_dict.insert("", "");
  compile_spec.insert("forward", fake_dict);
  auto any_dict_ty = DictType::create(StringType::get(), AnyType::get());
  // lowered module
  auto lm = torch::jit::detail::codegen_backend_module(
      "test_backend", m, compile_spec, any_dict_ty);
  // lowered module code:
  /*
    class test_backendLoweredModule(Module):
      __parameters__ = []
      __buffers__ = []
      __processed_module : Any
      __method_compile_spec : Dict[str, Any]
      __backend : __torch__.torch.classes.__backends__.test_backend
      __handles : Dict[str, Any]
      def __create_backend(self: torch.jit.test_backendLoweredModule) -> None:
        _0 =
    __torch__.torch.classes.__backends__.test_backend.__new__(__torch__.torch.classes.__backends__.test_backend)
        _1 = (_0).__init__()
        self.__backend = _0
        return None
      def __getstate__(self: torch.jit.test_backendLoweredModule) ->
    Tuple[Dict[str, Any], Any]: _2 = (self.__method_compile_spec,
    self.__processed_module) return _2 def __setstate__(self:
    torch.jit.test_backendLoweredModule, state: Tuple[Dict[str, Any], Any]) ->
    None: self.__method_compile_spec = (state)[0] self.__processed_module =
    (state)[1] _3 = (self).__create_backend() _4 =
    (self.__backend).compile(self.__processed_module,
    self.__method_compile_spec, ) self.__handles = _4 return None def
    forward(self: torch.jit.test_backendLoweredModule, x: Tensor, h: Tensor) ->
    Tuple[Tensor, Tensor]: _5 = uninitialized(Tensor) typed_inputs =
    annotate(List[Any], [x, h]) _6 =
    (self.__backend).execute((self.__handles)["forward"], typed_inputs, ) _7,
    _8, = _6 _9 = isinstance(_7, Tensor) if _9: _10 = unchecked_cast(Tensor, _7)
        else:
          ops.prim.RaiseException("AssertionError: ")
          _10 = _5
        _11 = isinstance(_8, Tensor)
        if _11:
          _12 = unchecked_cast(Tensor, _8)
        else:
          ops.prim.RaiseException("AssertionError: ")
          _12 = _5
        return (_10, _12)

   */
  auto res = lm.forward(inputs).toTuple()->elements();
  AT_ASSERT(res[0].toTensor().equal(ref[0].toTensor()));
  AT_ASSERT(res[1].toTensor().equal(ref[1].toTensor()));
}

TEST(BackendTest, ToBackendNotAvailable) {
  Module m("m");
  m.define(R"(
    def forward(self, x, h):
        return self.accum(x, h), self.sub_accum(x, h)

    def accum(self, x, h):
        return x + h

    def sub_accum(self, x, h):
        return x - h
  )");

  std::vector<IValue> inputs;
  inputs.emplace_back(2.0 * torch::ones({}));
  inputs.emplace_back(1.0 * torch::ones({}));
  auto ref = m.forward(inputs).toTuple()->elements();

  c10::Dict<IValue, IValue> compile_spec(StringType::get(), AnyType::get());
  c10::Dict<IValue, IValue> fake_dict(StringType::get(), AnyType::get());
  fake_dict.insert("", "");
  compile_spec.insert("forward", fake_dict);
  auto any_dict_ty = DictType::create(StringType::get(), AnyType::get());
  // Produce lowered module (backend not available).
  // Exception is not thrown at this point.
  auto lm = torch::jit::detail::codegen_backend_module(
      "test_backend_unavailable", m, compile_spec, any_dict_ty);
  // Validate exception is thrown when trying to execute and
  // the backend is not available.
  ASSERT_THROWS_WITH_MESSAGE(
      lm.forward(inputs).toTuple()->elements(), "Backend is not available.");
}

TEST(BackendTest, TestCompiler) {
  Module m("m");
  m.define(R"(
    def forward(self, x, h):
        return x + h
  )");

  std::vector<IValue> inputs;
  inputs.emplace_back(2.0 * torch::ones({}));
  inputs.emplace_back(1.0 * torch::ones({}));
  auto ref = m.forward(inputs);

  c10::Dict<IValue, IValue> compile_spec(StringType::get(), AnyType::get());
  c10::Dict<IValue, IValue> fake_dict(StringType::get(), AnyType::get());
  fake_dict.insert("", "");
  compile_spec.insert("forward", fake_dict);
  auto any_dict_ty = DictType::create(StringType::get(), AnyType::get());
  // lowered module
  auto lm = torch::jit::detail::codegen_backend_module(
      "backend_with_compiler_demo", m, compile_spec, any_dict_ty);
  auto res = lm.forward(inputs);
  AT_ASSERT(res.toTensor().equal(ref.toTensor()));

  std::stringstream ss;
  lm._save_for_mobile(ss);
  auto mlm = _load_for_mobile(ss);
  auto mres = mlm.forward(inputs);
  AT_ASSERT(mres.toTensor().equal(ref.toTensor()));
}

TEST(BackendTest, TestCompilerNotSupport) {
  Module m("m");
  m.define(R"(
    def forward(self, x, h):
        return x * h
  )");

  c10::Dict<IValue, IValue> compile_spec(StringType::get(), AnyType::get());
  c10::Dict<IValue, IValue> fake_dict(StringType::get(), AnyType::get());
  fake_dict.insert("", "");
  compile_spec.insert("forward", fake_dict);
  auto any_dict_ty = DictType::create(StringType::get(), AnyType::get());
  // lowered module
  ASSERT_THROWS_WITH_MESSAGE(
      torch::jit::detail::codegen_backend_module(
          "backend_with_compiler_demo", m, compile_spec, any_dict_ty),
      "The node of aten::mul is not supported in this compiler. Source code:");
}
} // namespace jit
} // namespace torch
