#include <gtest/gtest.h>
#include <torch/csrc/jit/api/module.h>
#include <torch/csrc/jit/backends/backend.h>
#include <torch/csrc/jit/backends/backend_detail.h>
#include <torch/csrc/jit/frontend/resolver.h>
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
  auto res = lm.forward(inputs).toTuple()->elements();
  AT_ASSERT(res[0].toTensor().equal(ref[0].toTensor()));
  AT_ASSERT(res[1].toTensor().equal(ref[1].toTensor()));
}
} // namespace jit
} // namespace torch
