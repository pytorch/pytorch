#pragma once

#include "test/cpp/jit/test_base.h"
#include "test/cpp/jit/test_utils.h"

#include "torch/csrc/autograd/generated/variable_factories.h"
#include "torch/jit.h"
#include "torch/csrc/jit/script/module.h"
#include "torch/script.h"

#include <ATen/ATen.h>

namespace torch {
namespace jit {
namespace test {

void testLiteExecutor() {
  auto m = std::make_shared<script::Module>();
  m->register_parameter("foo", torch::ones({}), false);
  m->define(R"(
    def add_it(self, x, b : int = 4):
      return self.foo + x + b
  )");
  std::vector<IValue> inputs;
  inputs.emplace_back(torch::ones({}));
  std::stringstream ss;
  m->save_method("add_it", inputs, ss);

  // TODO:
  // 1. Load ss to a InstructionList
  // 2. Execute InstructionList
  // 3. Compare the result to n->run_method("add_it", torch::ones({})).toTensor()
  AT_ASSERT(!ss.str().empty());
}

} // namespace test
} // namespace jit
} // namespace torch
