#include <gtest/gtest.h>
#include <torch/csrc/jit/runtime/static/impl.h>
#include "deep_wide_pt.h"

TEST(StaticRuntime, TrivialModel) {
  torch::jit::Module mod = getTrivialScriptModel();
  auto a = torch::randn({2, 2});
  auto b = torch::randn({2, 2});
  auto c = torch::randn({2, 2});

  // run jit graph executor
  std::vector<at::IValue> input_ivalues({a, b, c});
  at::Tensor output_1 = mod.forward(input_ivalues).toTensor();

  // run static runtime
  std::vector<at::Tensor> input_tensors({a, b, c});
  torch::jit::StaticRuntime runtime(mod);
  at::Tensor output_2 = runtime.run(input_tensors)[0];
  EXPECT_TRUE(output_1.equal(output_2));
}

TEST(StaticRuntime, DeepWide) {
  const int embedding_size = 32;
  const int num_features = 50;
  torch::jit::Module mod = getDeepAndWideSciptModel();
  torch::jit::StaticRuntime runtime(mod);

  for (int batch_size : {1, 8, 32}) {
    for (int i = 0; i < 5; ++i) {
      auto ad_emb_packed = torch::randn({batch_size, 1, embedding_size});
      auto user_emb = torch::randn({batch_size, 1, embedding_size});
      auto wide = torch::randn({batch_size, num_features});

      // run jit graph executor
      std::vector<at::IValue> inputs({ad_emb_packed, user_emb, wide});
      at::Tensor output_1 = mod.forward(inputs).toTensor();

      // run static runtime
      std::vector<at::Tensor> input_tensors({ad_emb_packed, user_emb, wide});
      at::Tensor output_2 = runtime.run(input_tensors)[0];
      EXPECT_TRUE(output_1.equal(output_2));
    }
  }
}
