#include "deep_wide_pt.h"
#include <gtest/gtest.h>
#include <torch/csrc/jit/runtime/static/impl.h>

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
  auto g = torch::jit::PrepareForStaticRuntime(mod);
  torch::jit::StaticRuntime runtime(g);
  at::Tensor output_2 = runtime.run(input_tensors)[0];
  EXPECT_TRUE(output_1.equal(output_2));
}

static at::Tensor getTensor(const at::IValue &ival) {
  if (ival.isTensor()) {
    return ival.toTensor();
  } else if (ival.isTensorList()) {
    auto tensor_vec = ival.toTensorVector();
    TORCH_CHECK(tensor_vec.size() == 1);
    return tensor_vec[0];
  } else if (ival.isTuple()) {
    auto tuple = ival.toTuple();
    auto ivalue_vec = tuple->elements();
    TORCH_CHECK(ivalue_vec.size() == 1);
    return ivalue_vec[0].toTensor();
  } else {
    CAFFE_THROW("Unknown input IValue");
  }
}

TEST(StaticRuntime, DeepWide) {
  const int embedding_size = 32;
  const int num_features = 50;
  torch::jit::Module mod = getDeepAndWideSciptModel();
  auto g = torch::jit::PrepareForStaticRuntime(mod);
  torch::jit::StaticRuntime runtime(g);

  for (int batch_size : {1, 8, 32}) {
    for (int i = 0; i < 2; ++i) {
      auto ad_emb_packed = torch::randn({batch_size, 1, embedding_size});
      auto user_emb = torch::randn({batch_size, 1, embedding_size});
      auto wide = torch::randn({batch_size, num_features});

      // run jit graph executor
      std::vector<at::IValue> inputs({ad_emb_packed, user_emb, wide});
      auto output_1 = getTensor(mod.forward(inputs));

      // run static runtime
      std::vector<at::Tensor> input_tensors({ad_emb_packed, user_emb, wide});
      at::Tensor output_2 = runtime.run(input_tensors)[0];
      EXPECT_TRUE(output_1.equal(output_2));
    }
  }
}

TEST(StaticRuntime, KWargsAPI_1) {
  const int embedding_size = 32;
  const int num_features = 50;
  auto module = getDeepAndWideSciptModel();
  torch::jit::StaticRuntime runtime(module);

  for (int batch_size : {1, 8, 32}) {
    for (int i = 0; i < 2; ++i) {
      auto ad_emb_packed = torch::randn({batch_size, 1, embedding_size});
      auto user_emb = torch::randn({batch_size, 1, embedding_size});
      auto wide = torch::randn({batch_size, num_features});

      // run jit graph executor
      std::vector<at::IValue> inputs({ad_emb_packed, user_emb, wide});
      at::Tensor output_1 = getTensor(module.forward(inputs));

      // run static runtime
      at::Tensor output_2 = getTensor(runtime.run(inputs, {}));
      EXPECT_TRUE(output_1.equal(output_2));
    }
  }
}

TEST(StaticRuntime, KWargsAPI_2) {
  const int embedding_size = 32;
  const int num_features = 50;
  auto module = getDeepAndWideSciptModel();
  auto g = torch::jit::PrepareForStaticRuntime(module);
  torch::jit::StaticRuntime runtime(module);

  for (int batch_size : {1, 8, 32}) {
    for (int i = 0; i < 2; ++i) {
      auto ad_emb_packed = torch::randn({batch_size, 1, embedding_size});
      auto user_emb = torch::randn({batch_size, 1, embedding_size});
      auto wide = torch::randn({batch_size, num_features});

      // run jit graph executor
      std::vector<at::IValue> args({ad_emb_packed, user_emb, wide});
      at::Tensor output_1 = getTensor(module.forward(args));

      std::unordered_map<std::string, c10::IValue> kwargs(
          {{"ad_emb_packed", ad_emb_packed},
           {"user_emb", user_emb},
           {"wide", wide}});

      // run static runtime
      at::Tensor output_2 = getTensor(runtime.run({}, kwargs));
      EXPECT_TRUE(output_1.equal(output_2));
    }
  }
}

TEST(StaticRuntime, CleanUpMemory) {
  const int embedding_size = 32;
  const int num_features = 50;
  torch::jit::Module mod = getDeepAndWideSciptModel();
  auto g = torch::jit::PrepareForStaticRuntime(mod);

  for (auto cleanup_memory : {true, false}) {
    for (auto enable_out_variant : {true, false}) {
      VLOG(1) << "cleanup_memory: " << cleanup_memory
              << ", enable_out_variant: " << enable_out_variant;
      torch::jit::StaticRuntimeOptions opts{cleanup_memory, enable_out_variant};
      torch::jit::StaticRuntime runtime(g, opts);

      for (int batch_size : {1, 8, 32}) {
        for (int i = 0; i < 2; ++i) {
          auto ad_emb_packed = torch::randn({batch_size, 1, embedding_size});
          auto user_emb = torch::randn({batch_size, 1, embedding_size});
          auto wide = torch::randn({batch_size, num_features});

          // run jit graph executor
          std::vector<at::IValue> inputs({ad_emb_packed, user_emb, wide});
          auto output_1 = getTensor(mod.forward(inputs));

          // run static runtime
          std::vector<at::Tensor> input_tensors(
              {ad_emb_packed, user_emb, wide});
          at::Tensor output_2 = runtime.run(input_tensors)[0];
          EXPECT_TRUE(output_1.equal(output_2));
        }
      }
    }
  }
}
