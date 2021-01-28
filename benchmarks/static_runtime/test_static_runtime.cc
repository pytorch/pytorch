#include <gtest/gtest.h>
#include <torch/csrc/jit/runtime/static/fusion.h>
#include <torch/csrc/jit/runtime/static/impl.h>
#include "deep_wide_pt.h"
#include "test_scripts.h"

using namespace caffe2;
using namespace torch;
using namespace torch::jit;
using c10::IValue;

namespace {
static at::Tensor getTensor(const at::IValue& ival) {
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

void compareTensorLists(
    const std::vector<IValue>& l, /* values */
    const std::vector<IValue>& r /* expects */) {
  EXPECT_TRUE(l.size() == r.size());
  for (int i = 0; i < l.size(); ++i) {
    ASSERT_TRUE(l[i].isTensor());
    ASSERT_TRUE(r[i].isTensor());
    LOG(INFO) << "output " << i << ": \n" << l[i] << std::endl;
    LOG(INFO) << "expect " << i << ": \n" << r[i] << std::endl;
    EXPECT_TRUE(l[i].toTensor().equal(r[i].toTensor()));
  }
}

void compareTensorLists(
    const std::vector<at::Tensor>& l, /* values */
    const std::vector<at::Tensor>& r /* expects */) {
  EXPECT_TRUE(l.size() == r.size());
  for (int i = 0; i < l.size(); ++i) {
    LOG(INFO) << "output " << i << ": \n" << l[i] << std::endl;
    LOG(INFO) << "expect " << i << ": \n" << r[i] << std::endl;
    EXPECT_TRUE(l[i].equal(r[i]));
  }
}

// Given a model/function in jit script, run the model/function
// with the jit interpreter and static runtime, and compare the results
void testStaticRuntime(
    const std::string& jit_script,
    const std::vector<IValue>& args) {
  script::Module module("module");
  module.define(jit_script);

  auto expect = module.forward(args);

  StaticRuntime runtime(module);
  auto actual = runtime.run(args, {});

  if (expect.isTuple()) {
    compareTensorLists(
        expect.toTuple()->elements(), actual.toTuple()->elements());
  } else if (expect.isList()) {
    compareTensorLists(
        expect.toTensorVector(), actual.toTensorVector());
  } else {
    EXPECT_TRUE(expect.toTensor().equal(actual.toTensor()));
  }
}
} // namespace

TEST(StaticRuntime, IndividualOps_Binary) {
  auto a = at::randn({2, 3});
  auto b = at::ones({2, 3});

  std::vector<IValue> args{a, b};

  testStaticRuntime(add_script, args);
  testStaticRuntime(list_construct_script, args);
  testStaticRuntime(list_unpack_script, args);
  testStaticRuntime(tuple_construct_script, args);
}

TEST(StaticRuntime, LongModel) {
  torch::jit::Module mod = getLongScriptModel();
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

TEST(StaticRuntime, LeakyReLU) {
  torch::jit::Module mod = getLeakyReLUConstScriptModel();
  auto inputs = torch::randn({2, 2});

  // run jit graph executor
  std::vector<at::IValue> input_ivalues({inputs});
  at::Tensor output_1 = mod.forward(input_ivalues).toTensor();

  // run static runtime
  std::vector<at::Tensor> input_tensors({inputs});
  auto g = torch::jit::PrepareForStaticRuntime(mod);
  torch::jit::StaticRuntime runtime(g);
  at::Tensor output_2 = runtime.run(input_tensors)[0];
  EXPECT_TRUE(output_1.equal(output_2));
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

TEST(StaticRuntime, FusionPass) {
  const int embedding_size = 32;
  const int num_features = 50;
  for (int batch_size : {1, 8, 32}) {
    for (int i = 0; i < 2; ++i) {
      torch::jit::Module module = getDeepAndWideSciptModel();
      auto ad_emb_packed = torch::randn({batch_size, 1, embedding_size});
      auto user_emb = torch::randn({batch_size, 1, embedding_size});
      auto wide = torch::randn({batch_size, num_features});

      // run jit graph executor
      std::vector<at::IValue> inputs({ad_emb_packed, user_emb, wide});
      auto output_1 = getTensor(module.forward(inputs));

      Method method = module.get_method("forward");
      auto graph = method.graph();
      fuseStaticSubgraphs(graph);
      bool hit = false;
      for (const auto& n : module.get_method("forward").graph()->nodes()) {
        if (n->kind() == torch::jit::prim::StaticSubgraph) {
        hit = true;
        }
      }
      EXPECT_TRUE(hit);
      auto output_2 = getTensor(module.forward(inputs));
      EXPECT_TRUE(output_1.equal(output_2));
    }
  }
}

