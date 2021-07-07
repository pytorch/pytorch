#include <gtest/gtest.h>
#include <torch/csrc/deploy/deploy.h>
#include <torch/cuda.h>
#include <torch/script.h>
#include <torch/torch.h>
#include <future>
#include <iostream>
#include <string>

int main(int argc, char* argv[]) {
  ::testing::InitGoogleTest(&argc, argv);
  int rc = RUN_ALL_TESTS();
  return rc;
}

// NOLINTNEXTLINE(cppcoreguidelines-avoid-non-const-global-variables)
const char* simple = "torch/csrc/deploy/example/generated/simple";
// NOLINTNEXTLINE(cppcoreguidelines-avoid-non-const-global-variables)
const char* simple_jit = "torch/csrc/deploy/example/generated/simple_jit";

const char* path(const char* envname, const char* path) {
  const char* e = getenv(envname);
  return e ? e : path;
}

// NOLINTNEXTLINE(cppcoreguidelines-avoid-non-const-global-variables)
TEST(TorchDeployGPUTest, SimpleModel) {
  if (!torch::cuda::is_available()) {
    GTEST_SKIP();
  }
  const char* model_filename = path("SIMPLE", simple);
  const char* jit_filename = path("SIMPLE_JIT", simple_jit);

  // Test
  torch::deploy::InterpreterManager m(1);
  torch::deploy::Package p = m.load_package(model_filename);
  auto model = p.load_pickle("model", "model.pkl");
  {
    auto M = model.acquire_session();
    M.self.attr("to")({"cuda"});
  }
  std::vector<at::IValue> inputs;
  {
    auto I = p.acquire_session();
    auto eg = I.self.attr("load_pickle")({"model", "example.pkl"}).toIValue();
    inputs = eg.toTuple()->elements();
    inputs[0] = inputs[0].toTensor().to("cuda");
  }
  at::Tensor output = model(inputs).toTensor();
  ASSERT_TRUE(output.device().is_cuda());

  // Reference
  auto ref_model = torch::jit::load(jit_filename);
  ref_model.to(torch::kCUDA);
  at::Tensor ref_output = ref_model.forward(inputs).toTensor();

  ASSERT_TRUE(ref_output.allclose(output, 1e-03, 1e-05));
}
