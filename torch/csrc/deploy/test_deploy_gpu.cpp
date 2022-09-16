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

const char* simple = "torch/csrc/deploy/example/generated/simple";
const char* simple_jit = "torch/csrc/deploy/example/generated/simple_jit";

const char* path(const char* envname, const char* path) {
  const char* e = getenv(envname);
  return e ? e : path;
}

TEST(TorchDeployGPUTest, SimpleModel) {
  if (!torch::cuda::is_available()) {
    GTEST_SKIP();
  }
  const char* model_filename = path("SIMPLE", simple);
  const char* jit_filename = path("SIMPLE_JIT", simple_jit);

  // Test
  torch::deploy::InterpreterManager m(1);
  torch::deploy::Package p = m.loadPackage(model_filename);
  auto model = p.loadPickle("model", "model.pkl");
  {
    auto M = model.acquireSession();
    M.self.attr("to")({"cuda"});
  }
  // NOLINTNEXTLINE(cppcoreguidelines-init-variables)
  std::vector<at::IValue> inputs;
  {
    auto I = p.acquireSession();
    auto eg = I.self.attr("load_pickle")({"model", "example.pkl"}).toIValue();
    inputs = eg.toTupleRef().elements();
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

TEST(TorchDeployGPUTest, UsesDistributed) {
  const auto model_filename = path(
      "USES_DISTRIBUTED",
      "torch/csrc/deploy/example/generated/uses_distributed");
  torch::deploy::InterpreterManager m(1);
  torch::deploy::Package p = m.loadPackage(model_filename);
  {
    auto I = p.acquireSession();
    I.self.attr("import_module")({"uses_distributed"});
  }
}

#ifdef FBCODE_CAFFE2
TEST(TorchDeployGPUTest, TensorRT) {
  if (!torch::cuda::is_available()) {
    GTEST_SKIP();
  }
  auto packagePath = path(
      "MAKE_TRT_MODULE", "torch/csrc/deploy/example/generated/make_trt_module");
  torch::deploy::InterpreterManager m(1);
  torch::deploy::Package p = m.loadPackage(packagePath);
  auto makeModel = p.loadPickle("make_trt_module", "model.pkl");
  {
    auto I = makeModel.acquireSession();
    auto model = I.self(at::ArrayRef<at::IValue>{});
    auto input = at::ones({1, 2, 3}).cuda();
    auto output = input * 2;
    ASSERT_TRUE(
        output.allclose(model(at::IValue{input}).toIValue().toTensor()));
  }
}
#endif

// OSS build does not have bultin numpy support yet. Use this flag to guard the
// test case.
#if HAS_NUMPY
TEST(TorchpyTest, TestNumpy) {
  torch::deploy::InterpreterManager m(2);
  auto noArgs = at::ArrayRef<torch::deploy::Obj>();
  auto I = m.acquireOne();
  auto mat35 = I.global("numpy", "random").attr("rand")({3, 5});
  auto mat58 = I.global("numpy", "random").attr("rand")({5, 8});
  auto mat38 = I.global("numpy", "matmul")({mat35, mat58});
  EXPECT_EQ(2, mat38.attr("shape").attr("__len__")(noArgs).toIValue().toInt());
  EXPECT_EQ(3, mat38.attr("shape").attr("__getitem__")({0}).toIValue().toInt());
  EXPECT_EQ(8, mat38.attr("shape").attr("__getitem__")({1}).toIValue().toInt());
}
#endif

#if HAS_PYYAML
TEST(TorchpyTest, TestPyYAML) {
  const std::string kDocument = "a: 1\n";

  torch::deploy::InterpreterManager m(2);
  auto I = m.acquireOne();

  auto load = I.global("yaml", "load")({kDocument});
  EXPECT_EQ(1, load.attr("__getitem__")({"a"}).toIValue().toInt());

  auto dump = I.global("yaml", "dump")({load});
  EXPECT_EQ(kDocument, dump.toIValue().toString()->string());
}
#endif
