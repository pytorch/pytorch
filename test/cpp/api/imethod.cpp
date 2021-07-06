// (c) Facebook, Inc. and its affiliates. Confidential and proprietary.

#include <gtest/gtest.h>
#include <torch/csrc/deploy/deploy.h>
#include <torch/script.h>
#include <torch/torch.h>

using namespace ::testing;
using namespace caffe2;

TEST(IMethodTest, CallMethod) {
  auto script_model = torch::jit::load(getenv("SIMPLE_JIT"));
  auto script_method = script_model.get_method("forward");

  torch::deploy::InterpreterManager manager(3);
  torch::deploy::Package p = manager.load_package(getenv("SIMPLE"));
  auto py_model = p.load_pickle("model", "model.pkl");
  torch::deploy::PythonMethodWrapper py_method(py_model, "forward");

  auto input = torch::ones({10, 20});
  auto output_py = py_method({input});
  auto output_script = script_method({input});
  EXPECT_TRUE(output_py.isTensor());
  EXPECT_TRUE(output_script.isTensor());
  auto output_py_tensor = output_py.toTensor();
  auto output_script_tensor = output_script.toTensor();

  EXPECT_TRUE(output_py_tensor.equal(output_script_tensor));
  EXPECT_EQ(output_py_tensor.numel(), 200);
}

TEST(IMethodTest, GetArgumentNames) {
  auto script_model = torch::jit::load(getenv("SIMPLE_JIT"));
  auto script_method = script_model.get_method("forward");

  torch::deploy::InterpreterManager manager(3);
  torch::deploy::Package p = manager.load_package(getenv("SIMPLE"));
  auto py_model = p.load_pickle("model", "model.pkl");
  torch::deploy::PythonMethodWrapper py_method(py_model, "forward");

  // TODO(whc) implement and test these
  EXPECT_THROW(script_method.getArgumentNames(), std::runtime_error);
  EXPECT_THROW(py_method.getArgumentNames(), std::runtime_error);
}
