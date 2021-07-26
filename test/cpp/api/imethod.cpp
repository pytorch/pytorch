// (c) Facebook, Inc. and its affiliates. Confidential and proprietary.

#include <gtest/gtest.h>
#include <torch/csrc/deploy/deploy.h>
#include <torch/script.h>
#include <torch/torch.h>

using namespace ::testing;
using namespace caffe2;

// TODO(T96218435): Enable the following tests in OSS.
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
  auto scriptModel = torch::jit::load(getenv("SIMPLE_JIT"));
  auto scriptMethod = scriptModel.get_method("forward");

  auto& scriptNames = scriptMethod.getArgumentNames();
  EXPECT_EQ(scriptNames.size(), 2);
  EXPECT_STREQ(scriptNames[0].c_str(), "self");
  EXPECT_STREQ(scriptNames[1].c_str(), "input");

  torch::deploy::InterpreterManager manager(3);
  torch::deploy::Package package = manager.load_package(getenv("SIMPLE"));
  auto pyModel = package.load_pickle("model", "model.pkl");
  torch::deploy::PythonMethodWrapper pyMethod(pyModel, "forward");

  auto& pyNames = pyMethod.getArgumentNames();
  EXPECT_EQ(pyNames.size(), 2);
  EXPECT_STREQ(pyNames[0].c_str(), "input");
  EXPECT_STREQ(pyNames[1].c_str(), "kwargs");
}
