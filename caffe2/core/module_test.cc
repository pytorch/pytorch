#include <iostream>
#include <memory>

#include "caffe2/core/module.h"
#include "caffe2/core/operator.h"
#include <gtest/gtest.h>
#include "caffe2/core/logging.h"

// An explicitly defined module, testing correctness when we statically link a
// module
CAFFE2_MODULE(caffe2_module_test_static, "Static module for testing.");

namespace caffe2 {

class Caffe2ModuleTestStaticDummyOp : public OperatorBase {
 public:
  using OperatorBase::OperatorBase;
  bool Run(int /* unused */ /*stream_id*/) override {
    return true;
  }
  virtual string type() {
    return "base";
  }
};

REGISTER_CPU_OPERATOR(
  Caffe2ModuleTestStaticDummy, Caffe2ModuleTestStaticDummyOp);
OPERATOR_SCHEMA(Caffe2ModuleTestStaticDummy);

TEST(ModuleTest, StaticModule) {
  const string name = "caffe2_module_test_static";
  const auto& modules = CurrentModules();
  EXPECT_EQ(modules.count(name), 1);
  EXPECT_TRUE(HasModule(name));

  // LoadModule should not raise an error, since the module is already present.
  LoadModule(name);
  // Even a non-existing path should not cause error.
  LoadModule(name, "/does/not/exist.so");
  EXPECT_EQ(modules.count(name), 1);
  EXPECT_TRUE(HasModule(name));

  // The module will then introduce the Caffe2ModuleTestStaticDummyOp.
  OperatorDef op_def;
  Workspace ws;
  op_def.set_type("Caffe2ModuleTestStaticDummy");
  unique_ptr<OperatorBase> op = CreateOperator(op_def, &ws);
  EXPECT_NE(nullptr, op.get());
}

#ifdef CAFFE2_BUILD_SHARED_LIBS
TEST(ModuleTest, DynamicModule) {
  const string name = "caffe2_module_test_dynamic";
  const auto& modules = CurrentModules();
  EXPECT_EQ(modules.count(name), 0);
  EXPECT_FALSE(HasModule(name));

  // Before loading, we should not be able to create the op.
  OperatorDef op_def;
  Workspace ws;
  op_def.set_type("Caffe2ModuleTestDynamicDummy");
  EXPECT_THROW(
      CreateOperator(op_def, &ws),
      EnforceNotMet);

  // LoadModule should load the proper module.
  LoadModule(name);
  EXPECT_EQ(modules.count(name), 1);
  EXPECT_TRUE(HasModule(name));

  // The module will then introduce the Caffe2ModuleTestDynamicDummyOp.
  unique_ptr<OperatorBase> op_after_load = CreateOperator(op_def, &ws);
  EXPECT_NE(nullptr, op_after_load.get());
}
#endif

}  // namespace caffe2
