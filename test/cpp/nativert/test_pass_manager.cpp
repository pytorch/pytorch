#include <gtest/gtest.h>

#include <torch/nativert/graph/Graph.h>
#include <torch/nativert/graph/passes/pass_manager/PassManager.h>

#include <torch/csrc/jit/testing/file_check.h>

using namespace ::testing;
using namespace torch::nativert;

TEST(PassManagerTest, TestEmptyPass) {
  GraphPassManager manager({"EmptyPass"});
  EXPECT_FALSE(manager.run(Graph::createGraph().get()));
}

TEST(PassPipelineTest, TestConcat) {
  GraphPassPipeline p1({"test"});
  EXPECT_EQ(p1.size(), 1);
  EXPECT_EQ(p1.at(0), "test");
  p1.concat({"test1", "test2"});
  EXPECT_EQ(p1.at(0), "test");
  EXPECT_EQ(p1.at(1), "test1");
  EXPECT_EQ(p1.at(2), "test2");
}

TEST(PassPipelineTest, TestPushFront) {
  GraphPassPipeline p1({"test"});
  EXPECT_EQ(p1.size(), 1);
  EXPECT_EQ(p1.at(0), "test");
  p1.push_front("test1");
  EXPECT_EQ(p1.at(0), "test1");
  EXPECT_EQ(p1.at(1), "test");
}
