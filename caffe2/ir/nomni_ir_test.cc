#include "caffe2/ir/nomni_ir.h"

#include <gtest/gtest.h>

TEST(Types, VerifyTypes) {
  torch::jit::Node node(torch::jit::Symbol::fromQualString("test::test"));
  torch::jit::Value value;

  nom::DFGraph dd;
  auto operatorNode = dd.createNode(std::move(node));
  auto valueNode = dd.createNode(torch::jit::Value());

  ASSERT_NO_THROW({
    auto& testOp = operatorNode->data().getOperator();
    LOG(INFO) << (uint32_t)testOp.kind();
  });
  ASSERT_NO_THROW({
    auto& testVal = valueNode->data().getValue();
    LOG(INFO) << (uint32_t)testVal.isTensor();
  });
}

TEST(Types, Errors) {
  torch::jit::Node node(torch::jit::Symbol::fromQualString("test::test"));
  torch::jit::Value value;

  nom::DFGraph dd;
  auto operatorNode = dd.createNode(std::move(node));
  auto valueNode = dd.createNode(torch::jit::Value());

  ASSERT_ANY_THROW(auto& testOp = operatorNode->data().getValue());
  ASSERT_ANY_THROW(auto& testVal = valueNode->data().getOperator());
}
