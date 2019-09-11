// Copyright 2004-present Facebook. All Rights Reserved.

#include <gtest/gtest.h>

#include <torch/torch.h>

#include <test/cpp/api/support.h>

#include<torch/nn/utils/clip_grad.h>

using namespace torch::nn;
using namespace torch::test;

struct ClipGradTest : torch::test::SeedingFixture {};


class TestModel : public torch::nn::Module {
 public:
  TestModel()
      : l1(register_module("l1", Linear(10, 3))),
        l2(register_module("l2", Linear(3, 5))),
        l3(register_module("l3", Linear(5, 100))) {}

  Linear l1, l2, l3;
};


TEST_F(ClipGradTest, ClipGrad) {
  TestModel m;
  ASSERT_FLOAT_EQ(torch::nn::clip_grad_norm_(m.parameters(), 10.0, 3.0), 3.0);
}
