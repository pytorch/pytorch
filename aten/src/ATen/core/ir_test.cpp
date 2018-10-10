#include <ATen/core/ir.h>

#include <gtest/gtest.h>

TEST(IR, Simple) {
  nom::NeuralNet nn;
  auto node = nn.dataFlow.createNode(torch::jit::IValue());
}
