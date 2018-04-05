#define CATCH_CONFIG_MAIN
#include "catch.hpp"

#include <torch/torch.h>

TEST_CASE("C++", "[cpp]") {
  // This compiles.
  torch::nn::LSTM lstm(4, 8);
}
