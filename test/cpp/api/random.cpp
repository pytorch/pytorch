#include <gtest/gtest.h>

#include <torch/torch.h>

struct RandomTest : torch::test::SeedingFixture {};

TEST_F(RandomTest, GetAndSetRNGState) {
  auto state = torch::get_rng_state();
  auto stateCloned = state.clone();
  auto before = torch::rand(1000);

  ASSERT_TRUE(torch::equal(state, stateCloned));
  ASSERT_FALSE(torch::equal(state, torch::get_rng_state()));

  torch::set_rng_state(state);
  auto after = torch::rand(1000);
  ASSERT_TRUE(torch::equal(before, after));
}

TEST_F(RandomTest, GetAndSetRNGState_CUDA) {
  auto state = torch::cuda::get_rng_state();
  auto stateCloned = state.clone();

  auto before = torch::rand({1000}, torch::Device(torch::kCUDA)); // auto before = torch::rand(1000, torch::TensorOptions(torch::Device(torch::kCUDA)));

  ASSERT_TRUE(torch::equal(state, stateCloned));
  ASSERT_FALSE(torch::equal(state, torch::cuda::get_rng_state()));

  torch::cuda::set_rng_state(state);
  auto after = torch::rand({1000}, torch::Device(torch::kCUDA)); // auto after = torch::rand(1000, torch::TensorOptions(torch::Device(torch::kCUDA)));
  ASSERT_TRUE(torch::equal(before, after));
}
