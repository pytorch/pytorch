#include <gtest/gtest.h>

#include <torch/torch.h>

#include <test/cpp/api/support.h>

struct RandomTest : torch::test::SeedingFixture {};

TEST_F(RandomTest, GetAndSetRNGState) {
  auto state = torch::get_rng_state();
  auto stateCloned = state.clone();
  auto before = torch::rand({1000});

  ASSERT_TRUE(torch::equal(state, stateCloned));
  ASSERT_FALSE(torch::equal(state, torch::get_rng_state()));

  torch::set_rng_state(state);
  auto after = torch::rand({1000});
  ASSERT_TRUE(torch::equal(before, after));
}

TEST_F(RandomTest, GetAndSetRNGState_CUDA) {
  auto state = torch::cuda::get_rng_state();
  auto stateCloned = state.clone();
  auto before = torch::rand({1000}, torch::Device(torch::kCUDA));

  ASSERT_TRUE(torch::equal(state, stateCloned));
  ASSERT_FALSE(torch::equal(state, torch::cuda::get_rng_state()));

  torch::cuda::set_rng_state(state);
  auto after = torch::rand({1000}, torch::Device(torch::kCUDA));
  ASSERT_TRUE(torch::equal(before, after));

  ASSERT_THROWS_WITH(torch::cuda::get_rng_state(torch::Device(torch::kCPU)), "only supports CUDA device");
  ASSERT_THROWS_WITH(torch::cuda::set_rng_state(state, torch::Device(torch::kCPU)), "only supports CUDA device");
}
