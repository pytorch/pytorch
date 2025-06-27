#include <c10/core/AllocatorConfig.h>
#include <c10/util/env.h>

#include <gtest/gtest.h>

TEST(AllocatorConfigTest, allocator_config_test) {
  using namespace c10::CachingAllocator;
  auto env_orig = c10::utils::get_env("PYTORCH_ALLOC_CONF");
  constexpr size_t kMB = 1024 * 1024ul;

  std::string env =
      "max_split_size_mb:40,"
      "max_non_split_rounding_mb:30,"
      "garbage_collection_threshold:0.5,"
      "roundup_power2_divisions:[64:8,128:2,256:4,512:2,1024:4,>:1],"
      "expandable_segments:True,"
      "pinned_use_background_threads:True,"
      "UNKNOWN_OPTION:42";
  c10::utils::set_env("PYTORCH_ALLOC_CONF", env.c_str());
  EXPECT_EQ(c10::CachingAllocator::getAllocatorSettings(), env);
  EXPECT_EQ(AllocatorConfig::max_split_size(), 40 * kMB);
  EXPECT_EQ(AllocatorConfig::max_non_split_rounding_size(), 30 * kMB);
  EXPECT_EQ(AllocatorConfig::garbage_collection_threshold(), 0.5);
  EXPECT_EQ(AllocatorConfig::roundup_power2_divisions(32 * kMB), 8);
  EXPECT_EQ(AllocatorConfig::roundup_power2_divisions(64 * kMB), 8);
  EXPECT_EQ(AllocatorConfig::roundup_power2_divisions(128 * kMB), 2);
  EXPECT_EQ(AllocatorConfig::roundup_power2_divisions(256 * kMB), 4);
  EXPECT_EQ(AllocatorConfig::roundup_power2_divisions(512 * kMB), 2);
  EXPECT_EQ(AllocatorConfig::roundup_power2_divisions(1024 * kMB), 4);
  EXPECT_EQ(AllocatorConfig::roundup_power2_divisions(2048 * kMB), 1);
  EXPECT_EQ(AllocatorConfig::roundup_power2_divisions(4096 * kMB), 1);
  EXPECT_EQ(AllocatorConfig::roundup_power2_divisions(8192 * kMB), 1);
  EXPECT_EQ(AllocatorConfig::use_expandable_segments(), true);
  EXPECT_EQ(AllocatorConfig::pinned_use_background_threads(), true);

  env =
      "max_split_size_mb:20,"
      "max_non_split_rounding_mb:40,"
      "UNKNOWN_OPTION:[1,2],"
      "garbage_collection_threshold:0.8";
  c10::CachingAllocator::setAllocatorSettings(env);
  EXPECT_EQ(c10::CachingAllocator::getAllocatorSettings(), env);
  EXPECT_EQ(AllocatorConfig::max_split_size(), 20 * kMB);
  EXPECT_EQ(AllocatorConfig::max_non_split_rounding_size(), 40 * kMB);
  EXPECT_EQ(AllocatorConfig::garbage_collection_threshold(), 0.8);

  // roundup_power2_divisions knob array syntax
  env = "roundup_power2_divisions:[128:8,256:16,512:1,2048:8,>:2]";
  c10::CachingAllocator::setAllocatorSettings(env);
  EXPECT_EQ(c10::CachingAllocator::getAllocatorSettings(), env);
  EXPECT_EQ(AllocatorConfig::roundup_power2_divisions(64 * kMB), 8);
  EXPECT_EQ(AllocatorConfig::roundup_power2_divisions(128 * kMB), 8);
  EXPECT_EQ(AllocatorConfig::roundup_power2_divisions(256 * kMB), 16);
  EXPECT_EQ(AllocatorConfig::roundup_power2_divisions(512 * kMB), 1);
  EXPECT_EQ(AllocatorConfig::roundup_power2_divisions(1024 * kMB), 0);
  EXPECT_EQ(AllocatorConfig::roundup_power2_divisions(2048 * kMB), 8);
  EXPECT_EQ(AllocatorConfig::roundup_power2_divisions(4096 * kMB), 2);

  // roundup_power2_divisions single value syntax for backward compatibility
  env = "roundup_power2_divisions:4";
  c10::CachingAllocator::setAllocatorSettings(env);
  EXPECT_EQ(c10::CachingAllocator::getAllocatorSettings(), env);
  EXPECT_EQ(AllocatorConfig::roundup_power2_divisions(64 * kMB), 4);
  EXPECT_EQ(AllocatorConfig::roundup_power2_divisions(256 * kMB), 4);
  EXPECT_EQ(AllocatorConfig::roundup_power2_divisions(2048 * kMB), 4);

  env = "expandable_segments:False,";
  c10::CachingAllocator::setAllocatorSettings(env);
  EXPECT_EQ(c10::CachingAllocator::getAllocatorSettings(), env);
  EXPECT_EQ(AllocatorConfig::use_expandable_segments(), false);

  env = "pinned_use_background_threads:False";
  c10::CachingAllocator::setAllocatorSettings(env);
  EXPECT_EQ(c10::CachingAllocator::getAllocatorSettings(), env);
  EXPECT_EQ(AllocatorConfig::pinned_use_background_threads(), false);

  // Reset the environment variable to its original value
  if (env_orig) {
    c10::utils::set_env("PYTORCH_ALLOC_CONF", env_orig->c_str());
  } else {
    c10::utils::set_env("PYTORCH_ALLOC_CONF", "");
  }
}
