#include <c10/core/AllocatorConfig.h>
#include <c10/utils/env.h>

#include <gtest/gtest.h>

TEST(AllocatorConfigTest, allocator_config_test) {
  auto env_orig = c10::utils::get_env("PYTORCH_ALLOC_CONF");

  std::string env =
      "max_split_size_mb:40,"
      "max_non_split_rounding_mb:30,"
      "garbage_collection_threshold:0.5,"
      "roundup_power2_divisions:[64:8,128:2,256:2,512:2,1024:1,>:1],"
      "backend:async,"
      "expandable_segments:True,"
      "release_lock_on_device_malloc:True,"
      "pinned_use_device_host_register:True,"
      "parsePinnedNumRegisterThreads:8,"
      "pinned_use_background_threads:True";
  c10::utils::set_env("PYTORCH_ALLOC_CONF", env.c_str());
  c10::CachingAllocator::AllocatorConfig& config =
      c10::CachingAllocator::AllocatorConfig::instance();
  EXPECT_EQ(config.last_allocator_settings(), env);
  EXPECT_EQ(config.max_split_size(), 40 * 1024 * 1024);
  EXPECT_EQ(config.max_non_split_rounding_size(), 30 * 1024 * 1024);
  EXPECT_EQ(config.garbage_collection_threshold(), 0.5);
  EXPECT_EQ(config.roundup_power2_divisions(32), 1);
  EXPECT_EQ(config.roundup_power2_divisions(64), 8);
  EXPECT_EQ(config.roundup_power2_divisions(128), 2);
  EXPECT_EQ(config.roundup_power2_divisions(256), 2);
  EXPECT_EQ(config.roundup_power2_divisions(512), 2);
  EXPECT_EQ(config.roundup_power2_divisions(1024), 1);
  EXPECT_EQ(config.roundup_power2_divisions(2048), 1);
  EXPECT_EQ(config.roundup_power2_divisions(4096), 1);
  EXPECT_EQ(config.roundup_power2_divisions(8192), 1);
  EXPECT_EQ(config.use_async_allocator(), true);
  EXPECT_EQ(config.use_expandable_segments(), true);
  EXPECT_EQ(config.use_release_lock_on_device_malloc(), true);
  EXPECT_EQ(config.pinned_use_device_host_register(), true);
  EXPECT_EQ(config.pinned_num_register_threads(), 8);
  EXPECT_EQ(config.pinned_use_background_threads(), true);

  env =
      "max_split_size_mb:20,"
      "max_non_split_rounding_mb:40,"
      "garbage_collection_threshold:0.8";
  c10::CachingAllocator::setAllocatorSettings(env);
  EXPECT_EQ(config.last_allocator_settings(), env);
  EXPECT_EQ(config.max_split_size(), 20 * 1024 * 1024);
  EXPECT_EQ(config.max_non_split_rounding_size(), 40 * 1024 * 1024);
  EXPECT_EQ(config.garbage_collection_threshold(), 0.8);

  env = "roundup_power2_divisions:[128:8,256:6,512:4,2048:4>:2]";
  c10::CachingAllocator::setAllocatorSettings(env);
  EXPECT_EQ(config.last_allocator_settings(), env);
  EXPECT_EQ(config.roundup_power2_divisions(64), 2);
  EXPECT_EQ(config.roundup_power2_divisions(128), 8);
  EXPECT_EQ(config.roundup_power2_divisions(256), 6);
  EXPECT_EQ(config.roundup_power2_divisions(512), 4);
  EXPECT_EQ(config.roundup_power2_divisions(1024), 2);
  EXPECT_EQ(config.roundup_power2_divisions(2048), 4);
  EXPECT_EQ(config.roundup_power2_divisions(4096), 2);

  env =
      "backend:native,"
      "expandable_segments:False,"
      "release_lock_on_device_malloc:False";
  c10::CachingAllocator::setAllocatorSettings(env);
  EXPECT_EQ(config.last_allocator_settings(), env);
  EXPECT_EQ(config.use_async_allocator(), false);
  EXPECT_EQ(config.use_expandable_segments(), false);
  EXPECT_EQ(config.use_release_lock_on_device_malloc(), false);

  env =
      "pinned_use_device_host_register:False,"
      "pinned_num_register_threads:4,"
      "pinned_use_background_threads:False";
  c10::CachingAllocator::setAllocatorSettings(env);
  EXPECT_EQ(config.last_allocator_settings(), env);
  EXPECT_EQ(config.pinned_use_device_host_register(), false);
  EXPECT_EQ(config.pinned_num_register_threads(), 4);
  EXPECT_EQ(config.pinned_use_background_threads(), false);

  // Reset the environment variable to its original value
  if (env_orig) {
    c10::utils::set_env("PYTORCH_ALLOC_CONF", env_orig->c_str());
  } else {
    c10::utils::set_env("PYTORCH_ALLOC_CONF", "");
  }
}
