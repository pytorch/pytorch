#include <c10/core/AllocatorConfig.h>

#include <gtest/gtest.h>

using namespace c10::CachingAllocator;
constexpr size_t kMB = 1024 * 1024ul;

struct ExtendedAllocatorConfig {
  static ExtendedAllocatorConfig& instance() {
    static ExtendedAllocatorConfig instance;
    return instance;
  }

  // Returns the device-specific option value in bytes.
  static size_t device_specific_option() {
    return instance().device_specific_option_;
  }

  static const std::unordered_set<std::string>& getKeys() {
    return keys_;
  }

  void parseArgs(const std::string& env) {
    // Parse device-specific options from the environment variable
    ConfigTokenizer tokenizer(env);
    for (size_t i = 0; i < tokenizer.size(); i++) {
      const auto& key = tokenizer[i];
      if (key == "device_specific_option_mb") {
        tokenizer.checkToken(++i, ":");
        device_specific_option_ = tokenizer.toSizeT(++i) * kMB;
      } else {
        i = tokenizer.skipKey(i);
      }

      if (i + 1 < tokenizer.size()) {
        tokenizer.checkToken(++i, ",");
      }
    }
  }

 private:
  // Device-specific option, e.g., memory limit for a specific device.
  std::atomic<size_t> device_specific_option_{0};
  inline static std::unordered_set<std::string> keys_{
      "device_specific_option_mb"};
};

REGISTER_ALLOCATOR_CONFIG_PARSE_HOOK(ExtendedAllocatorConfig)

TEST(AllocatorConfigTest, allocator_config_test) {
  std::string env =
      "max_split_size_mb:40,"
      "max_non_split_rounding_mb:30,"
      "garbage_collection_threshold:0.5,"
      "roundup_power2_divisions:[64:8,128:2,256:4,512:2,1024:4,>:1],"
      "expandable_segments:True,"
      "pinned_use_background_threads:True,"
      "device_specific_option_mb:64";
  c10::CachingAllocator::setAllocatorSettings(env);
  EXPECT_EQ(c10::CachingAllocator::getAllocatorSettings(), env);
  EXPECT_EQ(AcceleratorAllocatorConfig::large_segment_size(), 20 * kMB);
  EXPECT_EQ(AcceleratorAllocatorConfig::max_split_size(), 40 * kMB);
  EXPECT_EQ(
      AcceleratorAllocatorConfig::max_non_split_rounding_size(), 30 * kMB);
  EXPECT_EQ(AcceleratorAllocatorConfig::garbage_collection_threshold(), 0.5);
  EXPECT_EQ(AcceleratorAllocatorConfig::roundup_power2_divisions(32 * kMB), 8);
  EXPECT_EQ(AcceleratorAllocatorConfig::roundup_power2_divisions(64 * kMB), 8);
  EXPECT_EQ(AcceleratorAllocatorConfig::roundup_power2_divisions(128 * kMB), 2);
  EXPECT_EQ(AcceleratorAllocatorConfig::roundup_power2_divisions(256 * kMB), 4);
  EXPECT_EQ(AcceleratorAllocatorConfig::roundup_power2_divisions(512 * kMB), 2);
  EXPECT_EQ(
      AcceleratorAllocatorConfig::roundup_power2_divisions(1024 * kMB), 4);
  EXPECT_EQ(
      AcceleratorAllocatorConfig::roundup_power2_divisions(2048 * kMB), 1);
  EXPECT_EQ(
      AcceleratorAllocatorConfig::roundup_power2_divisions(4096 * kMB), 1);
  EXPECT_EQ(
      AcceleratorAllocatorConfig::roundup_power2_divisions(8192 * kMB), 1);
  EXPECT_EQ(AcceleratorAllocatorConfig::use_expandable_segments(), true);
  EXPECT_EQ(AcceleratorAllocatorConfig::pinned_use_background_threads(), true);
  EXPECT_EQ(ExtendedAllocatorConfig::device_specific_option(), 64 * kMB);

  env =
      "large_segment_size_mb:15,"
      "max_split_size_mb:20,"
      "max_non_split_rounding_mb:40,"
      "garbage_collection_threshold:0.8";
  c10::CachingAllocator::setAllocatorSettings(env);
  EXPECT_EQ(c10::CachingAllocator::getAllocatorSettings(), env);
  EXPECT_EQ(AcceleratorAllocatorConfig::large_segment_size(), 15 * kMB);
  EXPECT_EQ(AcceleratorAllocatorConfig::max_split_size(), 20 * kMB);
  EXPECT_EQ(
      AcceleratorAllocatorConfig::max_non_split_rounding_size(), 40 * kMB);
  EXPECT_EQ(AcceleratorAllocatorConfig::garbage_collection_threshold(), 0.8);

  // roundup_power2_divisions knob array syntax
  env = "roundup_power2_divisions:[128:8,256:16,512:1,2048:8,>:2]";
  c10::CachingAllocator::setAllocatorSettings(env);
  EXPECT_EQ(c10::CachingAllocator::getAllocatorSettings(), env);
  EXPECT_EQ(AcceleratorAllocatorConfig::roundup_power2_divisions(64 * kMB), 8);
  EXPECT_EQ(AcceleratorAllocatorConfig::roundup_power2_divisions(128 * kMB), 8);
  EXPECT_EQ(
      AcceleratorAllocatorConfig::roundup_power2_divisions(256 * kMB), 16);
  EXPECT_EQ(AcceleratorAllocatorConfig::roundup_power2_divisions(512 * kMB), 1);
  EXPECT_EQ(
      AcceleratorAllocatorConfig::roundup_power2_divisions(1024 * kMB), 0);
  EXPECT_EQ(
      AcceleratorAllocatorConfig::roundup_power2_divisions(2048 * kMB), 8);
  EXPECT_EQ(
      AcceleratorAllocatorConfig::roundup_power2_divisions(4096 * kMB), 2);

  // roundup_power2_divisions single value syntax for backward compatibility
  env = "roundup_power2_divisions:4";
  c10::CachingAllocator::setAllocatorSettings(env);
  EXPECT_EQ(c10::CachingAllocator::getAllocatorSettings(), env);
  EXPECT_EQ(AcceleratorAllocatorConfig::roundup_power2_divisions(64 * kMB), 4);
  EXPECT_EQ(AcceleratorAllocatorConfig::roundup_power2_divisions(256 * kMB), 4);
  EXPECT_EQ(
      AcceleratorAllocatorConfig::roundup_power2_divisions(2048 * kMB), 4);

  env = "expandable_segments:False,";
  c10::CachingAllocator::setAllocatorSettings(env);
  EXPECT_EQ(c10::CachingAllocator::getAllocatorSettings(), env);
  EXPECT_EQ(AcceleratorAllocatorConfig::use_expandable_segments(), false);

  env = "pinned_use_background_threads:False";
  c10::CachingAllocator::setAllocatorSettings(env);
  EXPECT_EQ(c10::CachingAllocator::getAllocatorSettings(), env);
  EXPECT_EQ(AcceleratorAllocatorConfig::pinned_use_background_threads(), false);

  // max_split_size_mb must be >= large_segment_size_mb
  env = "max_split_size_mb:20,large_segment_size_mb:25";
  ASSERT_THROW(c10::CachingAllocator::setAllocatorSettings(env), c10::Error);

  env = "foo:123,bar:456";
  ASSERT_THROW(c10::CachingAllocator::setAllocatorSettings(env), c10::Error);
}

TEST(AllocatorConfigTest, malformed_config_raises_value_error) {
  // gh-186453: a malformed allocator config (supplied via PYTORCH_ALLOC_CONF /
  // PYTORCH_CUDA_ALLOC_CONF / PYTORCH_HIP_ALLOC_CONF) must raise a clean
  // c10::ValueError, not a TORCH_INTERNAL_ASSERT that asks the user to report a
  // bug to PyTorch.
  const std::vector<std::string> malformed = {
      "max_split_size_mb", // key with no value (checkIndex)
      "max_split_size_mb:", // trailing colon, no value (checkIndex)
      "roundup_power2_divisions:[64:8", // unterminated list (operator[])
  };
  for (const auto& env : malformed) {
    EXPECT_THROW(
        c10::CachingAllocator::setAllocatorSettings(env), c10::ValueError)
        << "config: " << env;
  }

  // skipKey's closing-bracket guard: an unterminated list value for a key that
  // is skipped (rather than parsed) reaches the end of the config.
  EXPECT_THROW(
      ConfigTokenizer("large_segment_size_mb:[1").skipKey(0), c10::ValueError);

  // The message must be user-facing: no internal-assert text, and it should
  // point the user at the allocator config env var.
  try {
    c10::CachingAllocator::setAllocatorSettings(
        "garbage_collection_threshold:");
    ADD_FAILURE() << "expected a c10::ValueError";
  } catch (const c10::ValueError& e) {
    const std::string msg = e.what_without_backtrace();
    EXPECT_EQ(msg.find("INTERNAL ASSERT"), std::string::npos) << msg;
    EXPECT_NE(msg.find("PYTORCH_ALLOC_CONF"), std::string::npos) << msg;
    // gh-186453: the error must echo the offending config string.
    EXPECT_NE(msg.find("garbage_collection_threshold"), std::string::npos)
        << msg;
  }
}
