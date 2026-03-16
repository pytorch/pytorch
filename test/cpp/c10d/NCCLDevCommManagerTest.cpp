#include <gtest/gtest.h>

// Include nccl_dev_cap.hpp first to define NCCL_HAS_SYMMEM_DEVICE_SUPPORT
#include <torch/csrc/distributed/c10d/symm_mem/nccl_dev_cap.hpp>

#ifdef NCCL_HAS_SYMMEM_DEVICE_SUPPORT

#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <c10/cuda/CUDAStream.h>
#include <c10/util/Exception.h>
#include <torch/csrc/distributed/c10d/symm_mem/nccl_devcomm_manager.hpp>

using namespace c10d::symmetric_memory;

// Note: This test requires NCCL with symmetric memory device support.
// The ncclDevComm type is defined in nccl_device.h (included via
// nccl_dev_cap.hpp). For testing without full NCCL initialization, we use
// uninitialized values. In a real scenario, you would use ncclDevCommCreate to
// create valid communicators.

// Test register_comm and get_comm
void test_register_and_get_comm() {
  const std::string group_name = "test_group_comm";
  c10::Device device(c10::DeviceType::CUDA, 0);

  auto& manager = NCCLDevCommManager::get(device);

  // Register a host communicator
  ncclComm_t mock_comm = nullptr; // In real test, this would be a valid comm
  manager.register_comm(group_name, mock_comm);

  // Get the communicator back
  ncclComm_t retrieved_comm = manager.get_comm(group_name);
  EXPECT_EQ(retrieved_comm, mock_comm);

  // Registering the same communicator again should not throw
  manager.register_comm(group_name, mock_comm);

  // Getting from non-existent group should throw
  EXPECT_THROW(manager.get_comm("nonexistent_group"), c10::Error);
}

// Test basic registration and retrieval with default function name key
void test_register_and_get_default_key() {
  const std::string group_name = "test_group";
  c10::Device device(c10::DeviceType::CUDA, 0);

  auto& manager = NCCLDevCommManager::get(device);

  // Create a test device communicator (uninitialized for testing)
  // In real usage, this would be created via ncclDevCommCreate
  ncclDevComm devcomm0 = {};

  // Register with default key (function name)
  auto devcomm_opt = manager.register_devcomm(group_name, devcomm0);
  EXPECT_TRUE(devcomm_opt.has_value());

  // Get with default key
  auto retrieved_opt = manager.get_devcomm(group_name);
  EXPECT_TRUE(retrieved_opt.has_value());
  // Note: We can't easily compare ncclDevComm values without proper
  // initialization This test verifies the registration/retrieval logic works
}

// Test registration and retrieval with custom key
void test_register_and_get_custom_key() {
  const std::string group_name = "test_group_custom";
  c10::Device device(c10::DeviceType::CUDA, 0);

  auto& manager = NCCLDevCommManager::get(device);

  // Create test device communicators (uninitialized for testing)
  ncclDevComm devcomm0 = {};
  ncclDevComm devcomm1 = {};

  // Register with custom keys
  auto devcomm0_opt =
      manager.register_devcomm(group_name, devcomm0, "custom_key0");
  EXPECT_TRUE(devcomm0_opt.has_value());

  auto devcomm1_opt =
      manager.register_devcomm(group_name, devcomm1, "custom_key1");
  EXPECT_TRUE(devcomm1_opt.has_value());

  // Get with custom keys
  auto retrieved0_opt = manager.get_devcomm(group_name, "custom_key0");
  EXPECT_TRUE(retrieved0_opt.has_value());

  auto retrieved1_opt = manager.get_devcomm(group_name, "custom_key1");
  EXPECT_TRUE(retrieved1_opt.has_value());
}

// Test that registering the same key twice throws an error
void test_duplicate_registration_throws() {
  const std::string group_name = "test_group_duplicate";
  c10::Device device(c10::DeviceType::CUDA, 0);

  auto& manager = NCCLDevCommManager::get(device);

  // Create test device communicators (uninitialized for testing)
  ncclDevComm devcomm0 = {};
  ncclDevComm devcomm1 = {};

  // Register first time - should succeed
  auto devcomm0_opt =
      manager.register_devcomm(group_name, devcomm0, "duplicate_key");
  EXPECT_TRUE(devcomm0_opt.has_value());

  // Register second time with same key - should throw
  EXPECT_THROW(
      manager.register_devcomm(group_name, devcomm1, "duplicate_key"),
      c10::Error);
}

// Test that getting a non-existent key returns nullopt
void test_get_nonexistent_key() {
  const std::string group_name = "test_group_nonexistent";
  c10::Device device(c10::DeviceType::CUDA, 0);

  auto& manager = NCCLDevCommManager::get(device);

  // Try to get a key that doesn't exist
  auto retrieved_opt = manager.get_devcomm(group_name, "nonexistent_key");
  EXPECT_FALSE(retrieved_opt.has_value());

  // Try to get from a group that doesn't exist
  auto retrieved_group_opt =
      manager.get_devcomm("nonexistent_group", "any_key");
  EXPECT_FALSE(retrieved_group_opt.has_value());
}

// Test multiple groups with different keys
void test_multiple_groups() {
  const std::string group1 = "group1";
  const std::string group2 = "group2";
  c10::Device device(c10::DeviceType::CUDA, 0);

  auto& manager = NCCLDevCommManager::get(device);

  // Create test device communicators (uninitialized for testing)
  ncclDevComm devcomm1a = {};
  ncclDevComm devcomm1b = {};
  ncclDevComm devcomm2a = {};

  // Register in group1
  auto devcomm1a_opt = manager.register_devcomm(group1, devcomm1a, "key1a");
  EXPECT_TRUE(devcomm1a_opt.has_value());
  auto devcomm1b_opt = manager.register_devcomm(group1, devcomm1b, "key1b");
  EXPECT_TRUE(devcomm1b_opt.has_value());

  // Register in group2
  auto devcomm2a_opt = manager.register_devcomm(group2, devcomm2a, "key2a");
  EXPECT_TRUE(devcomm2a_opt.has_value());

  // Verify we can retrieve from both groups
  auto retrieved1a_opt = manager.get_devcomm(group1, "key1a");
  EXPECT_TRUE(retrieved1a_opt.has_value());

  auto retrieved1b_opt = manager.get_devcomm(group1, "key1b");
  EXPECT_TRUE(retrieved1b_opt.has_value());

  auto retrieved2a_opt = manager.get_devcomm(group2, "key2a");
  EXPECT_TRUE(retrieved2a_opt.has_value());

  // Verify keys from one group don't affect the other
  auto cross_group_opt = manager.get_devcomm(group1, "key2a");
  EXPECT_FALSE(cross_group_opt.has_value());
}

// Test the example pattern from the comments
void test_example_pattern() {
  const std::string group_name = "example_group";
  c10::Device device(c10::DeviceType::CUDA, 0);

  auto& manager = NCCLDevCommManager::get(device);

  // Example: foo function pattern
  auto foo = [&manager, &group_name]() {
    // Try to get first
    auto devcomm_opt = manager.get_devcomm(group_name);
    if (!devcomm_opt) {
      // Not found, create then register
      ncclDevComm devcomm = {}; // In real code: ncclDevCommCreate(...)
      devcomm_opt = manager.register_devcomm(group_name, devcomm);
    }
    ncclDevComm& devcomm_ref = devcomm_opt->get();
    (void)devcomm_ref; // In real code, used for NCCL operations
    EXPECT_TRUE(devcomm_opt.has_value());
  };

  foo();

  // Example: bar function pattern with multiple keys
  auto bar = [&manager, &group_name]() {
    ncclDevComm devcomm0 = {}; // In real code: ncclDevCommCreate(...)
    ncclDevComm devcomm1 = {}; // In real code: ncclDevCommCreate(...)
    // Register with custom keys
    manager.register_devcomm(group_name, devcomm0, "bar0");
    manager.register_devcomm(group_name, devcomm1, "bar1");

    // Verify both are registered
    auto bar0_opt = manager.get_devcomm(group_name, "bar0");
    EXPECT_TRUE(bar0_opt.has_value());

    auto bar1_opt = manager.get_devcomm(group_name, "bar1");
    EXPECT_TRUE(bar1_opt.has_value());
  };

  bar();
}

// Note: This test file requires CUDA and NCCL to be available.
// The actual ncclDevComm type would be used in real scenarios.
// For unit testing, we're using a simple integer as a placeholder.
// In a real test environment, you would need to:
// 1. Initialize CUDA
// 2. Initialize NCCL
// 3. Create actual ncclDevComm objects using ncclDevCommCreate
// 4. Clean up properly

TEST(NCCLDevCommManagerTest, RegisterAndGetComm) {
  if (!at::cuda::is_available()) {
    GTEST_SKIP() << "CUDA not available, skipping test";
  }
  test_register_and_get_comm();
}

TEST(NCCLDevCommManagerTest, RegisterAndGetDefaultKey) {
  if (!at::cuda::is_available()) {
    GTEST_SKIP() << "CUDA not available, skipping test";
  }
  test_register_and_get_default_key();
}

TEST(NCCLDevCommManagerTest, RegisterAndGetCustomKey) {
  if (!at::cuda::is_available()) {
    GTEST_SKIP() << "CUDA not available, skipping test";
  }
  test_register_and_get_custom_key();
}

TEST(NCCLDevCommManagerTest, DuplicateRegistrationThrows) {
  if (!at::cuda::is_available()) {
    GTEST_SKIP() << "CUDA not available, skipping test";
  }
  test_duplicate_registration_throws();
}

TEST(NCCLDevCommManagerTest, GetNonexistentKey) {
  if (!at::cuda::is_available()) {
    GTEST_SKIP() << "CUDA not available, skipping test";
  }
  test_get_nonexistent_key();
}

TEST(NCCLDevCommManagerTest, MultipleGroups) {
  if (!at::cuda::is_available()) {
    GTEST_SKIP() << "CUDA not available, skipping test";
  }
  test_multiple_groups();
}

TEST(NCCLDevCommManagerTest, ExamplePattern) {
  if (!at::cuda::is_available()) {
    GTEST_SKIP() << "CUDA not available, skipping test";
  }
  test_example_pattern();
}

#else // NCCL_HAS_SYMMEM_DEVICE_SUPPORT

TEST(NCCLDevCommManagerTest, Skipped) {
  GTEST_SKIP() << "NCCL symmetric memory device support not available";
}

#endif // NCCL_HAS_SYMMEM_DEVICE_SUPPORT
