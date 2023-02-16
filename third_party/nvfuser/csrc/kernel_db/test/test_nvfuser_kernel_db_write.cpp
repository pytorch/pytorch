#include <gmock/gmock-matchers.h>
#include <gtest/gtest.h>

#include <torch/torch.h>

#include <kernel_db/kernel_db.h>
#include <kernel_db/utils.h>
#include <test/test_gpu_validator.h>
#include <test/test_utils.h>

// RUN CMD: bin/test_jit --gtest_filter="NVFuserTest*KernelDb_Write*"
namespace nvfuser {

TEST_F(NVFuserTest, KernelDb_Write_CUDA) {
  // Setup the test db
  fs::path test_data =
      fs::path(__FILE__).parent_path() / "test_data/kernel_db_for_query_test";
  ASSERT_TRUE(fs::is_directory(test_data));
  fs::path test_data_cubin = test_data / "kernel_0.cubin";
  fs::path test_data_kernel = test_data / "kernel_0.cu";
  ASSERT_TRUE(fs::is_regular_file(test_data_cubin));
  ASSERT_TRUE(fs::is_regular_file(test_data_kernel));

  const std::string kernel_db_dir("nvfuser_kernel_db_write_test");
  const std::string kernel_db_file("db.csv");
  fs::path test_db_path = fs::temp_directory_path() / kernel_db_dir;
  if (fs::is_directory(test_db_path)) {
    fs::remove_all(test_db_path);
  }

  auto& kernel_db =
      KernelDb::get(kernel_db_dir, kernel_db_file, true, false, true);
  ASSERT_TRUE(kernel_db.enabled());
  ASSERT_TRUE(kernel_db.size() == 0);

  // Setup data for DB entry to write
  std::string code;
  const std::string compile_args(
      "--std=c++14 --gpu-architecture=sm_80 -default-device --fmad=true -DNDEBUG --ptxas-options --maxrregcount=255");
  const std::string kernel_signature(
      "_ZN11CudaCodeGen7kernel1ENS_6TensorIfLi3EEES1_S1_");
  std::vector<char> cubin;
  ASSERT_TRUE(copy_from_text_file(test_data_kernel, code));
  ASSERT_TRUE(copy_from_binary_file(test_data_cubin, cubin));

  // Test a successful write to the db
  try {
    ASSERT_TRUE(kernel_db.write(code, compile_args, kernel_signature, cubin));
    ASSERT_TRUE(kernel_db.size() == 1);
    SUCCEED();
  } catch (const std::exception& e) {
    FAIL() << "Unexpected failure while writing db entry!" << e.what();
  }

  // Try a write for an entry that already exists
  try {
    ASSERT_FALSE(kernel_db.write(code, compile_args, kernel_signature, cubin));
    ASSERT_TRUE(kernel_db.size() == 1);
    SUCCEED();
  } catch (const std::exception& e) {
    FAIL() << "Unexpected failure while writing existing db entry!" << e.what();
  }

  // Cleanup DB Directory
  if (fs::is_directory(test_db_path)) {
    fs::remove_all(test_db_path);
  }
}

} // namespace nvfuser
