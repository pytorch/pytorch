#include <gmock/gmock-matchers.h>
#include <gtest/gtest.h>

#include <torch/torch.h>

#include <kernel_db/kernel_db.h>
#include <kernel_db/utils.h>
#include <test/test_gpu_validator.h>
#include <test/test_utils.h>

// RUN CMD: bin/test_jit --gtest_filter="NVFuserTest*KernelDb_Query*"

namespace nvfuser {

TEST_F(NVFuserTest, KernelDb_Query_CUDA) {
  // Setup the test db
  fs::path test_db =
      fs::path(__FILE__).parent_path() / "test_data/kernel_db_for_query_test";
  ASSERT_TRUE(fs::is_directory(test_db));
  const std::string test_db_file_name("db.csv");
  fs::path test_db_file = test_db / test_db_file_name;
  ASSERT_TRUE(fs::is_regular_file(test_db_file));

  auto& kernel_db =
      KernelDb::get(test_db.string(), test_db_file_name, false, false, true);
  ASSERT_TRUE(kernel_db.enabled());
  ASSERT_TRUE(kernel_db.size() == 1);

  // Check a query with a bad code string
  try {
    const std::string bad_text("blahblahblah");
    std::string dummy_name;
    std::vector<char> dummy_cubin(0);

    ASSERT_FALSE(kernel_db.query(bad_text, bad_text, dummy_name, dummy_cubin));
    SUCCEED();
  } catch (const std::exception& e) {
    FAIL() << "Unexpected failure while querying db for non-existing entry!"
           << e.what();
  }

  // Check a query with a good code string and bad compiler args
  try {
    fs::path code_path = test_db / "kernel_0.cu";
    std::string code;
    ASSERT_TRUE(copy_from_text_file(code_path, code));
    const std::string bad_text("blahblahblah");
    std::string dummy_name;
    std::vector<char> dummy_cubin(0);

    ASSERT_FALSE(kernel_db.query(code, bad_text, dummy_name, dummy_cubin));
    SUCCEED();
  } catch (const std::exception& e) {
    FAIL() << "Unexpected failure while querying db for non-existing entry!"
           << e.what();
  }

  // Check a successful query
  try {
    fs::path code_path = test_db / "kernel_0.cu";
    std::string code;
    ASSERT_TRUE(copy_from_text_file(code_path, code));
    const std::string compiler_args(
        "--std=c++14 --gpu-architecture=sm_80 -default-device --fmad=true -DNDEBUG --ptxas-options --maxrregcount=255");
    std::string dummy_name;
    std::vector<char> dummy_cubin(0);

    ASSERT_TRUE(kernel_db.query(code, compiler_args, dummy_name, dummy_cubin));
    SUCCEED();
  } catch (const std::exception& e) {
    FAIL() << "Unexpected failure while querying db for existing entry!"
           << e.what();
  }
}

} // namespace nvfuser
