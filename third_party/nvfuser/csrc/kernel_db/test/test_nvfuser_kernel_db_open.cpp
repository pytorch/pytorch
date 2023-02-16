#include <gmock/gmock-matchers.h>
#include <gtest/gtest.h>

#include <torch/torch.h>

#include <kernel_db/kernel_db.h>
#include <kernel_db/utils.h>
#include <test/test_gpu_validator.h>
#include <test/test_utils.h>

// RUN CMD: bin/test_jit --gtest_filter="NVFuserTest*KernelDb_Open*"

namespace nvfuser {

TEST_F(NVFuserTest, KernelDb_Open_CUDA) {
  // Check a corrupted DB and reset the DB
  // 1.) Test writes a bad db.csv file and open fails to match header
  // 2.) Should delete cubins because of bad db.csv file.
  // 3.) Creates a new empty db.csv file with proper header
  try {
    const std::string kernel_db_dir("nvfuser_kernel_db_open_test");
    const std::string bad_text("blahblahblah\n");
    const std::string header(
        "kernel_signature,compile_args,kernel_code_file,cubin_file");
    fs::path test_db_path = fs::temp_directory_path() / kernel_db_dir;
    if (fs::is_directory(test_db_path)) {
      fs::remove_all(test_db_path);
    }
    ASSERT_TRUE(fs::create_directory(test_db_path));

    // Setup 1
    const std::string kernel_db_file("db.csv");
    fs::path test_db_file = test_db_path / kernel_db_file;
    ASSERT_FALSE(fs::is_regular_file(test_db_file));
    ASSERT_TRUE(copy_to_text_file(test_db_file.string(), bad_text));
    ASSERT_TRUE(fs::is_regular_file(test_db_file));
    // Setup 2
    fs::path test_cubin_file = test_db_path / "test1.cubin";
    ASSERT_TRUE(copy_to_text_file(test_cubin_file.string(), bad_text));
    // Execute 1, 2, 3
    KernelDb::get(kernel_db_dir, kernel_db_file, true, false, true);
    // Check 1
    ASSERT_TRUE(fs::is_regular_file(test_db_file));
    // Check 2
    ASSERT_FALSE(fs::is_regular_file(test_cubin_file));
    // Check 3
    std::ifstream db_file(test_db_file.c_str(), std::ios::in);
    std::string line;
    std::getline(db_file, line);
    ASSERT_TRUE(header == line);

    // Cleanup DB Directory
    if (fs::is_directory(test_db_path)) {
      fs::remove_all(test_db_path);
    }
    SUCCEED();
  } catch (const std::exception& e) {
    FAIL() << "Failed replacing a bad db.csv file and removing cubins!"
           << e.what();
  }

  // Check a successful opening of an existing DB
  try {
    // Setup DB Directory
    const std::string kernel_db_dir("nvfuser_kernel_db_test");
    fs::path test_db_path = fs::temp_directory_path() / kernel_db_dir;
    if (fs::is_directory(test_db_path)) {
      fs::remove_all(test_db_path);
    }
    ASSERT_TRUE(fs::create_directory(test_db_path));

    // Setup DB File and corresponding fake cubins and cuda files
    const std::string header(
        "kernel_signature,compile_args,kernel_code_file,cubin_file\n");
    const std::string test_text("blahblahblah\n");
    const std::string kernel_db_file("db.csv");
    const std::string kernel_db_cubin("test1.cubin");
    const std::string kernel_db_kernel("test1.cu");
    fs::path test_db_file_path = test_db_path / kernel_db_file;
    fs::path test_cubin_path = test_db_path / kernel_db_cubin;
    fs::path test_kernel_path = test_db_path / kernel_db_kernel;

    copy_to_text_file(test_db_file_path, header);
    const std::string db_line = test_text + "," + test_text + "," +
        kernel_db_kernel + "," + kernel_db_cubin;
    ASSERT_TRUE(append_to_text_file(test_db_file_path, db_line));
    ASSERT_TRUE(copy_to_text_file(test_cubin_path, test_text));
    ASSERT_TRUE(copy_to_text_file(test_kernel_path, test_text));

    // Open Db
    auto& kernel_db = KernelDb::get(kernel_db_dir, kernel_db_file, true, false);

    ASSERT_TRUE(kernel_db.enabled());

    // Cleanup DB Directory
    if (fs::is_directory(test_db_path)) {
      fs::remove_all(test_db_path);
    }

    SUCCEED();
  } catch (const std::exception& e) {
    FAIL() << "Failed to successfully read existing Kernel DB!" << e.what();
  }
}

} // namespace nvfuser
