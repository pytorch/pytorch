#include <gtest/gtest.h>

#include <exception>

#include <torch/csrc/lazy/core/util.h>

namespace torch {
namespace lazy {

TEST(UtilTest, ExceptionCleanup) {
  std::exception_ptr exception;
  EXPECT_EQ(exception, nullptr);

  {
    ExceptionCleanup cleanup(
        [&](std::exception_ptr&& e) { exception = std::move(e); });

    cleanup.SetStatus(std::make_exception_ptr(std::runtime_error("Oops!")));
  }
  EXPECT_NE(exception, nullptr);

  try {
    std::rethrow_exception(exception);
  } catch (const std::exception& e) {
    EXPECT_STREQ(e.what(), "Oops!");
  }

  exception = nullptr;
  {
    ExceptionCleanup cleanup(
        [&](std::exception_ptr&& e) { exception = std::move(e); });

    cleanup.SetStatus(std::make_exception_ptr(std::runtime_error("")));
    cleanup.Release();
  }
  EXPECT_EQ(exception, nullptr);
}

TEST(UtilTest, MaybeRef) {
  std::string storage("String storage");
  MaybeRef<std::string> refStorage(storage);
  EXPECT_FALSE(refStorage.IsStored());
  EXPECT_EQ(*refStorage, storage);

  MaybeRef<std::string> effStorage(std::string("Vanishing"));
  EXPECT_TRUE(effStorage.IsStored());
  EXPECT_EQ(*effStorage, "Vanishing");
}

TEST(UtilTest, Iota) {
  auto result = Iota<int>(0);
  EXPECT_TRUE(result.empty());

  result = Iota<int>(1);
  EXPECT_EQ(result.size(), 1);
  EXPECT_EQ(result[0], 0);

  result = Iota<int>(2);
  EXPECT_EQ(result.size(), 2);
  EXPECT_EQ(result[0], 0);
  EXPECT_EQ(result[1], 1);

  result = Iota<int>(3, 1, 3);
  EXPECT_EQ(result.size(), 3);
  EXPECT_EQ(result[0], 1);
  EXPECT_EQ(result[1], 4);
  EXPECT_EQ(result[2], 7);
}

} // namespace lazy
} // namespace torch
