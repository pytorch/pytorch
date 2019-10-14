#include <gtest/gtest.h>

#include <ATen/MemoryFormatUtils.h>
#include <ATen/Config.h>

using namespace at;

TEST(MemoryFormatUtilsTest, CheckSpareMemoryFormat) {
  auto spare_tensor = at::empty({2, 2}, at::dtype<float>().layout(at::kSparse));
  ASSERT_TRUE(spare_tensor.is_sparse());
  auto spare_clone = clone_if_possible_with_memory_format(spare_tensor);
  ASSERT_TRUE(spare_clone.is_sparse());
}

#if AT_MKLDNN_ENABLED()

TEST(MemoryFormatUtilsTest, CheckMkldnnMemoryFormat) {
  auto mkldnn_tensor = at::empty({2, 2}, at::dtype<float>().layout(at::kMkldnn));
  ASSERT_TRUE(mkldnn_tensor.is_mkldnn());
  auto mkldnn_clone = clone_if_possible_with_memory_format(mkldnn_tensor);
  ASSERT_TRUE(mkldnn_clone.is_mkldnn());
}

#endif

TEST(MemoryFormatUtilsTest, CheckContiguousMemoryFormat) {
  auto tensor_contiguous = at::empty({2, 2});
  ASSERT_TRUE(tensor_contiguous.is_contiguous());
  auto contiguous_clone = clone_if_possible_with_memory_format(tensor_contiguous);
  ASSERT_TRUE(contiguous_clone.is_contiguous());
}

TEST(MemoryFormatUtilsTest, CheckNonContiguousMemoryFormat) {
  auto tensor_noncontiguous = at::empty({2, 2}).t();
  ASSERT_FALSE(tensor_noncontiguous.is_contiguous());
  auto noncontiguous_clone = clone_if_possible_with_memory_format(tensor_noncontiguous);
  ASSERT_TRUE(noncontiguous_clone.is_contiguous());
}

