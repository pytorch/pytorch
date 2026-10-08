#include <gtest/gtest.h>
#include <torch/csrc/stable/c10d.h>

TEST(TorchStableC10d, PythonBridgeRequiresPython) {
  int dummy = 0;
  TorchProcessGroupHandle result = nullptr;
  EXPECT_EQ(
      torch_process_group_from_pyobject(&dummy, &result), AOTI_TORCH_FAILURE);
  EXPECT_EQ(result, nullptr);
}

TEST(TorchStableC10d, InvalidHandles) {
  int64_t rank = -1;
  EXPECT_EQ(torch_process_group_rank(nullptr, &rank), AOTI_TORCH_FAILURE);
  bool completed = false;
  EXPECT_EQ(torch_work_wait(nullptr, 0, &completed), AOTI_TORCH_FAILURE);
  EXPECT_EQ(torch_work_is_completed(nullptr, &completed), AOTI_TORCH_FAILURE);
  EXPECT_EQ(torch_delete_process_group(nullptr), AOTI_TORCH_SUCCESS);
  EXPECT_EQ(torch_delete_work(nullptr), AOTI_TORCH_SUCCESS);
}
