#include <gtest/gtest.h>
#include <test/cpp/api/support.h>

#include <torch/csrc/autograd/variable.h>
#include <torch/csrc/utils/tensor_flatten.h>
#include <torch/torch.h>

using namespace torch::test;

TEST(UnflattenDenseTensorTest, TestEmptyTensor) {
  auto emptyTensor1 = at::tensor(std::vector<int>());
  auto emptyTensor2 = at::tensor(std::vector<int>());
  auto tensor1 = at::tensor({1, 2, 3});
  auto tensor2 = at::tensor({4, 5});
  auto tensorList =
      std::vector<at::Tensor>({tensor1, emptyTensor1, emptyTensor2, tensor2});
  auto flatTensor = at::tensor({1, 2, 3, 4, 5});
  auto unflatten_results =
      torch::utils::unflatten_dense_tensors(flatTensor, tensorList);
  ASSERT_EQ(unflatten_results.size(), 4);
  ASSERT_EQ(unflatten_results.at(0).numel(), 3);
  ASSERT_EQ(unflatten_results.at(1).numel(), 0);
  ASSERT_EQ(unflatten_results.at(2).numel(), 0);
  ASSERT_EQ(unflatten_results.at(3).numel(), 2);

  // empty tensor address is 0 as memory is not allocated yet
  ASSERT_EQ(unflatten_results.at(1).data_ptr(), nullptr);
  ASSERT_EQ(unflatten_results.at(2).data_ptr(), nullptr);
  // without fix in unflatten_dense_tensors() for empty tensors,
  // unflattend empty tensor unflatten_results.at(1) will share the same storage
  // as other non-empty tensor like unflatten_results.at(3).
  // after fix, the empty tensor and non-empty tensor do not share the same
  // storage.
  ASSERT_NE(
      unflatten_results.at(1).data_ptr(), unflatten_results.at(3).data_ptr());
  unflatten_results.at(1).resize_(1);
  unflatten_results.at(2).resize_(1);
  // after resizing the two empty tensors, the resized tensors do not share
  // the same storage. without fix in unflatten_dense_tensors() for empty
  // tensors, the resized tensors will share the same storage.
  ASSERT_NE(
      unflatten_results.at(1).data_ptr(), unflatten_results.at(2).data_ptr());
}
