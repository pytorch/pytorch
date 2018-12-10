#ifndef CAFFE2_UTILS_TEST_UTILS_H_
#define CAFFE2_UTILS_TEST_UTILS_H_

#include "caffe2/core/tensor.h"
#include "caffe2/core/workspace.h"

#include <gtest/gtest.h>

// Utilities that make it easier to write caffe2 C++ unit tests.
namespace caffe2 {
namespace testing {

// Asserts that the numeric values of two tensors are the same.
template <typename T>
void assertsTensorEquals(const TensorCPU& tensor1, const TensorCPU& tensor2) {
  EXPECT_EQ(tensor1.sizes(), tensor2.sizes());
  for (auto idx = 0; idx < tensor1.numel(); ++idx) {
    EXPECT_EQ(tensor1.data<T>()[idx], tensor2.data<T>()[idx]);
  }
}

// Read a tensor from the workspace.
const caffe2::Tensor& getTensor(
    const caffe2::Workspace& workspace,
    const std::string& name);

// Create a new tensor in the workspace.
caffe2::Tensor* createTensor(
    caffe2::Workspace& workspace,
    const std::string& name);

} // namespace testing
} // namespace caffe2

#endif // CAFFE2_UTILS_TEST_UTILS_H_
