#ifndef CAFFE2_UTILS_TEST_UTILS_H_
#define CAFFE2_UTILS_TEST_UTILS_H_

#include "caffe2/core/tensor.h"
#include "caffe2/core/workspace.h"

// Utilities that make it easier to write caffe2 C++ unit tests.
// These utils are designed to be concise and easy to use. They may sacrifice
// performance and should only be used in tests/non production code.
namespace caffe2 {
namespace testing {

// Asserts that the numeric values of two tensors are the same.
template <typename T>
void assertTensorEquals(const TensorCPU& tensor1, const TensorCPU& tensor2) {
  CAFFE_ENFORCE_EQ(tensor1.sizes(), tensor2.sizes());
  for (auto idx = 0; idx < tensor1.numel(); ++idx) {
    CAFFE_ENFORCE_EQ(tensor1.data<T>()[idx], tensor2.data<T>()[idx]);
  }
}

// Read a tensor from the workspace.
const caffe2::Tensor& getTensor(
    const caffe2::Workspace& workspace,
    const std::string& name);

// Create a new tensor in the workspace.
caffe2::Tensor* createTensor(
    const std::string& name,
    caffe2::Workspace* workspace);

} // namespace testing
} // namespace caffe2

#endif // CAFFE2_UTILS_TEST_UTILS_H_
