#ifndef CAFFE2_UTILS_TEST_UTILS_H_
#define CAFFE2_UTILS_TEST_UTILS_H_

#include "caffe2/core/tensor.h"
#include "caffe2/core/workspace.h"

// Utilities that make it easier to write caffe2 C++ unit tests.
// These utils are designed to be concise and easy to use. They may sacrifice
// performance and should only be used in tests/non production code.
namespace caffe2 {
namespace testing {

// Asserts that the values of two tensors are the same.
void assertTensorEquals(const TensorCPU& tensor1, const TensorCPU& tensor2);

// Asserts a list of tensors presented in two workspaces are equal.
void assertTensorListEquals(
    const std::vector<std::string>& tensorNames,
    const Workspace& workspace1,
    const Workspace& workspace2);

// Read a tensor from the workspace.
const caffe2::Tensor& getTensor(
    const caffe2::Workspace& workspace,
    const std::string& name);

// Create a new tensor in the workspace.
caffe2::Tensor* createTensor(
    const std::string& name,
    caffe2::Workspace* workspace);

// Create a new operator in the net.
caffe2::OperatorDef* createOperator(
    const std::string& type,
    const std::vector<string>& inputs,
    const std::vector<string>& outputs,
    caffe2::NetDef* net);

// Fill data from a vector to a tensor.
template <typename T>
void fillTensor(
    const vector<int64_t>& shape,
    const vector<T>& data,
    TensorCPU* tensor) {
  tensor->Resize(shape);
  CAFFE_ENFORCE_EQ(data.size(), tensor->numel());
  auto ptr = tensor->mutable_data<T>();
  for (int i = 0; i < tensor->numel(); ++i) {
    ptr[i] = data[i];
  }
}

// Create a tensor and fill data.
template <typename T>
caffe2::Tensor* createTensorAndFill(
    const string& name,
    const vector<int64_t>& shape,
    const vector<T>& data,
    Workspace* workspace) {
  auto* tensor = createTensor(name, workspace);
  fillTensor<T>(shape, data, tensor);
  return tensor;
}

// Fill a constant to a tensor.
template <typename T>
void constantFillTensor(
    const vector<int64_t>& shape,
    const T& data,
    TensorCPU* tensor) {
  tensor->Resize(shape);
  auto ptr = tensor->mutable_data<T>();
  for (int i = 0; i < tensor->numel(); ++i) {
    ptr[i] = data;
  }
}

// Create a tensor and fill a constant.
template <typename T>
caffe2::Tensor* createTensorAndConstantFill(
    const string& name,
    const vector<int64_t>& shape,
    const T& data,
    Workspace* workspace) {
  auto* tensor = createTensor(name, workspace);
  constantFillTensor<T>(shape, data, tensor);
  return tensor;
}

// Coincise util class to mutate a net in a chaining fashion.
class NetMutator {
 public:
  explicit NetMutator(caffe2::NetDef* net) : net_(net) {}

  NetMutator& newOp(
      const std::string& type,
      const std::vector<string>& inputs,
      const std::vector<string>& outputs);

 private:
  caffe2::NetDef* net_;
};

// Coincise util class to mutate a workspace in a chaining fashion.
class WorkspaceMutator {
 public:
  explicit WorkspaceMutator(caffe2::Workspace* workspace)
      : workspace_(workspace) {}

  // New tensor filled by a data vector.
  template <typename T>
  WorkspaceMutator& newTensor(
      const string& name,
      const vector<int64_t>& shape,
      const vector<T>& data) {
    createTensorAndFill<T>(name, shape, data, workspace_);
    return *this;
  }

  // New tensor filled by a constant.
  template <typename T>
  WorkspaceMutator& newTensorConst(
      const string& name,
      const vector<int64_t>& shape,
      const T& data) {
    createTensorAndConstantFill<T>(name, shape, data, workspace_);
    return *this;
  }

 private:
  caffe2::Workspace* workspace_;
};

} // namespace testing
} // namespace caffe2

#endif // CAFFE2_UTILS_TEST_UTILS_H_
