#include "caffe2/core/tensor.h"
#include "caffe2/core/workspace.h"

#include "test_utils.h"

namespace caffe2 {
namespace testing {

const caffe2::Tensor& getTensor(
    const caffe2::Workspace& workspace,
    const std::string& name) {
  return workspace.GetBlob(name)->Get<caffe2::Tensor>();
}

caffe2::Tensor* createTensor(
    const std::string& name,
    caffe2::Workspace* workspace) {
  return BlobGetMutableTensor(workspace->CreateBlob(name), caffe2::CPU);
}

} // namespace testing
} // namespace caffe2
