#include <iostream>

#include "caffe2/core/init.h"
#include "caffe2/core/operator.h"
#include "caffe2/core/context.h"
#include "caffe2/core/context_gpu.h"
#include "caffe2/proto/caffe2.pb.h"

#define PRINT_SIZE(cls) \
  std::cout << "Size of " #cls ": " << sizeof(cls) << " bytes." \
            << std::endl;

int main(int /* unused */, char** /* unused */) {
  PRINT_SIZE(caffe2::Blob);
  PRINT_SIZE(caffe2::Tensor<caffe2::CPUContext>);
  PRINT_SIZE(caffe2::Tensor<caffe2::CUDAContext>);
  PRINT_SIZE(caffe2::CPUContext);
  PRINT_SIZE(caffe2::CUDAContext);
  PRINT_SIZE(caffe2::OperatorBase);
  PRINT_SIZE(caffe2::OperatorDef);
  PRINT_SIZE(caffe2::Operator<caffe2::CPUContext>);
  PRINT_SIZE(caffe2::Operator<caffe2::CUDAContext>);
  PRINT_SIZE(caffe2::TypeMeta);
  PRINT_SIZE(caffe2::Workspace);
  return 0;
}
