#include <iostream>

#include "caffe2/core/init.h"
#include "caffe2/core/operator.h"

int main(int argc, char** argv) {
  caffe2::GlobalInit(&argc, &argv);
  std::cout << "CPU operator registry:" << std::endl;
  caffe2::CPUOperatorRegistry()->TEST_PrintRegisteredNames();
  std::cout << "CUDA operator registry:" << std::endl;
  caffe2::CUDAOperatorRegistry()->TEST_PrintRegisteredNames();
}
