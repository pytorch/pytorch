#include <iostream>

#include "caffe2/core/operator.h"

int main(int argc, char** argv) {
  google::InitGoogleLogging(argv[0]);
  std::cout << "CPU operator registry:" << std::endl;
  caffe2::CPUOperatorRegistry()->TEST_PrintRegisteredNames();
  std::cout << "CUDA operator registry:" << std::endl;
  caffe2::CUDAOperatorRegistry()->TEST_PrintRegisteredNames();
}
