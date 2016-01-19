#include <iostream>

#include "caffe2/core/init.h"
#include "caffe2/core/operator.h"

int main(int argc, char** argv) {
  caffe2::GlobalInit(&argc, argv);
  std::cout << "CPU operator registry:" << std::endl;
  for (const auto& key : caffe2::CPUOperatorRegistry()->Keys()) {
    std::cout << "\t" << key << std::endl;
  }
  std::cout << "CUDA operator registry:" << std::endl;
  for (const auto& key : caffe2::CUDAOperatorRegistry()->Keys()) {
    std::cout << "\t" << key << std::endl;
  }
  std::cout << "Operators that have gradients registered:" << std::endl;
  for (const auto& key : caffe2::GradientRegistry()->Keys()) {
    std::cout << "\t" << key << std::endl;
  }
}

