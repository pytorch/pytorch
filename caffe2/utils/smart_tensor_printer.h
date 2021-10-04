#pragma once

#include "caffe2/core/tensor.h"

namespace caffe2 {

// This is a wrapper around the TensorPrinter that doesn't require the user to
// explicit specify the type of the tensor while calling the Print() method.
// It also supports a convenience function with a default constructed printer as
// a static method.
class TORCH_API SmartTensorPrinter {
 public:
  // The proliferation of constructors is to give the feature parity with
  // TensorPrinter
  // yet not repeat the default arguments explicitly in case they change in the
  // future.
  SmartTensorPrinter() = default;

  explicit SmartTensorPrinter(const std::string& tensor_name);

  SmartTensorPrinter(
      const std::string& tensor_name,
      const std::string& file_name);

  SmartTensorPrinter(
      const std::string& tensor_name,
      const std::string& file_name,
      int limit);

  void Print(const Tensor& tensor);

  void PrintMeta(const Tensor& tensor) {
    tensorPrinter_.PrintMeta(tensor);
  }

  // Uses a default constructed SmartTensorPrinter
  static void PrintTensor(const Tensor& tensor);

  // Uses a default constructed SmartTensorPrinter
  void PrintTensorMeta(const Tensor& tensor) {
    DefaultTensorPrinter().PrintMeta(tensor);
  }

 private:
  // Returns a thread local default constructed TensorPrinter
  static SmartTensorPrinter& DefaultTensorPrinter();

  TensorPrinter tensorPrinter_;
};
}
