/**
 * Copyright (c) 2016-present, Facebook, Inc.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#pragma once

#include "caffe2/core/tensor.h"

namespace caffe2 {

// This is a wrapper around the TensorPrinter that doesn't require the user to
// explicit specify the type of the tensor while calling the Print() method.
// It also supports a convenience function with a default constructed printer as
// a static method.
class SmartTensorPrinter {
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

  void Print(const Tensor<CPUContext>& tensor);

  template <class Context>
  void PrintMeta(const Tensor<Context>& tensor) {
    tensorPrinter_.PrintMeta(tensor);
  }

  // Uses a default constructed SmartTensorPrinter
  static void PrintTensor(const Tensor<CPUContext>& tensor);

  // Uses a default constructed SmartTensorPrinter
  template <class Context>
  void PrintTensorMeta(const Tensor<Context>& tensor) {
    DefaultTensorPrinter().PrintMeta(tensor);
  }

 private:
  // Returns a thread local default constructed TensorPrinter
  static SmartTensorPrinter& DefaultTensorPrinter();

  TensorPrinter tensorPrinter_;
};
}
