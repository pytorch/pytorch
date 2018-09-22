#ifndef CAFFE2_CORE_TENSOR_H_
#define CAFFE2_CORE_TENSOR_H_

#include "caffe2/core/storage.h"
#include "caffe2/core/tensor_impl.h"

#include <ATen/core/intrusive_ptr.h>
#include <ATen/core/UndefinedTensorImpl.h>
#include <ATen/core/Tensor.h>

namespace caffe2 {

using at::UndefinedTensorImpl;
using at::Tensor;

using TensorCPU = Tensor;

constexpr int k_limit_default_ = 1000;

// TODO: the following logic can be merged into regular Tensor class methods
// after MKLMemory starts to implement Tensor interface

// Type call registry
typedef TypeMeta (*TypeCall)(const void*);
TypeCall GetTypeCallFunction(TypeIdentifier id);
void RegisterTypeCallFunction(TypeIdentifier id, TypeCall c);

// Shape call registry
typedef vector<TIndex> (*TensorInfoCall)(
    const void*,
    size_t* capacity,
    DeviceOption* device);
TensorInfoCall GetTensorInfoFunction(TypeIdentifier id);
void RegisterTensorInfoFunction(TypeIdentifier id, TensorInfoCall c);

// resize helper function
void TensorVectorResize(
    std::vector<Tensor>& tensors,
    int size,
    DeviceType type);

class CAFFE2_API TensorPrinter {
 public:
  explicit TensorPrinter(
      const std::string& tensor_name = "",
      const std::string& file_name = "",
      int limit = k_limit_default_);
  ~TensorPrinter();

  template <class T>
  void Print(const Tensor& tensor);

  void PrintMeta(const Tensor& tensor);

  string MetaStr(const Tensor& tensor);

 private:
  bool to_file_;
  int limit_;
  std::unique_ptr<std::ofstream> log_file_;
  std::string tensor_name_;
};

template <class T>
void TensorPrinter::Print(const Tensor& tensor) {
  std::stringstream values_stream;
  // One most likely doesn't want to print int64-number of items for visual
  // inspection, so we cast down to int here.
  int total_count = static_cast<int>(std::min(tensor.size(), TIndex(limit_)));
  const T* tensor_data = tensor.template data<T>();
  for (int i = 0; i < total_count - 1; ++i) {
    values_stream << tensor_data[i] << ",";
  }
  // We do not add a comma after the last item.
  values_stream << tensor_data[total_count - 1];
  if (to_file_) {
    (*log_file_) << MetaStr(tensor) << values_stream.str() << std::endl;
  } else {
    // Log to console.
    LOG(INFO) << MetaStr(tensor) << values_stream.str();
  }
}

} // namespace caffe2
#endif // CAFFE2_CORE_TENSOR_H_
