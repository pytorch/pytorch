#include "smart_tensor_printer.h"

#include "caffe2/core/operator.h"

namespace caffe2 {

namespace {

// Since DispatchHelper doesn't support passing arguments through the call()
// method to DoRunWithType we have to create an object that will hold these
// arguments explicitly.
struct ProxyPrinter {
  template <typename T>
  bool DoRunWithType() {
    tensorPrinter->Print<T>(*tensor);
    return true;
  }

  void Print() {
    // Pulled in printable types from caffe2/core/types.cc
    // Unfortunately right now one has to add them by hand
    DispatchHelper<TensorTypes<
        float,
        int,
        std::string,
        bool,
        uint8_t,
        int8_t,
        uint16_t,
        int16_t,
        int64_t,
        double,
        char>>::call(this, tensor->meta());
  }

  const Tensor<CPUContext>* tensor;
  TensorPrinter* tensorPrinter;
};
}

SmartTensorPrinter::SmartTensorPrinter(const std::string& tensor_name)
    : tensorPrinter_(tensor_name) {}

SmartTensorPrinter::SmartTensorPrinter(
    const std::string& tensor_name,
    const std::string& file_name)
    : tensorPrinter_(tensor_name, file_name) {}

SmartTensorPrinter::SmartTensorPrinter(
    const std::string& tensor_name,
    const std::string& file_name,
    int limit)
    : tensorPrinter_(tensor_name, file_name, limit) {}

void SmartTensorPrinter::Print(const Tensor<CPUContext>& tensor) {
  ProxyPrinter printer;

  printer.tensor = &tensor;
  printer.tensorPrinter = &tensorPrinter_;
  printer.Print();
}

SmartTensorPrinter& SmartTensorPrinter::DefaultTensorPrinter() {
// TODO(janusz): thread_local does not work under mac.
#if __APPLE__
  CAFFE_THROW(
      "SmartTensorPrinter does not work on mac yet due to thread_local.");
#else
  static thread_local SmartTensorPrinter printer;
  return printer;
#endif
}

void SmartTensorPrinter::PrintTensor(const Tensor<CPUContext>& tensor) {
  DefaultTensorPrinter().Print(tensor);
}
}
