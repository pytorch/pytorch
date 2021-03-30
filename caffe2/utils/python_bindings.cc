#include "caffe2/utils/signal_handler.h"
#include <torch/csrc/utils/pybind.h>

namespace py = pybind11;

namespace caffe2 {

PYBIND11_MODULE(python_bindings, m) {
  m.def("set_print_stack_traces_on_fatal_signal",
    &caffe2::setPrintStackTracesOnFatalSignal);
}

} // namespace caffe2
