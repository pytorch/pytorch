#include <torch/extension.h>
#include <torch/csrc/fx/fx2trt/plugins/init_fx2trt_plugins.h>

namespace fx2trt {

PYBIND11_MODULE(plugins, m) {
  m.def("init_fx2trt_plugins", []() {
      return init_fx2trt_plugins();
  });
}
} // namespace fx2trt
