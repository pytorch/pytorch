#include <torch/extension.h>

struct Net : torch::nn::Module {
  Net(int64_t in, int64_t out) : fc(in, out) {
    register_module("fc", fc);
    buffer = register_buffer("buf", torch::eye(5));
  }

  torch::Tensor forward(torch::Tensor x) {
    return fc->forward(x);
  }

  void set_bias(torch::Tensor bias) {
    torch::NoGradGuard guard;
    fc->bias.set_(bias);
  }

  torch::Tensor get_bias() const {
    return fc->bias;
  }

  torch::nn::Linear fc;
  torch::Tensor buffer;
};

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  torch::python::bind_module<Net>(m, "Net")
      .def(py::init<int64_t, int64_t>())
      .def("set_bias", &Net::set_bias)
      .def("get_bias", &Net::get_bias);
}
