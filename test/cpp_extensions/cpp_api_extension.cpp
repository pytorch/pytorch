#include <torch/extension.h>

struct Net : torch::nn::Module {
  Net(int64_t in, int64_t out)
      : fc(in, out),
        bn(torch::nn::BatchNormOptions(out).stateful(true)),
        dropout(0.5) {
    register_module("fc", fc);
    register_module("bn", bn);
    register_module("dropout", dropout);
  }

  torch::Tensor forward(torch::Tensor x) {
    return dropout->forward(bn->forward(torch::relu(fc->forward(x))));
  }

  void set_bias(torch::Tensor bias) {
    fc->bias = bias;
  }

  torch::Tensor get_bias() const {
    return fc->bias;
  }

  torch::nn::Linear fc;
  torch::nn::BatchNorm bn;
  torch::nn::Dropout dropout;
};

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  torch::python::bind_module<Net>(m, "Net")
      .def(py::init<int64_t, int64_t>())
      .def("forward", &Net::forward)
      .def("set_bias", &Net::set_bias)
      .def("get_bias", &Net::get_bias);
}
