#include <torch/extension.h>
#include <torch/torch.h>

using namespace torch::autograd;

class Identity : public Function<Identity> {
 public:
  static torch::Tensor forward(AutogradContext* ctx, torch::Tensor input) {
    return input;
  }

  static tensor_list backward(AutogradContext* ctx, tensor_list grad_outputs) {
    return {grad_outputs[0]};
  }
};

torch::Tensor identity(torch::Tensor input) {
  return Identity::apply(input);
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("identity", &identity, "identity");
}
