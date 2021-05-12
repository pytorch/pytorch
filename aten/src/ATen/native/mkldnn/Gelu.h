namespace at { namespace native {

Tensor _mkldnn_gelu(const Tensor& input, const Tensor& result);

Tensor _mkldnn_gelu_backward(const Tensor& grad_output, const Tensor& input, const Tensor& grad_input);

}
}