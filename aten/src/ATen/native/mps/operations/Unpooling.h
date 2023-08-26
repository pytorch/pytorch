namespace at::native {

Tensor max_unpooling2d_backward_mps(const Tensor& grad_output,
                                    const Tensor& self,
                                    const Tensor& indices,
                                    IntArrayRef output_size);

Tensor& max_unpooling2d_backward_out_mps(const Tensor& grad_output_,
                                      const Tensor& self_,
                                      const Tensor& indices_,
                                      IntArrayRef output_size,
                                      Tensor& grad_input);
}
