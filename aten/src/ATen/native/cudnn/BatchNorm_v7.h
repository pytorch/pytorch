namespace at::native {

TORCH_API size_t
_get_cudnn_batch_norm_reserve_space_size(const Tensor& input_t, bool training);

} // namespace at::native
