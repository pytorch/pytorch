namespace at::native {

TORCH_API size_t
_get_cudnn_batch_norm_reserve_space_size(const Tensor& input_t, bool training);

#if AT_CUDNN_ENABLED()

// v7 functions for runtime switching (similar to Conv)
std::tuple<Tensor, Tensor, Tensor, Tensor> cudnn_batch_norm_v7(
    const Tensor& input,
    const Tensor& weight,
    const std::optional<Tensor>& bias_opt,
    const std::optional<Tensor>& running_mean_opt,
    const std::optional<Tensor>& running_var_opt,
    bool training,
    double exponential_average_factor,
    double epsilon);

std::tuple<Tensor&, Tensor&, Tensor&, Tensor&> cudnn_batch_norm_out_v7(
    const Tensor& input,
    const Tensor& weight,
    const std::optional<Tensor>& bias,
    const std::optional<Tensor>& running_mean,
    const std::optional<Tensor>& running_var,
    bool training,
    double exponential_average_factor,
    double epsilon,
    Tensor& out,
    Tensor& save_mean,
    Tensor& save_var,
    Tensor& reserve);

std::tuple<Tensor, Tensor, Tensor> cudnn_batch_norm_backward_v7(
    const Tensor& input,
    const Tensor& grad_output,
    const Tensor& weight,
    const std::optional<Tensor>& running_mean_opt,
    const std::optional<Tensor>& running_var_opt,
    const std::optional<Tensor>& save_mean_opt,
    const std::optional<Tensor>& save_var_opt,
    double epsilon,
    const Tensor& reservedSpace);

size_t _get_cudnn_batch_norm_reserve_space_size_v7(
    const Tensor& input_t,
    bool training);

#endif

} // namespace at::native
