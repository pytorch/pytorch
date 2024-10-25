import torch


"""
tensor_factory_functions defines the list of torch functions that create tensors.
The list is grabbed by searching thru native_functions.yaml by the following
regular expression:

  cat native_functions.yaml | grep 'func:' | grep -v "Tensor.*->" | grep "[-]>.*Tensor"

It's possible that new tensor factory functions are added making this list stale.
Use at your own risk or regenerate the list.
"""
tensor_factory_functions = (
    torch._cudnn_init_dropout_state,
    torch.arange,
    torch.bartlett_window,
    torch.blackman_window,
    torch._empty_affine_quantized,
    torch.empty_strided,
    torch.eye,
    torch.full,
    torch.from_file,
    torch.hann_window,
    torch.hamming_window,
    torch.kaiser_window,
    torch.linspace,
    torch.logspace,
    torch.ones,
    torch.scalar_tensor,
    torch.rand,
    torch.randint,
    torch.randn,
    torch.randperm,
    torch.range,
    torch._efficientzerotensor,
    torch.zeros,
    torch.tril_indices,
    torch.triu_indices,
    # Note: the following functions match the regular expression search above but
    # they are not available in the torch module. Comment out.
    # torch._sparse_coo_tensor_with_dims,
    # torch.fft_fftfreq,
    # torch.fft_rfftfreq,
) + (
    # torch.tensor is special since it's not in native_functions.yaml
    # add it separately
    torch.tensor,
)
