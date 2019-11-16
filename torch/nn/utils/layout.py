import torch

def convert_conv2d_weight_memory_layout(module, layout):
    r"""Convert ``memory_format`` of ``nn.Conv2d.weight`` to ``layout``

    The conversion recursively applies to nested ``nn.Module``, including ``module``.
    Note that it only changes the memory_format, but not the semantics of each dimensions.

    This function is used to facilitate the computation to adopt NHWC kernels, which
    provides considerable speed up for fp16 data on CUDA devices with compute capability >= 7.0

    Example:
        >>>  input = torch.randint(1, 10, (2, 8, 4, 4), dtype=torch.float16, device="cuda")
        >>>  model = nn.Sequential(
        >>>      nn.Conv2d(8, 4, 3)).cuda().half()
        >>>  # This is identical to:
        >>>  # nn.utils.convert_conv2d_weight_memory_layout(model, torch.channels_last)
        >>>  model = nn.utils.convert_conv2d_weight_memory_layout(model, torch.channels_last)
        >>>  out = model(input)

    Arguments:
        module (nn.Module): ``nn.Conv2d`` or container ``nn.Module``
        layout: user specified ``memory_layout``,
            e.g. ``torch.channels_last`` or ``torch.contiguous_format``

    Returns:
        The original module with updated ``nn.Conv2d``
    """

    # TODO: expand this to `_ConvNd` when channels_last support is extended
    # beyond only 4d tensors.
    if isinstance(module, torch.nn.Conv2d):
        module.weight.data = module.weight.contiguous(memory_format=layout)
    for child in module.children():
        convert_conv2d_weight_memory_layout(child, layout)
    return module
