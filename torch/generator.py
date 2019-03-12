import torch


class Generator(torch._C._GeneratorBase):
    r"""Generator(device='cpu', default=False) -> Generator

    Creates and returns a generator object which manages the state of the algorithm that
    produces pseudo random numbers. Used as a keyword argument in many random tensors in
    :ref:`inplace-random-sampling`. Currently only creation of CPU Generator is supported through
    this API.

    Arguments:
        device (:class:`torch.device`, optional): the desired device for the generator.
        default (bool, optional): If using the default CPU/CUDA generator.

    Returns:
        Generator: An ATen Generator object.

    Example::

        >>> g_cpu = torch.Generator()
        >>> g_cpu_default = torch.Generator(default=True);
    """

    def __new__(cls, device='cpu', default=False):
        return super(Generator, cls).__new__(cls, device=device, default=default)
