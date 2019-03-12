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

    def set_state(self, new_state):
        r"""Generator.set_state(new_state) -> void

        Sets the Generator state.

        Arguments:
            new_state (torch.ByteTensor): The desired state.

        Example::

            >>> g_cpu = torch.Generator()
            >>> g_cpu_default = torch.Generator(default=True)
            >>> g_cpu.set_state(g_cpu_default.get_state())
        """
        return super(Generator, self).set_state(new_state)

    def get_state(self):
        r"""Generator.get_state() -> Tensor

        Returns the Generator state as a ``torch.ByteTensor``.

        Returns:
            Tensor: A ``torch.ByteTensor`` which contains all the necessary bits
            to restore a Generator to a specific point in time.

        Example::

            >>> g_cpu = torch.Generator()
            >>> g_cpu_default = torch.Generator(default=True)
            >>> g_cpu.set_state(g_cpu_default.get_state())
        """
        return super(Generator, self).get_state()

    def manual_seed(self, seed):
        r"""Generator.manual_seed(seed) -> Generator

        Sets the seed for generating random numbers. Returns a `torch.Generator` object.
        It is recommended to set a large seed, i.e. a number that has a good balance of 0
        and 1 bits. Avoid having many 0 bits in the seed.

        Arguments:
            seed (int): The desired seed.

        Returns:
            Generator: An ATen Generator object.

        Example::

            >>> g_cpu_default = torch.Generator(default=True)
            >>> g_cpu_default.manual_seed(2147483647)
        """
        return super(Generator, self).manual_seed(seed)

    def initial_seed(self):
        r"""Generator.initial_seed() -> int

        Returns the initial seed for generating random numbers.

        Example::

            >>> g_cpu = torch.Generator()
            >>> g_cpu.initial_seed()
            2147483647
        """
        return super(Generator, self).initial_seed()

    def seed(self):
        r"""Generator.seed() -> int

        Gets a non-deterministic random number from std::random_device or the current
        time and uses it to seed a Generator.

        Example::

            >>> g_cpu = torch.Generator()
            >>> g_cpu.seed()
            1516516984916
        """
        return super(Generator, self).seed()
