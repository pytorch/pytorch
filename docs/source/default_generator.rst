default_generator
==================

.. py:data:: torch.default_generator

    The default random number generator used by PyTorch. This is an instance of :class:`torch.Generator` for the CPU device.

    All functions in the :mod:`torch.random` module and many other random functions in PyTorch use this generator by default. 
    You can manipulate this generator to control the randomness in your program.

    **Example**::

        import torch

        # Get the current state of the default generator
        state = torch.get_rng_state()

        # Set a manual seed for the default generator
        torch.manual_seed(12345)

        # Generate random numbers
        random_tensor = torch.rand(3)

        # Restore the original state
        torch.set_rng_state(state)
