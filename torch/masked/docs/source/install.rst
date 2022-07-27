
Installation
============

pip
~~~

To install MaskedTensor via pip, use the following command:

::

    pip install maskedtensor-nightly

Note that MaskedTensor requires PyTorch >= 1.12 (stable version).

Verification
~~~~~~~~~~~~

To ensure that the MaskedTensor installation worked, we can verify with some simple code like:

::

    import torch
    from maskedtensor import masked_tensor

    data = torch.tensor([[1,2,3], [4,5,6]], dtype=torch.float)
    mask = torch.tensor([[True, False, True], [False, False, True]])
    mt = masked_tensor(data, mask)

and the masked tensor output should look like:

::

    masked_tensor(
    [
        [  1.0000,       --,   3.0000],
        [      --,       --,   6.0000]
    ]
    )
