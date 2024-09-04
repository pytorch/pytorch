Patching Batch Norm
===================

What's happening?
-----------------
Batch Norm requires in-place updates to running_mean and running_var of the same size as the input.
Functorch does not support inplace update to a regular tensor that takes in a batched tensor (i.e.
``regular.add_(batched)`` is not allowed). So when vmapping over a batch of inputs to a single module,
we end up with this error

How to fix
----------
One of the best supported ways is to switch BatchNorm for GroupNorm. Options 1 and 2 support this

All of these options assume that you don't need running stats. If you're using a module this means
that it's assumed you won't use batch norm in evaluation mode. If you have a use case that involves
running batch norm with vmap in evaluation mode, please file an issue

Option 1: Change the BatchNorm
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
If you want to change for GroupNorm, anywhere that you have BatchNorm, replace it with:

.. code-block:: python

    BatchNorm2d(C, G, track_running_stats=False)

Here ``C`` is the same ``C`` as in the original BatchNorm. ``G`` is the number of groups to
break ``C`` into. As such, ``C % G == 0`` and as a fallback, you can set ``C == G``, meaning
each channel will be treated separately.

If you must use BatchNorm and you've built the module yourself, you can change the module to
not use running stats. In other words, anywhere that there's a BatchNorm module, set the
``track_running_stats`` flag to be False

.. code-block:: python

    BatchNorm2d(64, track_running_stats=False)


Option 2: torchvision parameter
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
Some torchvision models, like resnet and regnet, can take in a ``norm_layer`` parameter. These are
often defaulted to be BatchNorm2d if they've been defaulted.

Instead you can set it to be GroupNorm.

.. code-block:: python

    import torchvision
    from functools import partial
    torchvision.models.resnet18(norm_layer=lambda c: GroupNorm(num_groups=g, c))

Here, once again, ``c % g == 0`` so as a fallback, set ``g = c``.

If you are attached to BatchNorm, be sure to use a version that doesn't use running stats

.. code-block:: python

    import torchvision
    from functools import partial
    torchvision.models.resnet18(norm_layer=partial(BatchNorm2d, track_running_stats=False))

Option 3: functorch's patching
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
functorch has added some functionality to allow for quick, in-place patching of the module to not
use running stats. Changing the norm layer is more fragile, so we have not offered that. If you
have a net where you want the BatchNorm to not use running stats, you can run
``replace_all_batch_norm_modules_`` to update the module in-place to not use running stats

.. code-block:: python

    from torch.func import replace_all_batch_norm_modules_
    replace_all_batch_norm_modules_(net)

Option 4: eval mode
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
When run under eval mode, the running_mean and running_var will not be updated. Therefore, vmap can support this mode

.. code-block:: python

    model.eval()
    vmap(model)(x)
    model.train()
