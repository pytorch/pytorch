Patching Batch Norm
===================

What's happening?
-----------------
Batch Norm requires in-place updates to running_mean and running_var of the same size as the input.
Functorch does not support inplace update to a regular tensor that takes in a batched tensor (i.e.
``regular.add_(batched)`` is not allowed). So when vmaping over a batch of inputs to a single module,
we end up with this error

How to fix
----------
All of these options assume that you don't need running stats. If you're using a module this means
that it's assumed you won't use batch norm in evaluation mode. If you have a use case that involves
running batch norm with vmap in evaluation mode, please file an issue

Option 1: Change the BatchNorm
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
If you've built the module yourself, you can change the module to not use running stats. In other
words, anywhere that there's a BatchNorm module, set the ``track_running_stats`` flag to be False

.. code-block:: python

    BatchNorm2d(64, track_running_stats=False)


Option 2: torchvision parameter
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
Some torchvision models, like resnet and regnet, can take in a ``norm_layer`` parameter. These are
often defaulted to be BatchNorm2d if they've been defaulted. Instead you can set it to BatchNorm
that doesn't use running stats

.. code-block:: python

    import torchvision
    from functools import partial
    torchvision.models.resnet18(norm_layer=partial(BatchNorm2d, track_running_stats=False))

Option 3: functorch's patching
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
functorch has added some functionality to allow for quick, in-place patching of the module. If you
have a net that you want to change, you can run ``replace_all_batch_norm_modules_`` to update the
module in-place to not use running stats

.. code-block:: python

    from functorch.experimental import replace_all_batch_norm_modules_
    replace_all_batch_norm_modules_(net)
