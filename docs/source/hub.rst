torch.hub
===================================
Pytorch Hub is a pre-trained model repository designed to facilitate research reproducibility.

Publishing models
-----------------

Pytorch Hub supports publishing pre-trained models(model definitions and pre-trained weights)
to a github repository by adding a simple ``hubconf.py`` file;

``hubconf.py`` can have multiple entrypoints. Each entrypoint is defined as a python function
(example: a pre-trained model you want to publish).

::

    def entrypoint_name(*args, **kwargs):
        # args & kwargs are optional, for models which take positional/keyword arguments.
        ...

How to implement an entrypoint?
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
Here is a code snippet from pytorch/vision repository, which specifies an entrypoint
for ``resnet18`` model. You can see a full script in
`pytorch/vision repo <https://github.com/pytorch/vision/blob/master/hubconf.py>`_

::

    dependencies = ['torch']

    def resnet18(pretrained=False, **kwargs):
        """
        Resnet18 model
        pretrained (bool): kwargs, load pretrained weights into the model
        """
        # Call the model in the repo
        from torchvision.models.resnet import resnet18 as _resnet18
        model = _resnet18(pretrained=pretrained, **kwargs)
        return model


- ``dependencies`` variable is a **list** of package names required to to run the model.
- ``pretrained`` controls whether to load the pre-trained weights provided by repo owners.
- ``args`` and ``kwargs`` are passed along to the real callable function.
- Docstring of the function works as a help message. It explains what does the model do and what
  are the allowed positional/keyword arguments. It's highly recommended to add a few examples here.
- Entrypoint function should **ALWAYS** return a model(nn.module).
- Pretrained weights can either be stored local in the github repo, or loadable by
  ``torch.hub.load_state_dict_from_url()``. In the example above ``torchvision.models.resnet.resnet18``
  handles ``pretrained``, alternatively you can put the following logic in the entrypoint.

::
    if kwargs.get('pretrained', False):
        # For checkpoint saved in local repo
        model.load_state_dict(<path_to_saved_checkpoint>)

        # For checkpoint saved elsewhere
        checkpoint = 'https://download.pytorch.org/models/resnet18-5c106cde.pth'
        model.load_state_dict(torch.hub.load_state_dict_from_url(checkpoint, progress=False))


Important Notice
^^^^^^^^^^^^^^^^

- The published models should be at least in a branch/tag. It can't be a random commit.


Loading models from Hub
-----------------------

Pytorch Hub provides convenient APIs to explore all available models in hub through ``torch.hub.list()``,
show docstring and examples through ``torch.hub.help()`` and load the pre-trained models using ``torch.hub.load()``


.. automodule:: torch.hub

.. autofunction:: list

.. autofunction:: help

.. autofunction:: load

Where are my downloaded models saved?
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

The locations are used in the order of

- hub_dir: user specified path. It can be set in the following ways:
  - Calling ``hub.set_dir(<PATH_TO_HUB_DIR>)``
  - ``$TORCH_HOME/hub``, if environment variable ``TORCH_HOME`` is set.
  - ``$XDG_CACHE_HOME/torch/hub``, if environment variable ``XDG_CACHE_HOME` is set.
  - ``~/.cache/torch/hub``

.. autofunction:: set_dir

Caching logic
^^^^^^^^^^^^^

By default, we don't clean up files after loading it. Hub uses the cache by default if it already exists in ``hub_dir``.

Users can force a reload by calling ``hub.load(..., force_reload=True)``. This will delete
the existing github folder and downloaded weights, reinitialize a fresh download. This is useful
when updates are published to the same branch, users can keep up with the latest release.
