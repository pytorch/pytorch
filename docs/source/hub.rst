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
Here is a code snippet specifies an entrypoint for ``resnet18`` model if we expand
the implementation in ``pytorch/vision/hubconf.conf``.
In most case importing the right function in ``hubconf.conf`` is sufficient. Here we
just want to use the expanded version as an example to show how it works.
You can see the full script in
`pytorch/vision repo <https://github.com/pytorch/vision/blob/master/hubconf.py>`_

::

    dependencies = ['torch']
    from torchvision.models.resnet import resnet18 as _resnet18

    # resnet18 is the name of entrypoint
    def resnet18(pretrained=False, **kwargs):
        """ # This docstring shows up in hub.help()
        Resnet18 model
        pretrained (bool): kwargs, load pretrained weights into the model
        """
        # Call the model, load pretrained weights
        model = _resnet18(pretrained=pretrained, **kwargs)
        return model


- ``dependencies`` variable is a **list** of package names required to to run the model.
- ``args`` and ``kwargs`` are passed along to the real callable function.
- Docstring of the function works as a help message. It explains what does the model do and what
  are the allowed positional/keyword arguments. It's highly recommended to add a few examples here.
- Entrypoint function should **ALWAYS** return a model(nn.module).
- Pretrained weights can either be stored locally in the github repo, or loadable by
  ``torch.hub.load_state_dict_from_url()``. In the example above ``torchvision.models.resnet.resnet18``
  handles ``pretrained``, alternatively you can put the following logic in the entrypoint definition.

::

    if pretrained:
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

Running a loaded model:
^^^^^^^^^^^^^^^^^^^^^^^

Note that ``*args, **kwargs`` in ``torch.load()`` are used to **instantiate** a model.
After you loaded a model, how can you find out what you can do with the model?
A suggested workflow is

- ``dir(model)`` to see all avaialble methods of the model.
- ``help(model.foo)`` to check what arguments ``model.foo`` takes to run

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


Known limitations:
^^^^^^^^^^^^^^^^^^
Torch hub works by importing the package as if it was installed. There're some side effects
introduced by importing in Python. For example, you can see new items in Python caches
``sys.modules`` and ``sys.path_importer_cache`` which is normal Python behavior.

A known limitation that worth mentioning here is user **CANNOT** load two different branches of
the same repo in the **same python process**. It's just like installing two packages with the
same name in Python, which is not good. Cache might join the party and give you surprises if you
actually try that. Of course it's totally fine to load them in separate processes.
