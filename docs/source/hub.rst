torch.hub
===================================
Pytorch Hub is a pre-trained model repository designed to facilitate research reproducibility.

Publishing models
-----------------

Pytorch Hub supports publishing pre-trained models(model definitions and pre-trained weights)
to a github repository by adding a simple ``hubconf.py`` file;

``hubconf.py`` can have multiple entrypoints. Each entrypoint is defined as a python function with
the following signature.

::

    def entrypoint_name(pretrained=False, *args, **kwargs):
        ...

How to implement an entrypoint?
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
Here is a code snippet from pytorch/vision repository, which specifies an entrypoint
for ``resnet18`` model. You can see a full script in
`pytorch/vision repo <https://github.com/pytorch/vision/blob/master/hubconf.py>`_

::

    dependencies = ['torch', 'math']

    def resnet18(pretrained=False, *args, **kwargs):
        """
        Resnet18 model
        pretrained (bool): a recommended kwargs for all entrypoints
        args & kwargs are arguments for the function
        """
        ######## Call the model in the repo ###############
        from torchvision.models.resnet import resnet18 as _resnet18
        model = _resnet18(*args, **kwargs)
        ######## End of call ##############################
        # The following logic is REQUIRED
        if pretrained:
            # For weights saved in local repo
            # model.load_state_dict(<path_to_saved_file>)

            # For weights saved elsewhere
            checkpoint = 'https://download.pytorch.org/models/resnet18-5c106cde.pth'
            model.load_state_dict(model_zoo.load_url(checkpoint, progress=False))
        return model

- ``dependencies`` variable is a **list** of package names required to to run the model.
- Pretrained weights can either be stored local in the github repo, or loadable by
  ``model_zoo.load()``.
- ``pretrained`` controls whether to load the pre-trained weights provided by repo owners.
- ``args`` and ``kwargs`` are passed along to the real callable function.
- Docstring of the function works as a help message, explaining what does the model do and what
  are the allowed arguments.
- Entrypoint function should **ALWAYS** return a model(nn.module).

Important Notice
^^^^^^^^^^^^^^^^

- The published models should be at least in a branch/tag. It can't be a random commit.

Loading models from Hub
-----------------------

Users can load the pre-trained models using ``torch.hub.load()`` API.


.. automodule:: torch.hub
.. autofunction:: load

Here's an example loading ``resnet18`` entrypoint from ``pytorch/vision`` repo.

::

    hub_model = hub.load(
        'pytorch/vision:master', # repo_owner/repo_name:branch
        'resnet18', # entrypoint
        1234, # args for callable [not applicable to resnet]
        pretrained=True) # kwargs for callable

Where are my downloaded model & weights saved?
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

The locations are used in the order of

- hub_dir: user specified path. It can be set in the following ways:
  - Setting the environment variable ``TORCH_HUB_DIR``
  - Calling ``hub.set_dir(<PATH_TO_HUB_DIR>)``
- ``~/.torch/hub``

.. autofunction:: set_dir

Caching logic
^^^^^^^^^^^^^

By default, we don't clean up files after loading it. Hub uses the cache by default if it already exists in ``hub_dir``.

Users can force a reload by calling ``hub.load(..., force_reload=True)``. This will delete
the existing github folder and downloaded weights, reinitialize a fresh download. This is useful
when updates are published to the same branch, users can keep up with the latest release.
