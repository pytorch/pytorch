
Serialization semantics
=======================

Best practices
--------------

.. _recommend-saving-models:

Recommended approach for saving a model
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

There are two main approaches for serializing and restoring a model.

The first (recommended) saves and loads only the model parameters::

    torch.save(the_model.state_dict(), PATH)

Then later::

    the_model = TheModelClass(*args, **kwargs)
    the_model.load_state_dict(torch.load(PATH))

The second saves and loads the entire model::

    torch.save(the_model, PATH)

Then later::

    the_model = torch.load(PATH)

However in this case, the serialized data is bound to the specific classes
and the exact directory structure used, so it can break in various ways when
used in other projects, or after some serious refactors.
