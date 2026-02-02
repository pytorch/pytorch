Samplers
========

Samplers control the order in which samples are accessed from a dataset.
They determine the indices that the DataLoader uses to fetch data.

Sampler Base Class
------------------

.. doxygenclass:: torch::data::samplers::Sampler
   :members:
   :undoc-members:

Sequential Sampler
------------------

Accesses samples in order from 0 to N-1. Use this for evaluation or when
order matters.

.. doxygenclass:: torch::data::samplers::SequentialSampler
   :members:
   :undoc-members:

Random Sampler
--------------

Accesses samples in random order. Use this for training to ensure the model
sees samples in different orders each epoch.

.. doxygenclass:: torch::data::samplers::RandomSampler
   :members:
   :undoc-members:

Distributed Random Sampler
--------------------------

For distributed training, ensures each process gets a different subset of
the data without overlap.

.. doxygenclass:: torch::data::samplers::DistributedRandomSampler
   :members:
   :undoc-members:
