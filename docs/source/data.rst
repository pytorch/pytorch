torch.utils.data
===================================

.. automodule:: torch.utils.data

At the heart of PyTorch data loading utility is the :class:`torch.utils.data.DataLoader`
class. It represents an Python Iterable over a given dataset, with the iterating
strategy specified by the given dataset and constructor arguments.

Dataset Types
-------------

:class:`~torch.utils.data.DataLoader` supports two different types of datasets:

Map-style datasets
^^^^^^^^^^^^^^^^^^^^

A map-style dataset is one that implements the ``__getitem__`` protocol,
and represents a map from indices/keys to data samples.  E.g., such a
dataset, when called ``dataset[idx]`` could read and the ``idx``-th image
and its corresponding label from a folder on the disk.

See :class:`~torch.utils.data.Dataset` for more details.

Iterable-style datasets
^^^^^^^^^^^^^^^^^^^^^^^

An iterable-style dataset is one that implements the ``__iter__`` protocol,
and represents an iterable over data samples.  E.g., such a dataset, when
called ``iter(dataset)``, could return a stream of data reading from a
database, a remote server, or even logs generated in real time.

See :class:`~torch.utils.data.IterableDataset` for more details.

.. note:: When using an :class:`~torch.utils.data.IterableDataset` with
          multiprocessing data loading. The same dataset object is replicated on
          each worker process, and thus the replicas must be configured
          differently to avoid duplicate data. See :class:`~torch.utils.data.IterableDataset`
          documentations for how to achieve this.

Data Loading Modes
------------------

:class:`~torch.utils.data.DataLoader` constructor receives a
:attr:`dataset` object as its first argument. Based on the other provided
arguments, a :class:`~torch.utils.data.DataLoader` operates in one of three
following modes:

* Batched loading from a map-style dataset (default).

  This is the most common case, and corresponds to fetching a minibatch of
  data and collate them into batched Tensors, i.e., Tensors with one dimension
  being the batch dimension (usually the first). Two combinations of arguments
  that starts this mode are:

  * (default) Using arguments :attr:`batch_size`, :attr:`shuffle`,
    :attr:`sampler`, and :attr:`drop_last` to specify the batch indices sampling
    behavior.

    With the default arguments, a :class:`~torch.utils.data.DataLoader`
    loads data as batches of size ``1`` with indices sampled with replacement.

  * Setting argument :attr:`batch_sampler` to a custom sampler returning a list
    of indices at each time, representing the indices for a batch.

  After fetching a list of samples using the indices from sampler, :attr:`collate_fn`
  is used to collate the list of samples into batched Tensors. Users may use
  customized :attr:`collate_fn` to achieve custom batching, e.g., along a
  dimension other than the first.

  The behavior of this mode is roughly equivalent with::

      for indices in batch_sampler:
          yield collate_fn([dataset[i] for i in indices])

* Loading individual elements from a map-style dataset.

* Loading from an iterable-style dataset.

Memory Pinning
^^^^^^^^^^^^^^

Host to GPU copies are much faster when they originate from pinned (page-locked)
memory. See :ref:`cuda-memory-pinning` for more details on when and how to use
pinned memory generally.

For data loading, passing ``pin_memory=True`` to a :class:`~torch.utils.data.DataLoader`
will automatically put the fetched data Tensors in pinned memory, and thus
enables faster data transfer to GPUs.

The default memory pinning logic only recognizes Tensors and maps and iterables
containing Tensors.  By default, if the pinning logic sees a batch that is a
custom type (which will occur if you have a :attr:`collate_fn` that returns a
custom batch type), or if each element of your batch is a custom type, the
pinning logic will not recognize them, and it will return that batch (or those
elements) without pinning the memory.  To enable memory pinning for custom batch
or data types, define a ``pin_memory`` method on your custom type(s).

See the example below.

Example::

    class SimpleCustomBatch:
        def __init__(self, data):
            transposed_data = list(zip(*data))
            self.inp = torch.stack(transposed_data[0], 0)
            self.tgt = torch.stack(transposed_data[1], 0)

        # custom memory pinning method on custom type
        def pin_memory(self):
            self.inp = self.inp.pin_memory()
            self.tgt = self.tgt.pin_memory()
            return self

    def collate_wrapper(batch):
        return SimpleCustomBatch(batch)

    inps = torch.arange(10 * 5, dtype=torch.float32).view(10, 5)
    tgts = torch.arange(10 * 5, dtype=torch.float32).view(10, 5)
    dataset = TensorDataset(inps, tgts)

    loader = DataLoader(dataset, batch_size=2, collate_fn=collate_wrapper,
                        pin_memory=True)

    for batch_ndx, sample in enumerate(loader):
        print(sample.inp.is_pinned())
        print(sample.tgt.is_pinned())


.. autoclass:: DataLoader
.. autoclass:: Dataset
.. autoclass:: IterableDataset
.. autoclass:: TensorDataset
.. autoclass:: ConcatDataset
.. autoclass:: ChainDataset
.. autoclass:: Subset
.. autofunction:: torch.utils.data.get_worker_info
.. autofunction:: torch.utils.data.random_split
.. autoclass:: torch.utils.data.Sampler
.. autoclass:: torch.utils.data.SequentialSampler
.. autoclass:: torch.utils.data.RandomSampler
.. autoclass:: torch.utils.data.SubsetRandomSampler
.. autoclass:: torch.utils.data.WeightedRandomSampler
.. autoclass:: torch.utils.data.BatchSampler
.. autoclass:: torch.utils.data.distributed.DistributedSampler
