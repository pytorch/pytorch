torch.utils.data
===================================

.. automodule:: torch.utils.data

At the heart of PyTorch data loading utility is the :class:`torch.utils.data.DataLoader`
class.  It represents an Python Iterable over a dataset, with
`the iterating strategy <Data Loading Strategies_>`_ specified by
`the type of the given dataset <Dataset Types_>`_  and the constructor arguments.
Moreover, it supports
`both single- and multi-process data loading <Single- and Multi-process Data Loading_>`_,
as well as automatic `memory pinning <Memory Pinning_>`_.

Dataset Types
-------------

:class:`~torch.utils.data.DataLoader` supports two different types of datasets:

* `map-style datasets <Map-style datasets_>`_

* `iterable-style datasets <Iterable-style datasets_>`_

Map-style datasets
^^^^^^^^^^^^^^^^^^

A map-style dataset is one that implements the :meth:`__getitem__` and
:meth:`__len__` protocols, and represents a map from (possibly non-integral)
indices/keys to data samples. E.g., such a dataset, when called ``dataset[idx]``
could read and the ``idx``-th image and its corresponding label from a folder
on the disk.

.. note::
  :class:`~torch.utils.data.DataLoader` by default constructs a index sampler
  that yields integral indices.  To make it work with a map-style dataset with
  non-integral indices/keys, a custom sampler must be provided.

See :class:`~torch.utils.data.Dataset` for more details.

Iterable-style datasets
^^^^^^^^^^^^^^^^^^^^^^^

An iterable-style dataset is one that implements the :meth:`__iter__` protocol,
and represents an iterable over data samples. This type of datasets is
particularly suitable for cases where random reads are expensive or even
improbable. E.g., such a dataset, when called ``iter(dataset)``, could return a
stream of data reading from a database, a remote server, or even logs generated
in real time.

See :class:`~torch.utils.data.IterableDataset` for more details.

.. note:: When using an :class:`~torch.utils.data.IterableDataset` with
          `multi-process data loading <Multi-process data loading_>`_. The same
          dataset object is replicated on each worker process, and thus the
          replicas must be configured differently to avoid duplicate data. See
          :class:`~torch.utils.data.IterableDataset` documentations for how to
          achieve this.

Samplers
--------

For `map-style datasets <Map-style datasets_>`_, we need a way to specify the
sequence of indices/keys used in data loading.  The :class:`torch.utils.data.Sampler`
classes are created for this purpose. They represent iterable objects over the
indices to map-style datasets.  E.g., in the common case with stochastic
gradient decent (SGD), a :class:`~torch.utils.data.Sampler` could randomly
permute a list of indices and yield each one at a time, or yield a small number
of them for mini-batch SGD.

`Data Loading Strategies`_ section talks about how to use a :class:`~torch.utils.data.Sampler`
with a :class:`~torch.utils.data.DataLoader`.

Data Loading Strategies
-----------------------

:class:`~torch.utils.data.DataLoader` constructor receives a
:attr:`dataset` object as its first argument. Based on the other provided
arguments, a :class:`~torch.utils.data.DataLoader` operates in one of three
following strategies:

* `Batched loading from a map-style dataset`_

* `Loading individual members of a map-style dataset`_

* `Loading from an iterable-style dataset`_

Batched loading from a map-style dataset
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

This is the most common case, and corresponds to fetching a minibatch of
data and collating them into batched Tensors, i.e., Tensors with one dimension
being the batch dimension (usually the first). Two combinations of arguments
that starts this mode are:

* (default) Using arguments :attr:`batch_size`, :attr:`shuffle`,
  :attr:`sampler`, and :attr:`drop_last` to specify the batch indices sampling
  behavior.

  With the default arguments :attr:`batch_size=1`, :attr:`shuffle=False`,
  :attr:`sampler=None` and :attr:`drop_last=False`, :class:`~torch.utils.data.DataLoader`
  loads data as batches of size ``1`` sequentially.

* Setting argument :attr:`batch_sampler` to a custom sampler returning a list
  of indices at each time, representing the indices for a batch.

After fetching a list of samples using the indices from sampler, the function
passed as the :attr:`collate_fn` argument is used to collate the list of samples
into batched Tensors.

The behavior of this mode is roughly equivalent with::

    for indices in batch_sampler:
        yield collate_fn([dataset[i] for i in indices])


Working with :attr:`collate_fn`
"""""""""""""""""""""""""""""""

For instance, if each data sample consists of a 3-channel image and an integral
class label, i.e., each element of the dataset returns a tuple
``(image, class_index)``, the default :attr:`collate_fn` collates a list of such
tuples into a single tuple of a batched image tensor and a batched class label
Tensor. In particular, the default :attr:`collate_fn` has the following
properties:

* It always prepends a new dimension as the batch dimension.

* It automatically converts NumPy arrays and Python numerical values into
  PyTorch Tensors.

* It preserves the data structure, e.g., if each sample is a dictionary, it
  outputs a dictionary with the same set of keys but batched Tensors as values
  (or lists if the values can not be converted into Tensors). Same
  for ``list`` s, ``tuple`` s, ``namedtuple`` s, etc.

Users may use customized :attr:`collate_fn` to achieve custom batching, e.g.,
along a dimension other than the first, or to add support for custom data types.

Loading individual members of a map-style dataset
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Sometimes it is cheaper to directly load batched data (e.g., bulk reads from a
database or a remote server), or your program is designed for working on
individual samples.  In such cases, it's probably better to not use the above
loading strategy (where :attr:`collate_fn` is used to collate the samples), but
let the data loader directly return each member of the :attr:`dataset` object.

To start this mode, simply set ``batch_size=None``.  If :attr:`sampler` is set,
it will be used to sample the indices/keys to :attr:`dataset`. Otherwise, a
sampler will be constructed according to :attr:`shuffle` argument.

Each sample is processed with the function passed as the :attr:`convert_fn`
argument. The default :attr:`convert_fn` simply converts NumPy arrays and
Python numerical values into PyTorch Tensors.

The behavior of this mode is roughly equivalent with::

    for index in sampler:
        yield convert_fn(dataset[index])


Loading from an iterable-style dataset
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

When loading from an iterable-style dataset (i.e., a :class:`~torch.utils.data.IterableDataset`),
there is no  notion of a :attr:`sampler`. The data samples are read from the
iterable generated by the :class:`~torch.utils.data.IterableDataset`.

To start this mode, simply passing an :class:`~torch.utils.data.IterableDataset`
instance as the :attr:`dataset` argument.

Each sample is processed with the function passed as the :attr:`convert_fn`
argument. The default :attr:`convert_fn` simply converts NumPy arrays and
Python numerical values into PyTorch Tensors.

For single-process loading, the behavior of this mode is roughly equivalent
with::

    for data in iter(dataset):
        yield convert_fn(data)


For multi-process loading, each worker process gets a different copy of
:attr:`dataset`, so it is often desired to configure each copy independently to
avoid having duplicate data returned from the workers. See the documentation of
:class:`~torch.utils.data.IterableDataset` for details on this.

Single- and Multi-process Data Loading
--------------------------------------

A :class:`~torch.utils.data.DataLoader` uses single-process data loading by
default.

Within a Python process, the `global interpreter lock <https://wiki.python.org/moin/GlobalInterpreterLock>`_
prevents true fully parallelizing Python code across threads. To avoid blocking
computation code with data loading, PyTorch provides an easy switch to perform
multi-process data loading by simply setting the argument :attr:`num_workers`
to a positive integer.

Single-process data loading (default)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

In this mode, data fetching is done in the same process a
:class:`~torch.utils.data.DataLoader` is initialized.  Therefore, data loading
may block computing.  However, this mode may be preferred when resource(s) used
for sharing data among processes (e.g., shared memory, file descriptors) is
limited.  Additionally, single-process loading often shows more readable error
traces and thus is useful for debugging.


Multi-process data loading
^^^^^^^^^^^^^^^^^^^^^^^^^^

Setting the argument :attr:`num_workers` as a positive integer will
turn on multi-process data loading with the specified number of loader worker
processes.

In this mode, each time an iterator of a :class:`~torch.utils.data.DataLoader`
is created (e.g., when you call ``enumerate(dataloader, 0)``), :attr:`num_workers`
worker processes are created. At this point, the :attr:`dataset`, :attr:`collate_fn`,
:attr:`convert_fn` and :attr:`worker_init_fn` are passed to each
worker, where they are used to initialize, and fetch data. This means that
dataset access together with its  internal IO, transforms
(including :attr:`collate_fn` and :attr:`convert_fn`) runs in the worker process.

For map-style datasets, the main process generates the indices using :attr:`sampler`
and sends them to the workers. So any shuffle randomization is done in the main
process which guides loading by assigning indices to load.

Workers are shut down once the end of the iteration is reached, or when the
iterator becomes garbage collected.

.. warning::
  It is generally not recommended to return CUDA tensors in multi-process
  loading because of many subtleties in using CUDA and sharing CUDA tensors in
  multiprocessing (see :ref:`multiprocessing-cuda-note`). Instead, we recommend
  using `automatic memory pinning <Memory Pinning_>`_ (i.e., setting
  :attr:``pin_memory=True``), which enables fast data transfer to CUDA-enabled
  GPUs.

Platform-specific behaviors
"""""""""""""""""""""""""""

Since workers rely on Python ``multiprocessing``, worker launch behavior is
different on Windows compared to Unix.

* On Unix, ``fork`` is the default ``multiprocessing`` start method. Using ``fork``,
  child workers typically can access the :attr:`dataset` and Python argument
  functions directly through the cloned address space.

* On Windows, ``spawn`` is the default ``multiprocessing`` start method. Using
  ``spawn``, another interpreter is launched which runs your main script,
  followed by the internal worker function that receives the :attr:`dataset`,
  :attr:`collate_fn` and other arguments through `Pickle <https://docs.python.org/3/library/pickle.html>`_
  serialization.

This separate serialization means that you should take two steps to ensure you
are compatible with Windows while using multi-process data loading:

- Wrap most of you main script's code within ``if __name__ == '__main__':`` block,
  to make sure it doesn't run again (most likely generating error) when each worker
  process is launched. You can place your dataset and :class:`~torch.utils.data.DataLoader`
  instance creation logic here, as it doesn't need to be re-executed in workers.

- Make sure that any custom :attr:`collate_fn`, :attr:`convert_fn`, :attr:`worker_init_fn`
  or dataset code is declared as top level definitions, outside of the ``__main__``
  check. This ensures that they are available in workers.
  (this is needed since functions are pickled as references only, not ``bytecode``.)

Randomness in multi-process data loading
""""""""""""""""""""""""""""""""""""""""""

By default, each worker will have its PyTorch seed set to
``base_seed + worker_id``, where ``base_seed`` is a long generated
by main process using its RNG. However, seeds for other libraies
may be duplicated upon initializing workers (w.g., NumPy), causing
each worker to return identical random numbers. (See
:ref:`this section <dataloader-workers-random-seed>` in FAQ.)

In :attr:`worker_init_fn`, you may access the PyTorch seed set for each worker
with either :func:`torch.utils.data.get_worker_info().seed <torch.utils.data.get_worker_info>`
or :func:`torch.initial_seed()`, and use it to seed other libraries before data
loading.

Memory Pinning
--------------

Host to GPU copies are much faster when they originate from pinned (page-locked)
memory. See :ref:`cuda-memory-pinning` for more details on when and how to use
pinned memory generally.

For data loading, passing :attr:``pin_memory=True`` to a
:class:`~torch.utils.data.DataLoader` will automatically put the fetched data
Tensors in pinned memory, and thus enables faster data transfer to CUDA-enabled
GPUs.

The default memory pinning logic only recognizes Tensors and maps and iterables
containing Tensors.  By default, if the pinning logic sees a batch that is a
custom type (which will occur if you have a :attr:`collate_fn` that returns a
custom batch type), or if each element of your batch is a custom type, the
pinning logic will not recognize them, and it will return that batch (or those
elements) without pinning the memory.  To enable memory pinning for custom batch
or data types, define a :meth:`pin_memory` method on your custom type(s).

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
