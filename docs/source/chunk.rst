torch.utils.data.chunk
===================================

.. automodule:: torch.utils.data.chunk

This module provides a Python `IterableDataset` class called
:py:class:`torch.utils.data.chunk.ChunkDatasetWrapper` which wraps
the C++ counterpart as described in the
`PyTorch C++ API documentation <https://pytorch.org/cppdocs/api/library_root.html>`_

.. autoclass:: ChunkDatasetWrapper

Bindings
--------

The following classes were implemented in C++
(`torch/csrc/api/include/torch/data/datasets/chunk.h`) but also exposed
to Python through `pybind11 <https://github.com/pybind/pybind11/>.`

.. py:class:: ChunkDatasetOptions(preloader_count, batch_size, cache_size = 2048)

  Create and return a new ChunkDatasetOptions instance

  :param int preloader_count: Number of preloaders threads
  :param int batch_size: Batch size
  :param int cache_size: Cache size to be preloaded before batching

.. py:class:: Sampler

  Base class for a sampler, which yields an index with which to access a dataset
  This class should be used as extension point only, and it shouldn't be instantiated

.. py:class:: RandomSampler(size)

  A :py:class:`Sampler` that returns random indices

  :param int size: Range of the sampler will be `0...size - 1`

.. py:class:: SequentialSampler(size)

  A :py:class:`Sampler` that returns indices sequentially

  :param int size: Range of the sampler will be `0...size - 1`

.. py:class:: DistributedSampler(size, num_replicas=1, rank=0, allow_duplicates=True)

  A :py:class:`Sampler` that selects a subset of indices to sample from and defines a
  sampling behavior. In a distributed setting, this selects a subset of the
  indices depending on the provided `num_replicas` and `rank` parameters. The
  :py:class:`Sampler` performs a rounding operation based on the `allow_duplicates`
  parameter to decide the local sample count.

  :param int size: Range of the sampler will be `0...size - 1`
  :param int num_replicas: Number of replicas (processes)
  :param int rank: Rank (process ID)
  :param bool allow_duplicates: Drops the last batch if it doesnt have the same size as the other batches

.. py:class:: DistributedRandomSampler(size, num_replicas=1, rank=0, allow_duplicates=True)

  Similar to :py:class:`RandomSampler`, but for a distributed setup

  :param int size: Range of the sampler will be `0...size - 1`
  :param int num_replicas: Number of replicas (processes)
  :param int rank: Rank (process ID)
  :param bool allow_duplicates: Drops the last batch if it doesnt have the same size as the other batches

.. py:class:: DistributedSequentialSampler(size, num_replicas=1, rank=0, allow_duplicates=True)

  Similar to :py:class:`SequentialSampler`, but for a distributed setup

  :param int size: Range of the sampler will be `0...size - 1`
  :param int num_replicas: Number of replicas (processes)
  :param int rank: Rank (process ID)
  :param bool allow_duplicates: Drops the last batch if it doesnt have the same size as the other batches

.. py:class:: SamplerWrapper(sampler, stride=0)

  Implementation of a wrapper around :py:class:`Sampler` classes

  This is needed for generating Python bindings, only.
  On C++ implementation, a single process share the same `ChunkDataset` class among all threads, so sampling is straingthforward.
  On Python implementation, on the other hand, `DataLoader` uses multi-processing instead of multi-threading for parallelism.

  Each Python `Dataloader` is a separate process with a copy of `ChunkDataset` library (including :py:class:`Sampler`).
  To prevent different processes to read the same data in parallel, sampler strides are needed to coordinate workers.
  Each instance of SamplerWrapper must be configured with different strides, so that sampling happens in a round-robin fashion:
  `stride0, stride1, ..., strideN, stride0, stride1, ..., strideN, ...`

  For example, assume 2 Python Dataloader workers reading the same `ChunkDataset`.
  Each worker needs to reset their `ChunkDataset` instance so that one of them reads all even (stride 0)
  batches while the other reads all odd batches (stride 1).

  NOTE: To preserve back compatibility with C++ implementation, which doesn't need stride support due to multi-threading,
  `reset()` method doesn't reset stride setting.

  :param torch.utils.data.chunk.Sampler sampler: Instance of :py:class:`Sampler` to be wrapped
  :param int stride: A unique stride for each sampler to coordinate multiple samplers in different workers

.. py:class:: ChunkDataReaderUint8T

  Extends :py:class:`ChunkDataReader` backed by an array of `uint8_t` as Example type.
  It performs data chunking and reading of entire data chunks

.. py:class:: ChunkDataReaderInt8T

  Extends :py:class:`ChunkDataReader` backed by an array of `int8_t` as Example type.
  It performs data chunking and reading of entire data chunks

.. py:class:: ChunkDataReaderInt16T

  Extends :py:class:`ChunkDataReader` backed by an array of `int16_t` as Example type.
  It performs data chunking and reading of entire data chunks

.. py:class:: ChunkDataReaderInt32T

  Extends :py:class:`ChunkDataReader` backed by an array of `int32_t` as Example type.
  It performs data chunking and reading of entire data chunks

.. py:class:: ChunkDataReaderInt64T

  Extends :py:class:`ChunkDataReader` backed by an array of `int64_t` as Example type.
  It performs data chunking and reading of entire data chunks

.. py:class:: ChunkDataReaderFloat

  Extends :py:class:`ChunkDataReader` backed by an array of `float` as Example type.
  It performs data chunking and reading of entire data chunks

.. py:class:: ChunkDataReaderDouble

  Extends :py:class:`ChunkDataReader` backed by an array of `double` as Example type.
  It performs data chunking and reading of entire data chunks

Getting started
---------------

Using the Python API is very similar to the C++ API documented at
`PyTorch C++ API documentation <https://pytorch.org/cppdocs/api/library_root.html>`_

The only differences are that the Python API requires:
  * Samplers must be wrapped by :py:class:`SamplerWrapper`
    before passing them to :py:class:`ChunkDataReader`.
  * `ChunkDataset` class must be wrapped by :py:class:`ChunkDatasetWrapper`
    before passing it to :py:class:`torch.utils.data.DataLoader`

**Example 1 - Reusing `Example`, `ChunkDataReader` and `ChunkDataset` classes:**

The main steps for a `ChunkDataset` implementation with built-in type `Example` are:

1. **C++ Steps: Check `test/cpp_extensions/extension.cpp` for a full implementation**

  a) [optional] Make `std::vector<BUILT_IN_TYPE>` opaque to Python through `PYBIND11_MAKE_OPAQUE()`

2. **Python Steps: Check `test/test_cpp_extensions.py` for a full implementation**

  a) Instantiate a `Sampler` for the chunk sampler
  b) Instantiate a `SamplerWrapper` for chunk sampler
  c) Instantiate a `Sampler` for the example sampler
  d) Instantiate a `SamplerWrapper` for example sampler
  e) Instantiate a specific `ChunkDataReader`
  f) Instantiate a `ChunkDatasetOptions`
  g) Instantiate a specific `ChunkDataset` implementation
  h) Instantiate a `ChunkDatasetWrapper`
  i) Instantiate a `DataLoader`
  j) [optional] Set `DataLoader`'s :attr:`batch_size=None` to disable `auto collation`
  k) [optional] Set `DataLoader`'s :attr:`pin_memory=True` to pin memory
  l) [optional] If DataLoader's :attr:`num_workers` > 1, implement a worker initialization function and set :attr:`worker_init_fn` on :py:class:`DataLoader` constructor
  m) Iterate over `DataLoader`

*ps: Opaque types prevent memory copy between C++ and Python*

Python snippet::

>>> # Importing modules
>>> from torch.utils.data import DataLoader
>>> import torch.utils.data.chunk as chunk
>>> import torch.utils.cpp_extension
>>> # Dummy parameters
>>> chunk_count=3
>>> batch_size=5
>>> cache_size=100
>>> preloaders=1
>>> # Main ChunkDataset classes
>>> chunk_sampler = chunk.SequentialSampler(size=chunk_count)
>>> example_sampler = chunk.SequentialSampler(size=batch_size)
>>> chunk_sampler_wrapper = chunk.SamplerWrapper(sampler=chunk_sampler)
>>> example_sampler_wrapper = chunk.SamplerWrapper(sampler=example_sampler)
>>> reader = cpp_extension.DummyChunkDataReader()
>>> opt = chunk.ChunkDatasetOptions(preloader_count=preloaders, batch_size=batch_size, cache_size=cache_size)
>>> dummy_chunkdataset = cpp_extension.DummyChunkDataset(chunk_reader=reader, chunk_sampler=chunk_sampler_wrapper,example_sampler=example_sampler_wrapper,options=opt)
>>> trainset = chunk.ChunkDataset(dummy_chunkdataset)
>>> trainset.reset()
>>> # Integrating with Python DataLoader
>>> trainloader = DataLoader(dataset=trainset, num_workers=1)
>>> # Iterating over the ChunkDataset
>>> for i, batch in enumerate(trainloader, 0):
>>>   print('Batch {} is {} examples long'.format(i, len(actual)))


**Example 2 - Defining new `Example`, `ChunkDataReader` and `ChunkDataset` custom classes:**

The next example builds on the previous and requires a
`Pytorch C++ Extension <https://pytorch.org/tutorials/advanced/cpp_extension.html#writing-a-c-extension>`

The main steps for a `ChunkDataset` implementation with custom type `Example` are:

1. **C++ Steps: Check `test/cpp_extensions/extension.cpp` for a full implementation:**

  a) Define `FooExampleType` struct (aka ExampleType)
  b) Define `FooChunkDataReader` class by extending `torch::data::datasets::ChunkDataReader<FooExampleType>`
  c) Bind `FooExampleType` struct
  d) [optional] Make `std::vector<FooExampleType>` opaque to Python through `PYBIND11_MAKE_OPAQUE()`
  e) Bind `std::vector<FooExampleType>` through `py::bind_vector<FooExampleType>()` (aka BatchType)
  f) Bind `FooChunkDataReader` binding through `py::class_<FooChunkDataReader>`
  g) Bind `FooChunkDataset` binding through `py::class_<FooChunkDataReader>`

2. **Python Steps: Check `test_foo_chunkdataset_bindings` for a full implementation:**

  a) Instantiate a `Sampler` for the chunk sampler
  b) Instantiate a `SamplerWrapper` for the chunk sampler in the previous step
  c) Instantiate a `Sampler` for the example sampler
  d) Instantiate a `SamplerWrapper` for the example sampler in the previous step
  e) Instantiate a specific `ChunkDataReader`
  f) Instantiate a `ChunkDatasetOptions`
  g) Instantiate a specific `ChunkDataset` implementation
  h) Instantiate a `ChunkDatasetWrapper`
  i) [optional] If `ChunkDataset::BatchType` doesn't contain tensors, numpy arrays, numbers, dicts or lists, implement a transformation function and set :attr:`transform_fn` on :py:class:`ChunkDatasetWrapper` constructor
  j) Instantiate a `DataLoader`
  k) [optional] Set `DataLoader`'s :attr:`batch_size=None` to disable `auto collation`
  l) [optional] Set `DataLoader`'s :attr:`pin_memory=True` to pin memory
  m) [optional] If DataLoader's :attr:`num_workers` > 1, implement a worker initialization function and set :attr:`worker_init_fn` on :py:class:`DataLoader` constructor
  n) Iterate over `DataLoader`

ps: Opaque types prevent memory copy between C++ and Python

C++ snippet:

.. code-block:: cpp

  #include <torch/extension.h>

  using namespace torch::data::samplers;
  using namespace torch::data::datasets;

  /// Custom example type
  struct FooExampleType {
    FooExampleType(int feature, int label) : feature_(feature), label_(label){};
    int feature_;
    int label_;
  };
  PYBIND11_MAKE_OPAQUE(std::vector<FooExampleType>);

  /// Custom ChunkDataReader
  class FooChunkDataReader
      : public ChunkDataReader<FooExampleType> {
  public:
    using BatchType = ChunkDataReader<FooExampleType>::ChunkType;
    using DataType = ChunkDataReader<FooExampleType>::ExampleType;

    /// Read an entire chunk.
    BatchType read_chunk(size_t chunk_index) override {
      BatchType batch_data;
      int start_index = chunk_index == 0
          ? 0
          : std::accumulate(chunk_sizes, chunk_sizes + chunk_index, 0);

      // Similar to std::iota(), but for BatchType
      for (size_t i = 0; i < chunk_sizes[chunk_index]; ++i, ++start_index) {
        batch_data.emplace_back(start_index, start_index + 1);
      }
      return batch_data;
    }

    // Hard coded for 1 chunk
    size_t chunk_count() override {
      return chunk_count_;
    };

    // Not used
    void reset() override{};

  private:
    const static size_t chunk_count_ = 3;
    size_t chunk_sizes[chunk_count_] = {10, 5, 20};
  };

  PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {

    /// Exposing custom type FooExampleType (example type)
    py::class_<FooExampleType>(
        m,
        "FooExampleType",
        "FooExampleType holds a single example with its feature and label")
        .def_readwrite("feature_", &FooExampleType::feature_, "Feature data")
        .def_readwrite("label_", &FooExampleType::label_, "Label data");

    /// Exposing std::vector<FooExampleType> (batch type)
    py::bind_vector<std::vector<FooExampleType>>(
        m,
        "VectorFooExampleType",
        "VectorFooExampleType holds the custom typed dataset");

    /// Specific ChunkDataReader implementation based on FooExampleType
    py::class_<FooChunkDataReader>(
        m, "FooChunkDataReader", "Dummy chunk data reader for testing the API")
        .def(
            py::init<>(),
            "Create and return a new `FooChunkDataReader` instance")
        .def(
            "read_chunk",
            &FooChunkDataReader::read_chunk,
            "Returns dummy data",
            py::arg("chunk_index"),
            py::return_value_policy::take_ownership)
        .def(
            "chunk_count",
            &FooChunkDataReader::chunk_count,
            "Returns the number of chunks")
        .def("reset", &FooChunkDataReader::reset, "Not used");

    /// Specific ChunkDataset implementation based on FooChunkDataReader
    using FooChunkDataset =
        ChunkDataset<FooChunkDataReader, SamplerWrapper, SamplerWrapper>;
    py::class_<FooChunkDataset>(
        m,
        "FooChunkDataset",
        "A stateful dataset that support hierarchical sampling and prefetching of entire chunks."
        "Unlike regular dataset, chunk dataset require two samplers to operate and keeps internal state."
        "`ChunkSampler` selects, which chunk to load next"
        "`ExampleSampler` determines the order of Examples that are returned in each `get_batch` call")
        .def(
            py::init<
                FooChunkDataReader,
                SamplerWrapper,
                SamplerWrapper,
                ChunkDatasetOptions>(),
            "Create and return a new `FooChunkDataset` instance",
            py::arg("chunk_reader"),
            py::arg("chunk_sampler"),
            py::arg("example_sampler"),
            py::arg("options"))
        .def(
            "get_batch",
            (FooChunkDataset::BatchType(FooChunkDataset::*)(size_t)) &
                FooChunkDataset::get_batch,
            "Returns a batch created from preloaded chunks",
            py::arg("batch_size"),
            py::return_value_policy::take_ownership)
        .def(
            "get_batch",
            (FooChunkDataset::BatchType(FooChunkDataset::*)()) &
                FooChunkDataset::get_batch,
            "Returns a batch created from preloaded chunks",
            py::return_value_policy::take_ownership)
        .def(
            "reset",
            &FooChunkDataset::reset,
            "Resets any internal state and starts the internal prefetching mechanism for the chunk dataset")
        .def("size", &FooChunkDataset::size, "Not used")
        .def(
            "chunk_sampler",
            &FooChunkDataset::chunk_sampler,
            "Returns the reference to chunk sampler."
            "Used mainly in distributed data loading to set the epoch number for the sampler.",
            py::return_value_policy::reference_internal);
  }

Python snippet:

.. code-block:: python

  def transform_fn(batch):
      if batch is not None:
          # Output is a dictionary
          dict = {}

          # Utils to allocate shared memory for tensors
          # We need to know the exact amount of memory to preallocate it
          # in shared memory so the main process can access it quickly
          # Send each tensor individually to shared memory leads to 2x perf hit
          features = []
          features_numel = 0
          features_out = None
          labels = []
          labels_numel = 0
          labels_out = None

          for b in batch:
              # C++ FooExampleType to Pytorch Tensors (zero-copy)
              feature_tensor = torch.from_numpy(numpy.array(b.feature_))
              label_tensor = torch.from_numpy(numpy.array(b.label_))

              # Determining tensor size for using shared memory
              features_numel += feature_tensor.numel()
              features.append(feature_tensor)
              labels_numel += label_tensor.numel()
              labels.append(label_tensor)

          # Allocating shared memory
          features_shared_storage = features[0].storage(
          )._new_shared(features_numel)
          features_out = features[0].new(features_shared_storage)
          labels_shared_storage = labels[0].storage()._new_shared(labels_numel)
          labels_out = labels[0].new(labels_shared_storage)

          # Stacking (copying) tensors into shared memory
          torch.stack(features, out=features_out)
          torch.stack(labels, out=labels_out)

          # Return shared memory on torch.multprocessing queues (zero-copy)
          dict['feature'] = features_out
          dict['label'] = labels_out
          return dict

  def worker_init_fn(worker_id):
    # A recent change on pytorch enabled multithreading by default
    # Dataloader logic requires a single thread, though
    # Until the https://github.com/pytorch/pytorch/issues/19213 is resolved,
    # you have to create an environment variable OMP_NUM_THREADS=1 as a workaround
    torch.set_num_threads(1)
    dataset = torch.utils.data.get_worker_info().dataset
    chunk_sampler = dataset.chunk_sampler()
    chunk_sampler.set_current_stride(stride=worker_id)
    dataset.reset()

  chunk_count = 1
  batch_size = 5
  cache_size = 100
  preloaders = 1
  num_workers = 2
  chunk_sampler = chunk.SequentialSampler(size=chunk_count)
  example_sampler = chunk.SequentialSampler(size=batch_size)
  chunk_sampler_wrapper = chunk.SamplerWrapper(sampler=chunk_sampler)
  example_sampler_wrapper = chunk.SamplerWrapper(sampler=example_sampler)
  reader = cpp_extension.FooChunkDataReader()
  opt = chunk.ChunkDatasetOptions(preloader_count=preloaders, batch_size=batch_size, cache_size=cache_size)

  foo_chunkdataset = cpp_extension.FooChunkDataset(chunk_reader=reader,
                                                   chunk_sampler=chunk_sampler_wrapper,
                                                   example_sampler=example_sampler_wrapper,
                                                   options=opt)

  trainset = chunk.ChunkDatasetWrapper(foo_chunkdataset, transform_fn)
  trainset.reset()
  trainloader = DataLoader(dataset=trainset,
                           num_workers=num_workers,
                           worker_init_fn=worker_init_fn)
  for i, batch in enumerate(trainloader, 0):
    print('Batch {}: {}'.format(i, batch))
