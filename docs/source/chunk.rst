torch.utils.data.chunk
===================================

This module provides a Python `IterableDataset` implementation called
:py:class:`torch.utils.data.chunk.ChunkDatasetWrapper` which wraps
the C++ ChunkDataset API as described in the
`PyTorch C++ API documentation <https://pytorch.org/cppdocs/api/library_root.html>`_.
It also exposes Python bindings for the existing C++ implementation,
so that users can leverage both C++ performance and Python flexibility!

.. automodule:: torch.utils.data.chunk

.. autoclass:: ChunkDatasetWrapper

Getting started
---------------

The ChunkDataset Python API is similar to the C++ API documented at
`PyTorch C++ API documentation <https://pytorch.org/cppdocs/api/library_root.html>`_,
but two main differences:

* To leverage the existing Python DataLoader
  (:py:class:`torch.utils.data.DataLoader`), the C++ `ChunkDataset` class must
  be wrapped by a Python :py:class:`ChunkDatasetWrapper`
* `ChunkDataset` constructor was simplified
  so that samplers are automatically instantiated

The following examples will discuss i overall lines the data flow and show some
snipets to demonstrate how the ChunkDataset Python API can be used.
The first example is intentionally simplistic while the second build on it
and unleash all power the API can provide.

Example 1 - Reading data with known batch type classes:
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

The simplest way to use the Chunk API is reading batches
with C++ standard types as `Example`
(aka `uint8_t`, `int8_t`, `int16_t`, `int32_t`, `int64_t`, `float`, `double`).

To achieve this, the main steps for both C++ and Python are:

**C++ Steps:**

* [optional] Make `std::vector<BUILT_IN_TYPE>` opaque to Python through `PYBIND11_MAKE_OPAQUE()`

Although opaque types prevent memory copy between C++ and Python,
we will not use this feature to keep the example as simple as possible.

**Python Steps:**

* Instantiate `ChunkDataReader`
* Instantiate `ChunkDatasetOptions`
* Instantiate `ChunkDataset`
* Instantiate `ChunkDatasetWrapper`
* Instantiate `DataLoader`
* [optional] Set `DataLoader`'s :attr:`batch_size=None` to disable `auto collation`
* [optional] Set `DataLoader`'s :attr:`pin_memory=True` to pin memory to GPU
* [optional] If DataLoader's :attr:`num_workers` > 1:

  * Implement worker initialization function
  * Set :attr:`worker_init_fn` on :py:class:`DataLoader` constructor

* Iterate over `DataLoader`


**Python snippet:**

*Check `test/test_cpp_extensions.py (test_dummy_chunkdataset_bindings)` for full implementation*::

>>> # Importing modules
>>> from torch.utils.data import DataLoader
>>> import torch.utils.data.chunk as chunk
>>> import torch.utils.cpp_extension
>>> # Dummy parameters
>>> batch_size=5
>>> cache_size=100
>>> preloaders=1
>>> # Main ChunkDataset classes
>>> reader = cpp_extension.DummyChunkDataReader()
>>> opt = chunk.ChunkDatasetOptions(preloader_count=preloaders, batch_size=batch_size, cache_size=cache_size)
>>> dummy_chunkdataset = cpp_extension.DummyChunkDataset(chunk_reader=reader, options=opt, shuffle_chunks=False, shuffle_samples=False)
>>> trainset = chunk.ChunkDataset(dummy_chunkdataset)
>>> trainset.reset()
>>> # Integrating with Python DataLoader
>>> trainloader = DataLoader(dataset=trainset, num_workers=1)
>>> # Iterating over the ChunkDataset
>>> for i, batch in enumerate(trainloader, 0):
>>>   print('Batch {} is {} examples long'.format(i, len(actual)))

Example 2 - Reading data with new `Example` type:
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

The next example builds on the previous and requires
`Pytorch C++ Extension <https://pytorch.org/tutorials/advanced/cpp_extension.html#writing-a-c-extension>`

In more advanced models, the C++ batch type will be more complex
structures that need to be exposed to Python.
In this scenario, the main steps are:

**C++ Steps:**

* Define `FooExampleType` struct (aka ExampleType)
* Define `FooChunkDataReader` class by extending `torch::data::datasets::ChunkDataReader<FooExampleType>`
* Bind `FooExampleType` struct
* [optional] Make `std::vector<FooExampleType>` opaque to Python through `PYBIND11_MAKE_OPAQUE()`
* Bind `std::vector<FooExampleType>` through `py::bind_vector<FooExampleType>()` (aka BatchType)
* Bind `FooChunkDataReader` binding through `bind_chunkdatareader<>()` helper
* Bind `FooChunkDataset` binding through `bind_chunkdatareader<>()` helper

Making opaque types prevent memory copy between C++ and Python as described at
`pybind11 documentation <https://pybind11-rtdtest.readthedocs.io/en/stable/advanced.html#treating-stl-data-structures-as-opaque-objects>`_

**Python Steps:**

* Instantiate a specific `ChunkDataReader`
* Instantiate a `ChunkDatasetOptions`
* Instantiate a specific `ChunkDataset` implementation
* Instantiate a `ChunkDatasetWrapper`
* Instantiate a `DataLoader`
* [optional] If `ChunkDataset::BatchType` doesn't contain tensors, numpy arrays, numbers, dicts or lists:

  * Implement a collate function
  * Set :attr:`collate_fn` on :py:class:`DataLoader` constructor

* [optional] Set `DataLoader`'s :attr:`batch_size=None` to disable `auto collation`
* [optional] Set `DataLoader`'s :attr:`pin_memory=True` to pin memory
* [optional] If DataLoader's :attr:`num_workers` > 1:

  * Implement a worker initialization function
  * Set :attr:`worker_init_fn` on :py:class:`DataLoader` constructor

* Iterate over `DataLoader`

**C++ snippet:**

*Check `test/cpp_extensions/extension.cpp` for full implementation*

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
    auto foo_reader = bind_chunkdatareader<FooChunkDataReader>(m, "FooChunkDataReader");
    foo_reader.def(py::init<>(), "Create and return a new `FooChunkDataReader` instance");

    /// Specific ChunkDataset implementation based on FooChunkDataReader
    using FooChunkDataset = ChunkDataset<FooChunkDataReader, SamplerWrapper, SamplerWrapper>;
    bind_chunkdatareader<FooChunkDataReader>(m, "FooChunkDataset");
  }

**Python snippet:**

*Check `test/test_cpp_extensions.py (test_foo_chunkdataset_bindings)` for full implementation*

.. code-block:: python

  def collate_fn(batch):
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
    dataset.set_chunk_sampler_stride(worker_id)
    dataset.reset()

  batch_size = 5
  cache_size = 100
  preloaders = 1
  num_workers = 2
  reader = cpp_extension.FooChunkDataReader()
  opt = chunk.ChunkDatasetOptions(preloader_count=preloaders, batch_size=batch_size, cache_size=cache_size)

  foo_chunkdataset = cpp_extension.FooChunkDataset(chunk_reader=reader,
                                                   options=opt,
                                                   shuffle_chunks=False,
                                                   shuffle_samples=False)

  trainset = chunk.ChunkDatasetWrapper(foo_chunkdataset)
  trainset.reset()
  trainloader = DataLoader(dataset=trainset,
                           num_workers=num_workers,
                           collate_fn=collate_fn,
                           worker_init_fn=worker_init_fn)
  for i, batch in enumerate(trainloader, 0):
    print('Batch {}: {}'.format(i, batch))

Python bindings
---------------

The following classes were implemented in C++
(`torch/csrc/api/include/torch/data/datasets/chunk.h`) but also exposed
to Python through `pybind11 <https://github.com/pybind/pybind11/>.`

.. py:class:: ChunkDatasetOptions(preloader_count, batch_size, cache_size = 2048)

  Create and return a new ChunkDatasetOptions instance

  :param int preloader_count: Number of preloaders threads
  :param int batch_size: Batch size
  :param int cache_size: Cache size to be preloaded before batching

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