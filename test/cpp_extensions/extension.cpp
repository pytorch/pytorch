#include <vector>

#include <test/cpp/api/dataloader.h>
#include <torch/extension.h>

torch::Tensor sigmoid_add(torch::Tensor x, torch::Tensor y) {
  return x.sigmoid() + y.sigmoid();
}

struct MatrixMultiplier {
  MatrixMultiplier(int A, int B) {
    tensor_ =
        torch::ones({A, B}, torch::dtype(torch::kFloat64).requires_grad(true));
  }
  torch::Tensor forward(torch::Tensor weights) {
    return tensor_.mm(weights);
  }
  torch::Tensor get() const {
    return tensor_;
  }

 private:
  torch::Tensor tensor_;
};

bool function_taking_optional(c10::optional<torch::Tensor> tensor) {
  return tensor.has_value();
}

///
/// BEGIN: ChunkDataset API example, for custom dataset types
///

/// Custom example type
struct FooExampleType {
  FooExampleType(int feature, int label) : feature_(feature), label_(label){};
  int feature_;
  int label_;
};
PYBIND11_MAKE_OPAQUE(std::vector<FooExampleType>);

/// Custom ChunkDataReader
class FooChunkDataReader
    : public torch::data::datasets::ChunkDataReader<FooExampleType> {
 public:
  using BatchType = datasets::ChunkDataReader<FooExampleType>::ChunkType;
  using DataType = datasets::ChunkDataReader<FooExampleType>::ExampleType;

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

///
/// END: ChunkDataset API example, for custom dataset types
///

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("sigmoid_add", &sigmoid_add, "sigmoid(x) + sigmoid(y)");
  m.def(
      "function_taking_optional",
      &function_taking_optional,
      "function_taking_optional");
  py::class_<MatrixMultiplier>(m, "MatrixMultiplier")
      .def(py::init<int, int>())
      .def("forward", &MatrixMultiplier::forward)
      .def("get", &MatrixMultiplier::get);

  ///
  /// BEGIN: ChunkDataset API example, for custom dataset types
  ///

  /// ChunkDataset API binding for testing
  using namespace torch::data::samplers;
  using namespace torch::data::datasets;

  /// Specific ChunkDataReader implementation
  py::class_<DummyChunkDataReader>(
      m, "DummyChunkDataReader", "Dummy chunk data reader for testing the API")
      .def(
          py::init<>(),
          "Create and return a new `DummyChunkDataReader` instance")
      .def(
          "read_chunk",
          &DummyChunkDataReader::read_chunk,
          "Returns dummy data",
          py::arg("chunk_index"),
          py::return_value_policy::take_ownership)
      .def(
          "chunk_count",
          &DummyChunkDataReader::chunk_count,
          "Returns the number of chunks")
      .def("reset", &DummyChunkDataReader::reset, "Not used");

  /// Specific ChunkDataset implementation
  using DummyChunkDataset =
      ChunkDataset<DummyChunkDataReader, SamplerWrapper, SamplerWrapper>;
  py::class_<DummyChunkDataset>(
      m,
      "DummyChunkDataset",
      "A stateful dataset that support hierarchical sampling and prefetching of entire chunks."
      "Unlike regular dataset, chunk dataset require two samplers to operate and keeps internal state."
      "`ChunkSampler` selects, which chunk to load next"
      "`ExampleSampler` determines the order of Examples that are returned in each `get_batch` call")
      .def(
          py::init<
              DummyChunkDataReader,
              SamplerWrapper,
              SamplerWrapper,
              ChunkDatasetOptions>(),
          "Create and return a new `DummyChunkDataset` instance",
          py::arg("chunk_reader"),
          py::arg("chunk_sampler"),
          py::arg("example_sampler"),
          py::arg("options"))
      .def(
          "get_batch",
          (DummyChunkDataset::BatchType(DummyChunkDataset::*)(size_t)) &
              DummyChunkDataset::get_batch,
          "Returns a batch created from preloaded chunks",
          py::arg("batch_size"),
          py::return_value_policy::take_ownership)
      .def(
          "get_batch",
          (DummyChunkDataset::BatchType(DummyChunkDataset::*)()) &
              DummyChunkDataset::get_batch,
          "Returns a batch created from preloaded chunks",
          py::return_value_policy::take_ownership)
      .def(
          "reset",
          &DummyChunkDataset::reset,
          "Resets any internal state and starts the internal prefetching mechanism for the chunk dataset")
      .def("size", &DummyChunkDataset::size, "Not used")
      .def(
          "chunk_sampler",
          &DummyChunkDataset::chunk_sampler,
          "Returns the reference to chunk sampler."
          "Used mainly in distributed data loading to set the epoch number for the sampler.",
          py::return_value_policy::reference_internal);

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

  ///
  /// END: ChunkDataset API example, for custom dataset types
  ///
}
