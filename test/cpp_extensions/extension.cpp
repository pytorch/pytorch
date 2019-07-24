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
  auto dummy_reader =
      torch::data::datasets::bind_chunkdatareader<DummyChunkDataReader>(
          m, "DummyChunkDataReader");
  dummy_reader.def(
      py::init<>(), "Create and return a new `DummyChunkDataReader` instance");

  /// Specific ChunkDataset implementation
  torch::data::datasets::bind_chunkdataset<DummyChunkDataReader>(
      m, "DummyChunkDataset");

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

  /// FooChunkDataReader bindings
  auto foo_reader =
      torch::data::datasets::bind_chunkdatareader<FooChunkDataReader>(
          m, "FooChunkDataReader");
  foo_reader.def(
      py::init<>(), "Create and return a new `FooChunkDataReader` instance");

  /// FooChunkDataset bindings
  torch::data::datasets::bind_chunkdataset<FooChunkDataReader>(
      m, "FooChunkDataset");

  ///
  /// END: ChunkDataset API example, for custom dataset types
  ///
}
