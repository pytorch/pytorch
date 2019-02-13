#pragma once
#include "caffe2/core/logging.h"
#include "caffe2/core/operator.h"
#include "caffe2/predictor/predictor.h"
#include "caffe2/utils/filler.h"

namespace caffe2 {
namespace emulator {

typedef caffe2::Predictor::TensorList TensorList_t;

/*
 * A filler to initialize the parameters and inputs of a predictor
 */
class Filler {
 protected:
  virtual void fill_input_internal(TensorList_t* input_data) const = 0;

 public:
  // initialize the workspace with parameter
  virtual void fill_parameter(Workspace* ws) const = 0;

  // generate input data and return input data size
  size_t fill_input(TensorList_t* input_data) const {
    CAFFE_ENFORCE(input_data, "input_data is null");
    input_data->clear();

    fill_input_internal(input_data);

    uint64_t bytes = 0;
    for (const auto& item : *input_data) {
      bytes += item.nbytes();
    }
    CAFFE_ENFORCE(bytes > 0, "input bytes should be positive");

    return bytes;
  }

  const std::vector<std::string>& get_input_names() const {
    CAFFE_ENFORCE(!input_names_.empty(), "input names is not initialized");
    return input_names_;
  }

  virtual ~Filler() noexcept {}

 protected:
  std::vector<std::string> input_names_;
};

/*
 * @init_net: a reader net to generate parameters
 * @data_net: a reader net to generate inputs
 */
class DataNetFiller : public Filler {
 public:
  DataNetFiller(const NetDef&& init_net, const NetDef&& data_net)
      : init_net_(init_net), data_net_(data_net) {
    // The output of the data_net_ will be served as the input
    int op_size = data_net_.op_size();
    for (int i = 0; i < op_size; ++i) {
      OperatorDef op_def = data_net_.op(i);
      // We rely on Fill op to generate inputs
      CAFFE_ENFORCE(op_def.type().find("Fill") != std::string::npos);
      int output_size = op_def.output_size();
      for (int j = 0; j < output_size; ++j) {
        input_names_.push_back(op_def.output(j));
      }
    }
  }

  void fill_input_internal(TensorList_t* input_data) const override;

  void fill_parameter(Workspace* ws) const override;

 private:
  const NetDef init_net_;
  const NetDef data_net_;
};

/*
 * @run_net: the predict net with parameter and input names
 * @input_dims: the input dimentions of all operator inputs of run_net
 * @input_types: the input types of all operator inputs of run_net
 */
class DataRandomFiller : public Filler {
 public:
  DataRandomFiller(
      const NetDef& run_net,
      const std::vector<std::vector<std::vector<int64_t>>>& input_dims,
      const std::vector<std::vector<std::string>>& input_types);

  void fill_input_internal(TensorList_t* input_data) const override;

  void fill_parameter(Workspace* ws) const override;

 protected:
  DataRandomFiller() {}

  TensorFiller get_tensor_filler(
      const OperatorDef& op_def,
      int input_index,
      const std::vector<std::vector<int64_t>>& input_dims) {
    Workspace ws;
    for (size_t i = 0; i < op_def.input_size(); ++i) {
      // CreateOperator requires all input blobs present
      ws.CreateBlob(op_def.input(i));
    }
    CAFFE_ENFORCE(op_def.has_type());
    const OpSchema* schema = caffe2::OpSchemaRegistry::Schema(op_def.type());
    if (schema == nullptr) {
      throw std::invalid_argument(
          op_def.type() + " does not have input fillers");
    }
    auto filler = schema->InputFillers(input_dims)[input_index];
    return filler;
  }

  using filler_type_pair_t = std::pair<TensorFiller, std::string>;
  std::unordered_map<std::string, filler_type_pair_t> parameters_;
  std::unordered_map<std::string, filler_type_pair_t> inputs_;
};

// A DataRandomFiller that is more convenient to use in unit tests.
// Callers just need to supply input dimensions and types for non-intermediate
// blobs.
// It also treats parameters the same way as non-intermediate inputs (no
// handling of parameters separately).
class TestDataRandomFiller : public DataRandomFiller {
 public:
  TestDataRandomFiller(
      const NetDef& net,
      const std::vector<std::vector<std::vector<int64_t>>>& inputDims,
      const std::vector<std::vector<std::string>>& inputTypes);

  // Fill input directly to the workspace.
  void fillInputToWorkspace(Workspace* workspace) const;
};

} // namespace emulator
} // namespace caffe2
