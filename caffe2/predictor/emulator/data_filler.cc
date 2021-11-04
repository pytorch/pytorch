#include "caffe2/predictor/emulator/data_filler.h"
#include "caffe2/predictor/emulator/utils.h"

#include <c10/util/irange.h>

namespace caffe2 {
namespace emulator {

void DataNetFiller::fill_parameter(Workspace* ws) const {
  // As we use initial parameter initialization for this BenchmarkState,
  // we can just run the init_net
  CAFFE_ENFORCE(
      ws->RunNetOnce(init_net_),
      "Failed running the init_net: ",
      ProtoDebugString(init_net_));
}

void DataNetFiller::fill_input_internal(TensorList_t* input_data) const {
  Workspace ws;
  CAFFE_ENFORCE(ws.RunNetOnce(data_net_));
  for (const auto& name : input_names_) {
    input_data->emplace_back(
        BlobGetMutableTensor(ws.GetBlob(name), CPU)->Clone());
  }
}

void fill_with_type(
    const TensorFiller& filler,
    const std::string& type,
    TensorCPU* output) {
  CPUContext context;
  if (type == "float") {
    filler.Fill<float>(output, &context);
  } else if (type == "double") {
    filler.Fill<double>(output, &context);
  } else if (type == "uint8_t" || type == "unsigned char") {
    filler.Fill<uint8_t>(output, &context);
  } else if (type == "uint16_t") {
    filler.Fill<uint16_t>(output, &context);
  } else if (type == "int8_t") {
    filler.Fill<int8_t>(output, &context);
  } else if (type == "int16_t") {
    filler.Fill<int16_t>(output, &context);
  } else if (type == "int32_t" || type == "int") {
    filler.Fill<int32_t>(output, &context);
  } else if (type == "int64_t" || type == "long") {
    filler.Fill<int64_t>(output, &context);
  } else if (type == "bool") {
    auto mutable_filler = filler;
    mutable_filler.Min(0).Max(2).Fill<uint8_t>(output, &context);
  } else {
    throw std::invalid_argument("filler does not support type " + type);
  }
}

DataRandomFiller::DataRandomFiller(
    const NetDef& run_net,
    const std::vector<std::vector<std::vector<int64_t>>>& input_dims,
    const std::vector<std::vector<std::string>>& input_types) {
  // parse dimensions
  CAFFE_ENFORCE_EQ(input_dims.size(), run_net.op_size());
  CAFFE_ENFORCE_EQ(input_types.size(), run_net.op_size());

  // load op inputs and outputs
  std::unordered_set<std::string> output_names;
  for (auto i : c10::irange(run_net.op_size())) {
    const auto& op = run_net.op(i);
    const auto& op_dims = input_dims[i];
    const auto& op_types = input_types[i];
    CAFFE_ENFORCE(
        op_dims.size() == static_cast<size_t>(op.input_size()),
        op.name() + " has " + c10::to_string(op.input_size()) +
            " inputs; while the input dimension size is " +
            c10::to_string(op_dims.size()));
    CAFFE_ENFORCE(
        op_types.size() == static_cast<size_t>(op.input_size()),
        op.name() + " has " + c10::to_string(op.input_size()) +
            " inputs; while the input type size is " +
            c10::to_string(op_types.size()));

    for (auto j : c10::irange(op.input_size())) {
      inputs_[op.input(j)] =
          std::make_pair(get_tensor_filler(op, j, op_dims), op_types[j]);
    }

    // Hack, we normal have a path of
    // length -> LengthsiRangeFill -> Gather -> w -> SparseLengthsWeighted*
    //       \---------------------------------------/
    // So when we generate the value of length, we need to bound it to the size
    // of weight input of Gather too
    if (op.type().find("SparseLengthsWeighted") == 0 && i > 0) {
      // NOLINTNEXTLINE(cppcoreguidelines-narrowing-conversions,bugprone-narrowing-conversions)
      const auto& prev_op = run_net.op(i - 1);
      if (prev_op.type() == "Gather") {
        const auto& prev_dims = input_dims[i - 1];
        VLOG(1) << "Setting max length value to " << prev_dims[0].front()
                << " for " << op.input(3);
        inputs_[op.input(3)].first.Max(prev_dims[0].front());
      }
    }

    for (auto j : c10::irange(op.output_size())) {
      output_names.emplace(op.output(j));
    }
  }

  // load parameters
  std::unordered_set<std::string> parameters;
  for (auto i : c10::irange(run_net.arg_size())) {
    const auto& arg = run_net.arg(i);
    // TODO: replace "PredictorParameters" with the constant in OSS bbp
    if (arg.has_name() && arg.name() == "PredictorParameters") {
      parameters.reserve(arg.strings_size());
      for (auto j : c10::irange(arg.strings_size())) {
        parameters.emplace(arg.strings(j));
      }
      break;
    }
  }
  if (parameters.size() == 0) {
    VLOG(1) << "Fail to find any parameters";
  }
  for (const auto& param : parameters) {
    // remove unused parameters
    if (inputs_.find(param) != inputs_.end()) {
      // inputs_[param] will be erase from inputs_ in the next step
      parameters_.emplace(param, inputs_[param]);
    }
  }

  for (const auto& param : parameters_) {
    inputs_.erase(param.first);
  }
  for (const auto& name : output_names) {
    inputs_.erase(name);
  }
  CAFFE_ENFORCE(inputs_.size() > 0, "Empty input for run net");

  // generate input names
  for (const auto& input : inputs_) {
    input_names_.push_back(input.first);
  }
}

void DataRandomFiller::fill_parameter(Workspace* ws) const {
  for (auto& param : parameters_) {
    Blob* blob = ws->CreateBlob(param.first);
    fill_with_type(
        param.second.first,
        param.second.second,
        BlobGetMutableTensor(blob, CPU));
    CAFFE_ENFORCE(ws->GetBlob(param.first)->GetRaw());
  }
}

void DataRandomFiller::fill_input_internal(TensorList_t* input_data) const {
  for (auto& name : input_names_) {
    input_data->emplace_back(CPU);
    const auto& it = inputs_.find(name);
    CAFFE_ENFORCE(it != inputs_.end());
    fill_with_type(it->second.first, it->second.second, &input_data->back());
  }
}

TestDataRandomFiller::TestDataRandomFiller(
    const NetDef& net,
    const std::vector<std::vector<std::vector<int64_t>>>& inputDims,
    const std::vector<std::vector<std::string>>& inputTypes)
    : DataRandomFiller() {
  std::unordered_set<std::string> outputNames;
  // Determine blobs that are outputs of some ops (intermediate blobs).
  for (auto opIdx = 0; opIdx < net.op_size(); ++opIdx) {
    const auto& op = net.op(opIdx);
    for (auto outputIdx = 0; outputIdx < op.output_size(); ++outputIdx) {
      outputNames.emplace(op.output(outputIdx));
    }
  }
  // Determine ops that have non-intermediate inputs.
  std::unordered_set<size_t> opWithRequiredInputs;
  for (auto opIdx = 0; opIdx < net.op_size(); ++opIdx) {
    const auto& op = net.op(opIdx);
    for (auto inputIdx = 0; inputIdx < op.input_size(); ++inputIdx) {
      if (!outputNames.count(op.input(inputIdx))) {
        opWithRequiredInputs.emplace(opIdx);
        break;
      }
    }
  }

  CAFFE_ENFORCE_EQ(inputDims.size(), opWithRequiredInputs.size());
  CAFFE_ENFORCE_EQ(inputTypes.size(), opWithRequiredInputs.size());

  int counter = 0;
  for (auto opIdx = 0; opIdx < net.op_size(); ++opIdx) {
    if (!opWithRequiredInputs.count(opIdx)) {
      // Skip intermediate ops.
      continue;
    }
    const auto& op = net.op(opIdx);
    const auto& op_dims = inputDims[counter];
    const auto& op_types = inputTypes[counter];
    ++counter;

    int countRequiredInputs = 0;
    for (auto inputIdx = 0; inputIdx < op.input_size(); ++inputIdx) {
      if (!outputNames.count(op.input(inputIdx))) {
        ++countRequiredInputs;
      }
    }

    CAFFE_ENFORCE(
        op_dims.size() == static_cast<unsigned>(countRequiredInputs),
        op.name() + " has " + c10::to_string(op.input_size()) +
            " (required) inputs; while the input dimension size is " +
            c10::to_string(op_dims.size()));
    CAFFE_ENFORCE(
        op_types.size() == static_cast<unsigned>(countRequiredInputs),
        op.name() + " has " + c10::to_string(op.input_size()) +
            " (required) inputs; while the input type size is " +
            c10::to_string(op_types.size()));

    int dimCounter = 0;
    for (auto inputIdx = 0; inputIdx < op.input_size(); ++inputIdx) {
      const auto& inputName = op.input(inputIdx);
      if (outputNames.count(inputName)) {
        // Skip intermediate inputs.
        continue;
      }
      inputs_[inputName] = std::make_pair(
          get_tensor_filler(op, dimCounter, op_dims), op_types[dimCounter]);
      ++dimCounter;
    }
  }
  CAFFE_ENFORCE(inputs_.size() > 0, "Empty input for run net");
  // generate input names
  for (const auto& input : inputs_) {
    input_names_.push_back(input.first);
  }
}

void TestDataRandomFiller::fillInputToWorkspace(Workspace* workspace) const {
  for (auto& name : input_names_) {
    const auto& it = inputs_.find(name);
    CAFFE_ENFORCE(it != inputs_.end());
    auto* tensor =
        BlobGetMutableTensor(workspace->CreateBlob(name), caffe2::CPU);
    fill_with_type(it->second.first, it->second.second, tensor);
  }
}

void fillRandomNetworkInputs(
    const NetDef& net,
    const std::vector<std::vector<std::vector<int64_t>>>& inputDims,
    const std::vector<std::vector<std::string>>& inputTypes,
    Workspace* workspace) {
  TestDataRandomFiller(net, inputDims, inputTypes)
      .fillInputToWorkspace(workspace);
}

} // namespace emulator
} // namespace caffe2
