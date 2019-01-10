#include "caffe2/core/operator.h"
#include "caffe2/core/tensor.h"
#include "caffe2/core/types.h"
#include "caffe2/utils/proto_utils.h"

#include "nnapi.h"

namespace {
// Bug: ANEURALNETWORKS_UNMAPPABLE and ANEURALNETWORKS_OP_FAILED share the same
// enum value
void reportError(int result_code) {
  switch (result_code) {
    case ANEURALNETWORKS_NO_ERROR:
      break;
    case ANEURALNETWORKS_OUT_OF_MEMORY:
      CAFFE_THROW("out of memory");
    case ANEURALNETWORKS_INCOMPLETE:
      CAFFE_THROW("incomplete");
    case ANEURALNETWORKS_UNEXPECTED_NULL:
      CAFFE_THROW("unexpected null");
    case ANEURALNETWORKS_BAD_DATA:
      CAFFE_THROW("bad data");
    case ANEURALNETWORKS_OP_FAILED:
      CAFFE_THROW("op failed or unmappable");
    case ANEURALNETWORKS_BAD_STATE:
      CAFFE_THROW("bad state");
    default:
      CAFFE_THROW("unknown error");
  }
}
} // namespace

namespace caffe2 {

bool NNApi::loadNNApiLibrary() {
  return dlnnapi_load(&libnnapi_, DLNNAPI_FLAG_VERSION_27);
}

NNApi::~NNApi() {
  if (run_end_) {
    libnnapi_.ANeuralNetworksEvent_free(run_end_);
  }
  if (run_) {
    libnnapi_.ANeuralNetworksExecution_free(run_);
  }
  if (compilation_) {
    libnnapi_.ANeuralNetworksCompilation_free(compilation_);
  }
  if (model_) {
    libnnapi_.ANeuralNetworksModel_free(model_);
  }
}

bool NNApi::run(const TensorVector& inputs, TensorVector* outputs) {
  CAFFE_ENFORCE(inputs.size() <= run_net_.external_input_size());
  try {
    init(inputs, outputs);
  } catch (const std::exception& e) {
    LOG(ERROR) << "Error duing model initialization: " << e.what();
    return false;
  }

  try {
    VLOG(1) << "Start compute";
    int result_code =
        libnnapi_.ANeuralNetworksExecution_startCompute(run_, &run_end_);
    if (result_code != ANEURALNETWORKS_NO_ERROR) {
      reportError(result_code);
    }
    result_code = libnnapi_.ANeuralNetworksEvent_wait(run_end_);
    if (result_code != ANEURALNETWORKS_NO_ERROR) {
      reportError(result_code);
    }
    VLOG(1) << "Finish compute";
  } catch (const std::exception& e) {
    LOG(ERROR) << "Error during model run: " << e.what();
    return false;
  }
  return true;
}

void NNApi::getConvPoolArgs(const ArgumentHelper& helper, ConvPoolArgs& args) {
  std::vector<int> kernel(helper.GetRepeatedArgument<int>("kernels"));
  std::vector<int> stride(helper.GetRepeatedArgument<int>("strides"));
  std::vector<int> pads(helper.GetRepeatedArgument<int>("pads"));

  // Get old arguments values
  if (helper.HasArgument("kernel")) {
    kernel.resize(2, helper.GetSingleArgument<int>("kernel", 0));
  } else if (helper.HasArgument("kernelh") && helper.HasArgument("kernelw")) {
    kernel.push_back(helper.GetSingleArgument<int>("kernelh", 0));
    kernel.push_back(helper.GetSingleArgument<int>("kernelw", 0));
  }

  if (helper.HasArgument("stride")) {
    stride.resize(2, helper.GetSingleArgument<int>("stride", 0));
  } else if (helper.HasArgument("stride_h") && helper.HasArgument("stride_w")) {
    stride.push_back(helper.GetSingleArgument<int>("stride_h", 0));
    stride.push_back(helper.GetSingleArgument<int>("stride_w", 0));
  }

  if (helper.HasArgument("pad")) {
    pads.resize(4, helper.GetSingleArgument<int>("pad", 0));
  } else if (
      helper.HasArgument("pad_t") && helper.HasArgument("pad_l") &&
      helper.HasArgument("pad_b") && helper.HasArgument("pad_r")) {
    pads.push_back(helper.GetSingleArgument<int>("pad_t", 0));
    pads.push_back(helper.GetSingleArgument<int>("pad_l", 0));
    pads.push_back(helper.GetSingleArgument<int>("pad_b", 0));
    pads.push_back(helper.GetSingleArgument<int>("pad_r", 0));
  }

  // Commit values
  args.kernel_h = kernel.size() > 0 ? kernel[0] : 1;
  args.kernel_w = kernel.size() > 1 ? kernel[1] : args.kernel_h;
  args.stride_x = stride.size() > 0 ? stride[0] : 1;
  args.stride_y = stride.size() > 1 ? stride[1] : 1;
  args.pad_t = pads.size() > 0 ? pads[0] : 0;
  args.pad_l = pads.size() > 1 ? pads[1] : 0;
  args.pad_b = pads.size() > 2 ? pads[2] : 0;
  args.pad_r = pads.size() > 3 ? pads[3] : 0;
}

void NNApi::addPooling(
    const OperatorDef& op,
    OperationCode op_code,
    bool fuse_relu)
// clang-format off
{
  // clang-format on
  VLOG(1) << "Add AveragePool to NN model";
  CAFFE_ENFORCE_EQ(op.input_size(), 1);
  CAFFE_ENFORCE_EQ(op.output_size(), 1);
  ArgumentHelper helper(op);
  StorageOrder order = StringToStorageOrder(
      helper.GetSingleArgument<std::string>("order", "NCHW"));
  if (order == NCHW) {
    CAFFE_THROW("NN API supports NHWC only");
  }

  ConvPoolArgs args;
  getConvPoolArgs(helper, args);
  CAFFE_ENFORCE_EQ(
      args.stride_x,
      args.stride_y,
      "NN API only supports stride_x == stride_y");

  // add input operands to model
  const uint32_t input_indices_count = 10;
  const uint32_t output_indices_count = 1;
  uint32_t input_indices[input_indices_count];
  uint32_t output_indices[output_indices_count];

  uint32_t idx = 0;
  // input
  const std::string& input = op.input(0);
  const std::vector<uint32_t>& input_dims = tensor_dims_[input];
  input_indices[idx++] = operand_map_[input];

  CAFFE_ENFORCE_EQ(input_dims.size(), 4);
  uint32_t batches = input_dims[0];
  uint32_t input_height = input_dims[1];
  uint32_t input_width = input_dims[2];
  uint32_t channel = input_dims[3];

  // pads in the order of left, right, top, bottom
  input_indices[idx++] = addScalarOperand(args.pad_l);
  input_indices[idx++] = addScalarOperand(args.pad_r);
  input_indices[idx++] = addScalarOperand(args.pad_t);
  input_indices[idx++] = addScalarOperand(args.pad_b);

  // strides
  input_indices[idx++] = addScalarOperand(args.stride_x);
  input_indices[idx++] = addScalarOperand(args.stride_y);

  // kernel size
  input_indices[idx++] = addScalarOperand(args.kernel_h);
  input_indices[idx++] = addScalarOperand(args.kernel_w);

  // fuse relu
  FuseCode fuse = fuse_relu ? FuseCode::ANEURALNETWORKS_FUSED_RELU
                            : FuseCode::ANEURALNETWORKS_FUSED_NONE;
  input_indices[idx] = addScalarOperand(fuse);

  // output
  uint32_t output_height =
      (input_height - args.kernel_h + args.pad_t + args.pad_b) / args.stride_y +
      1;
  uint32_t output_width =
      (input_width - args.kernel_w + args.pad_l + args.pad_r) / args.stride_x +
      1;

  float output_scale = helper.GetSingleArgument<float>("output_scale", 1.0);
  int output_zero_point = helper.GetSingleArgument<int>("output_zero_point", 0);

  std::vector<uint32_t> dims({batches, output_height, output_width, channel});
  output_indices[0] = addTensorOperand(
      op.output(0), tensor_type_, dims, output_scale, output_zero_point);

  int result_code = libnnapi_.ANeuralNetworksModel_addOperation(
      model_, op_code, input_indices_count, input_indices, 1, output_indices);
  if (result_code != ANEURALNETWORKS_NO_ERROR) {
    reportError(result_code);
  }
}

void NNApi::addConv(const OperatorDef& op, bool fuse_relu) {
  VLOG(1) << "Add Conv to NN model";
  CAFFE_ENFORCE_EQ(op.input_size(), 3);
  CAFFE_ENFORCE_EQ(op.output_size(), 1);

  ArgumentHelper helper(op);
  StorageOrder order = StringToStorageOrder(
      helper.GetSingleArgument<std::string>("order", "NCHW"));
  CAFFE_ENFORCE_EQ(order, NHWC, "NN API supports NHWC only");

  // input
  const std::string& input = op.input(0);
  const std::vector<uint32_t>& input_dims = tensor_dims_[input];

  CAFFE_ENFORCE_EQ(input_dims.size(), 4);
  uint32_t batches = input_dims[0];
  uint32_t input_height = input_dims[1];
  uint32_t input_width = input_dims[2];
  uint32_t input_channel = input_dims[3];

  uint32_t group = helper.GetSingleArgument<int>("group", 1);

  bool run_depthwise = false;
  if (group > 1) {
    CAFFE_ENFORCE_EQ(
        group,
        input_channel,
        "NN API doesn't support non-depthwise convolution with groups");
    run_depthwise = true;
  }

  ConvPoolArgs args;
  getConvPoolArgs(helper, args);

  CAFFE_ENFORCE_EQ(
      args.stride_x,
      args.stride_y,
      "NN API only supports stride_x == stride_y");

  vector<int> dilation(helper.GetRepeatedArgument<int>("dilations"));
  if (helper.HasArgument("dilation")) {
    dilation.resize(2, helper.GetSingleArgument<int>("dilation", 0));
  } else if (
      helper.HasArgument("dilationh") && helper.HasArgument("dilationw")) {
    dilation.push_back(helper.GetSingleArgument<int>("dilation_h", 0));
    dilation.push_back(helper.GetSingleArgument<int>("dilation_w", 0));
  }

  for (auto d : dilation) {
    CAFFE_ENFORCE_EQ(d, 1, "NN API only supports dialation == 1");
  }

  // add input operands to model
  const uint32_t input_indices_count = run_depthwise ? 11 : 10;
  const uint32_t output_indices_count = 1;
  uint32_t input_indices[input_indices_count];
  uint32_t output_indices[output_indices_count];

  uint32_t idx = 0;
  // input
  input_indices[idx++] = operand_map_[input];

  // weight
  const std::string& weight_name = op.input(1);
  const auto& weight = ws_.GetBlob(weight_name)->Get<TensorCPU>();
  std::vector<uint32_t> weight_dims;
  for (auto dim : weight.dims()) {
    weight_dims.push_back(dim);
  }
  CAFFE_ENFORCE_EQ(weight_dims.size(), 4);
  uint32_t num_kernels = weight_dims[0];
  uint32_t kernel_h = weight_dims[1];
  uint32_t kernel_w = weight_dims[2];
  uint32_t kernel_depth = weight_dims[3];
  CAFFE_ENFORCE_EQ(input_channel, kernel_depth);
  if (run_depthwise) {
    CAFFE_ENFORCE_EQ(num_kernels, 1);
  }

  float weight_scale = helper.GetSingleArgument<float>("weight_scale", 1.0);
  int weight_zero_point = helper.GetSingleArgument<int>("weight_zero_point", 0);

  uint32_t weight_idx = addTensorOperand(
      weight_name, tensor_type_, weight_dims, weight_scale, weight_zero_point);

  int result_code = libnnapi_.ANeuralNetworksModel_setOperandValue(
      model_, weight_idx, weight.raw_data(), weight.nbytes());
  if (result_code != ANEURALNETWORKS_NO_ERROR) {
    reportError(result_code);
  }
  input_indices[idx++] = weight_idx;

  // bias
  const std::string& bias_name = op.input(2);
  const auto& bias = ws_.GetBlob(bias_name)->Get<TensorCPU>();
  std::vector<uint32_t> bias_dims;
  CAFFE_ENFORCE_EQ(bias.ndim(), 1);
  uint32_t bias_size = bias.dim(0);
  if (!run_depthwise) {
    CAFFE_ENFORCE_EQ(num_kernels, bias_size);
  } else {
    CAFFE_ENFORCE_EQ(kernel_depth, bias_size);
  }
  bias_dims.push_back(bias_size);

  OperandCode bias_type = tensor_type_ == ANEURALNETWORKS_TENSOR_FLOAT32
      ? ANEURALNETWORKS_TENSOR_FLOAT32
      : ANEURALNETWORKS_TENSOR_INT32;
  if (bias_type == ANEURALNETWORKS_TENSOR_FLOAT32) {
    CAFFE_ENFORCE(bias.IsType<float>());
  } else if (bias_type == ANEURALNETWORKS_TENSOR_INT32) {
    CAFFE_ENFORCE(bias.IsType<int>());
  }
  uint32_t bias_idx = addTensorOperand(bias_name, bias_type, bias_dims);

  result_code = libnnapi_.ANeuralNetworksModel_setOperandValue(
      model_, bias_idx, bias.raw_data(), bias.nbytes());
  if (result_code != ANEURALNETWORKS_NO_ERROR) {
    reportError(result_code);
  }
  input_indices[idx++] = bias_idx;

  // pads in the order of left, right, top, bottom
  input_indices[idx++] = addScalarOperand(args.pad_l);
  input_indices[idx++] = addScalarOperand(args.pad_r);
  input_indices[idx++] = addScalarOperand(args.pad_t);
  input_indices[idx++] = addScalarOperand(args.pad_b);

  // strides
  input_indices[idx++] = addScalarOperand(args.stride_x);
  input_indices[idx++] = addScalarOperand(args.stride_y);

  // depth_wise
  if (run_depthwise) {
    // depthwise multiplier == 1
    input_indices[idx++] = addScalarOperand(1);
  }

  // fuse relu
  FuseCode fuse = fuse_relu ? FuseCode::ANEURALNETWORKS_FUSED_RELU
                            : FuseCode::ANEURALNETWORKS_FUSED_NONE;
  input_indices[idx] = addScalarOperand(fuse);

  // output
  uint32_t output_channel = run_depthwise ? kernel_depth : num_kernels;
  uint32_t output_height =
      (input_height - args.kernel_h + args.pad_t + args.pad_b) / args.stride_y +
      1;
  uint32_t output_width =
      (input_width - args.kernel_w + args.pad_l + args.pad_r) / args.stride_x +
      1;

  float output_scale = helper.GetSingleArgument<float>("output_scale", 1.0);
  int output_zero_point = helper.GetSingleArgument<int>("output_zero_point", 0);

  std::vector<uint32_t> dims(
      {batches, output_height, output_width, output_channel});
  output_indices[0] = addTensorOperand(
      op.output(0), tensor_type_, dims, output_scale, output_zero_point);
  if (run_depthwise) {
    CAFFE_ENFORCE_EQ(input_indices_count, 11);
    result_code = libnnapi_.ANeuralNetworksModel_addOperation(
        model_,
        ANEURALNETWORKS_DEPTHWISE_CONV_2D,
        input_indices_count,
        input_indices,
        output_indices_count,
        output_indices);
    if (result_code != ANEURALNETWORKS_NO_ERROR) {
      reportError(result_code);
    }
  } else {
    CAFFE_ENFORCE_EQ(input_indices_count, 10);
    result_code = libnnapi_.ANeuralNetworksModel_addOperation(
        model_,
        ANEURALNETWORKS_CONV_2D,
        input_indices_count,
        input_indices,
        output_indices_count,
        output_indices);
    if (result_code != ANEURALNETWORKS_NO_ERROR) {
      reportError(result_code);
    }
  }
}

void NNApi::addRelu(const OperatorDef& op) {
  VLOG(1) << "Add Relu to NN model";
  CAFFE_ENFORCE_EQ(op.input_size(), 1);
  CAFFE_ENFORCE_EQ(op.output_size(), 1);
  const std::string& input = op.input(0);
  uint32_t input_idx = operand_map_[input];

  ArgumentHelper helper(op);
  float output_scale = helper.GetSingleArgument<float>("output_scale", 1.0);
  int output_zero_point = helper.GetSingleArgument<int>("output_zero_point", 0);

  uint32_t output_idx = addTensorOperand(
      op.output(0),
      tensor_type_,
      tensor_dims_[input],
      output_scale,
      output_zero_point);

  int result_code = libnnapi_.ANeuralNetworksModel_addOperation(
      model_, ANEURALNETWORKS_RELU, 1, &input_idx, 1, &output_idx);
  if (result_code != ANEURALNETWORKS_NO_ERROR) {
    reportError(result_code);
  }
}

void NNApi::addSoftmax(const OperatorDef& op) {
  VLOG(1) << "Add Softmax to NN model";
  ArgumentHelper helper(op);
  CAFFE_ENFORCE_EQ(
      helper.GetSingleArgument<int>("axis", 1),
      1,
      "NN API only supports axis == 1");

  uint32_t input_indices[2];
  const std::string& input = op.input(0);
  input_indices[0] = operand_map_[input];
  const auto& input_dims = tensor_dims_[input];
  CAFFE_ENFORCE(
      input_dims.size() == 2 || input_dims.size() == 4,
      "Supported tensor rank: 2 or 4");

  // the positive scaling factor for the exponent, beta
  const float scale = 1.0;
  input_indices[1] = addFloatOperand(scale);

  float output_scale = helper.GetSingleArgument<float>("output_scale", 1.0);
  int output_zero_point = helper.GetSingleArgument<int>("output_zero_point", 0);
  if (tensor_type_ == ANEURALNETWORKS_TENSOR_QUANT8_ASYMM) {
    CAFFE_ENFORCE_EQ(output_scale, 1.f / 256);
    CAFFE_ENFORCE_EQ(output_zero_point, 0);
  }
  uint32_t output_idx = addTensorOperand(
      op.output(0),
      tensor_type_,
      tensor_dims_[input],
      output_scale,
      output_zero_point);

  int result_code = libnnapi_.ANeuralNetworksModel_addOperation(
      model_, ANEURALNETWORKS_SOFTMAX, 2, input_indices, 1, &output_idx);
  if (result_code != ANEURALNETWORKS_NO_ERROR) {
    reportError(result_code);
  }
}

// int32_t
uint32_t NNApi::addScalarOperand(int32_t val) {
  ANeuralNetworksOperandType scalar;
  scalar.type = ANEURALNETWORKS_INT32;
  scalar.scale = 0;
  scalar.zeroPoint = 0;
  scalar.dimensionCount = 0;
  scalar.dimensions = NULL;
  int result_code = libnnapi_.ANeuralNetworksModel_addOperand(model_, &scalar);
  if (result_code != ANEURALNETWORKS_NO_ERROR) {
    reportError(result_code);
  }

  result_code = libnnapi_.ANeuralNetworksModel_setOperandValue(
      model_, operand_idx, &val, sizeof(val));
  if (result_code != ANEURALNETWORKS_NO_ERROR) {
    reportError(result_code);
  }

  VLOG(1) << "Added scalar, " << val << ", at " << operand_idx;
  return operand_idx++;
}

// float32
uint32_t NNApi::addFloatOperand(float val) {
  ANeuralNetworksOperandType scalar;
  scalar.type = ANEURALNETWORKS_TENSOR_FLOAT32;
  scalar.scale = 0;
  scalar.zeroPoint = 0;
  scalar.dimensionCount = 0;
  scalar.dimensions = NULL;
  int result_code = libnnapi_.ANeuralNetworksModel_addOperand(model_, &scalar);
  if (result_code != ANEURALNETWORKS_NO_ERROR) {
    reportError(result_code);
  }

  result_code = libnnapi_.ANeuralNetworksModel_setOperandValue(
      model_, operand_idx, &val, sizeof(val));
  if (result_code != ANEURALNETWORKS_NO_ERROR) {
    reportError(result_code);
  }

  VLOG(1) << "Added scalar, " << val << ", at " << operand_idx;
  return operand_idx++;
}

uint32_t NNApi::addTensorOperand(
    const std::string& blob,
    OperandCode type,
    std::vector<uint32_t>& dims,
    float scale,
    int32_t zero_point)
// clang-format off
{
  // clang-format on
  auto found = operand_map_.find(blob);
  if (found == operand_map_.end()) {
    ANeuralNetworksOperandType tensor;
    tensor.type = type;
    tensor.scale = scale;
    tensor.zeroPoint = zero_point;
    tensor.dimensionCount = dims.size();
    tensor.dimensions = dims.data();

    int result_code =
        libnnapi_.ANeuralNetworksModel_addOperand(model_, &tensor);
    if (result_code != ANEURALNETWORKS_NO_ERROR) {
      reportError(result_code);
    }

    operand_map_[blob] = operand_idx++;
    tensor_dims_[blob] = dims;
    VLOG(1) << "Added operand, " << blob << ", at " << operand_map_[blob];
  }
  return operand_map_[blob];
}

void NNApi::init(const TensorVector& inputs, TensorVector* outputs) {
  // model
  if (!model_) {
    int result_code = libnnapi_.ANeuralNetworksModel_create(&model_);
    if (result_code != ANEURALNETWORKS_NO_ERROR) {
      reportError(result_code);
    }
    if (!model_) {
      CAFFE_THROW("Failed to create NN model");
    } else {
      LOG(INFO) << "Created NN model";
    }

    ArgumentHelper helper(run_net_);
    float scale = helper.GetSingleArgument<float>("scale", 1.0);
    int zero_point = helper.GetSingleArgument<int>("zero_point", 0);

    // add external input dimension
    for (int i = 0; i < inputs.size(); i++) {
      if (inputs[i]->IsType<float>()) {
        tensor_type_ = ANEURALNETWORKS_TENSOR_FLOAT32;
      } else if (inputs[i]->IsType<uint8_t>()) {
        tensor_type_ = ANEURALNETWORKS_TENSOR_QUANT8_ASYMM;
      } else {
        CAFFE_THROW("Unsupported tensor type");
      }
      const std::string& input_blob = run_net_.external_input(i);
      std::vector<uint32_t> dims;
      for (auto dim : inputs[i]->dims()) {
        dims.push_back(dim);
      }
      addTensorOperand(input_blob, tensor_type_, dims, scale, zero_point);
    }

    // add operands and operations
    for (const auto& op : run_net_.op()) {
      if (operator_map_.count(op.type()) == 0) {
        CAFFE_THROW("Unsupported operator");
      }
      switch (operator_map_[op.type()]) {
        case AVERAGEPOOL:
          addPooling(op, ANEURALNETWORKS_AVERAGE_POOL_2D);
          break;
        case CONV:
          addConv(op);
          break;
        case MAXPOOL:
          addPooling(op, ANEURALNETWORKS_MAX_POOL_2D);
          break;
        case RELU:
          addRelu(op);
          break;
        case SOFTMAX:
          addSoftmax(op);
          break;
        default:
          CAFFE_THROW("Unsupported operator");
          break;
      }
    }

    // model inputs and outputs
    int output_size = run_net_.external_output_size();
    std::vector<uint32_t> input_indices(inputs.size());
    std::vector<uint32_t> output_indices(output_size);
    for (int i = 0; i < inputs.size(); i++) {
      input_indices[i] = operand_map_[run_net_.external_input(i)];
    }
    for (int i = 0; i < output_size; i++) {
      output_indices[i] = operand_map_[run_net_.external_output(i)];
    }

    result_code = libnnapi_.ANeuralNetworksModel_identifyInputsAndOutputs(
        model_,
        inputs.size(),
        input_indices.data(),
        output_size,
        output_indices.data());
    if (result_code != ANEURALNETWORKS_NO_ERROR) {
      reportError(result_code);
    }

    result_code = libnnapi_.ANeuralNetworksModel_finish(model_);
    if (result_code != ANEURALNETWORKS_NO_ERROR) {
      reportError(result_code);
    }

    LOG(INFO) << "Finish creating model";

    // compile
    if (!compilation_) {
      result_code =
          libnnapi_.ANeuralNetworksCompilation_create(model_, &compilation_);
      if (result_code != ANEURALNETWORKS_NO_ERROR) {
        reportError(result_code);
      }

      result_code = libnnapi_.ANeuralNetworksCompilation_setPreference(
          compilation_, preference_);
      if (result_code != ANEURALNETWORKS_NO_ERROR) {
        reportError(result_code);
      }

      result_code = libnnapi_.ANeuralNetworksCompilation_finish(compilation_);
      if (result_code != ANEURALNETWORKS_NO_ERROR) {
        reportError(result_code);
      }

      LOG(INFO) << "Finish compilation";
    }

    // pre-execution
    if (!run_) {
      result_code =
          libnnapi_.ANeuralNetworksExecution_create(compilation_, &run_);
      if (result_code != ANEURALNETWORKS_NO_ERROR) {
        reportError(result_code);
      }
      LOG(INFO) << "Created model execution";
    }

    // set external input and output
    for (int i = 0; i < inputs.size(); i++) {
      result_code = libnnapi_.ANeuralNetworksExecution_setInput(
          run_, i, NULL, inputs[i]->raw_data(), inputs[i]->size());
      if (result_code != ANEURALNETWORKS_NO_ERROR) {
        reportError(result_code);
      }

      VLOG(1) << "Set external input " << i << " at " << inputs[i]->raw_data()
              << ", size = " << inputs[i]->size();
    }
    // allocate memory for outputs
    for (int i = 0; i < output_size; i++) {
      const std::string& blob = run_net_.external_output(i);
      if (operand_map_.find(blob) == operand_map_.end()) {
        CAFFE_THROW("Unknown external output, ", blob);
      }
      uint32_t idx = operand_map_[blob];
      if (tensor_dims_.find(blob) == tensor_dims_.end()) {
        CAFFE_THROW("Operand dimension unknown");
      }
      std::vector<int> output_dims;
      for (auto dim : tensor_dims_[blob]) {
        output_dims.push_back(dim);
      }

      auto* tensor = ws_.CreateBlob(blob)->GetMutable<TensorCPU>();
      tensor->Resize(output_dims);
      outputs->push_back(tensor);

      if (tensor_type_ == ANEURALNETWORKS_TENSOR_FLOAT32) {
        result_code = libnnapi_.ANeuralNetworksExecution_setOutput(
            run_,
            i,
            NULL,
            (void*)tensor->template mutable_data<float>(),
            tensor->size());
        if (result_code != ANEURALNETWORKS_NO_ERROR) {
          reportError(result_code);
        }

      } else {
        result_code = libnnapi_.ANeuralNetworksExecution_setOutput(
            run_,
            i,
            NULL,
            (void*)tensor->template mutable_data<uint8_t>(),
            tensor->size());
        if (result_code != ANEURALNETWORKS_NO_ERROR) {
          reportError(result_code);
        }
      }

      VLOG(1) << "Set external output " << i << " at " << tensor->raw_data()
              << ", size = " << tensor->size();
    }
  }
}

} // namespace caffe2
