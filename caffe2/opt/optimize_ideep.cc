#include "caffe2/opt/optimize_ideep.h"
#include "caffe2/opt/converter.h"

#ifdef CAFFE2_USE_MKLDNN
#include <cpuinfo.h>
#include "caffe2/ideep/ideep_utils.h"
#endif

namespace caffe2 {
namespace opt {

using namespace nom;

#ifndef CAFFE2_USE_MKLDNN
void OptimizeForMkldnn(
    repr::NNModule* nn,
    caffe2::Workspace* ws,
    bool training_mode) {
  LOG(WARNING) << "Only support optimizations for IDEEP";
}

#else
USE_IDEEP_DEF_ALIASES();

Blob* getBlob(const std::string name, caffe2::Workspace* ws) {
  CAFFE_ENFORCE(ws->HasBlob(name), "Blob ", name, " not in workspace");
  return ws->GetBlob(name);
}

Blob* getBlob(repr::NNGraph::NodeRef node, caffe2::Workspace* ws) {
  auto tensor = repr::nn::get<repr::Tensor>(node);
  return getBlob(tensor->getName(), ws);
}

template <class T>
T getTensor(Blob* blob) {
  CAFFE_ENFORCE(blob, "Blob is invalid");
  return blob->template Get<T>();
}

template <class T>
T* getMutableTensor(Blob* blob) {
  CAFFE_ENFORCE(blob, "Blob is invalid");
  if (blob && blob->template IsType<T>()) {
    return blob->template GetMutable<T>();
  }
  return nullptr;
}

const caffe2::OperatorDef& getOpDef(const repr::NeuralNetOperator& nnOp) {
  auto annotation = nnOp.getAnnotation();
  if (annotation == nullptr) {
    CAFFE_THROW("Cannot get Operator annotation");
  }
  return dyn_cast<Caffe2Annotation>(annotation)->getOperatorDef();
}

caffe2::OperatorDef* getMutableOpDef(repr::NeuralNetOperator& nnOp) {
  auto annotation = nnOp.getMutableAnnotation();
  if (annotation == nullptr) {
    CAFFE_THROW("Cannot get Operator annotation");
  }
  return dyn_cast<Caffe2Annotation>(annotation)->getMutableOperatorDef();
}

bool isOpType(const repr::NNGraph::NodeRef& nodeRef, string typeName) {
  if (!repr::nn::is<repr::NeuralNetOperator>(nodeRef)) {
    return false;
  }
  auto op = repr::nn::get<repr::NeuralNetOperator>(nodeRef);
  auto opDef = getOpDef(*op);
  return opDef.type() == typeName;
}

bool isOnIdeepDevice(const repr::NeuralNetOperator& nnOp) {
  // We only want to fuse for IDEEP operators
  const auto& op = getOpDef(nnOp);
  return op.device_option().device_type() == DeviceTypeProto::PROTO_IDEEP;
}

bool isConvFusion(repr::NNGraph::NodeRef convNode, int fusion_type) {
  // Here we only check the type of ConvFusion op (for FP32 only)
  if (!repr::nn::is<repr::Conv>(convNode)) {
    return false;
  }

  auto conv = repr::nn::get<repr::Conv>(convNode);
  auto& op = getOpDef(*conv);

  if (op.type() == "ConvFusion") {
    for (const auto& arg : op.arg()) {
      if (arg.name() == "fusion_type") {
        if (fusion_type == FUSION_MAX) {
          return true;
        }
        return arg.i() == fusion_type;
      }
    }
  }

  return false;
}

void resetConvForFusion(repr::NNGraph::NodeRef convNode, int fusion_type) {
  auto conv = repr::nn::get<repr::Conv>(convNode);
  auto* op = getMutableOpDef(*conv);

  if (op == nullptr) {
    return;
  }

  if (op->type() == "ConvFusion") {
    CAFFE_ENFORCE(fusion_type == FUSION_CONV_RELU, "Invalid nest fusion");
    for (auto& arg : *op->mutable_arg()) {
      if (arg.name() == "fusion_type") {
        CAFFE_ENFORCE(arg.i() == FUSION_CONV_SUM, "Invalid nest fusion");
        // Only from FUSION_CONV_SUM to FUSION_CONV_SUM_RELU
        arg.set_i(FUSION_CONV_SUM_RELU);
        return;
      }
    }
    CAFFE_THROW("Can not find fusion type in ConvFusion");
  }

  CAFFE_ENFORCE_LT(fusion_type, FUSION_CONV_SUM_RELU, "Invalid fusion type");
  op->set_type("ConvFusion");
  auto* arg = op->add_arg();
  arg->set_name("fusion_type");
  arg->set_i(fusion_type);
}

void removeArg(repr::NeuralNetOperator& nnOp, std::string argName) {
  auto* op = getMutableOpDef(nnOp);
  auto& opArgs = *op->mutable_arg();
  auto remove_arg = [](decltype(opArgs)& args, std::string& name) {
    for (auto it = args.begin(); it != args.end(); it++) {
      if (it->name() == name) {
        args.erase(it);
        return true;
      }
    }
    return false;
  };
  while (remove_arg(opArgs, argName))
    ;
}

void moveOpArg(
    caffe2::Workspace* ws,
    std::string argName,
    repr::NeuralNetOperator* srcOp,
    repr::NeuralNetOperator* dstOp) {
  if (argName.empty() || srcOp == nullptr || dstOp == nullptr || srcOp == dstOp)
    return;
  removeArg(*dstOp, argName);

  auto& src = getOpDef(*srcOp);
  auto& src_args = src.arg();
  auto src_it = src_args.begin();
  for (; src_it != src_args.end(); src_it++) {
    if (src_it->name() == argName)
      break;
  }
  if (src_it == src_args.end())
    return;

  auto* dst = getMutableOpDef(*dstOp);
  auto* arg = dst->add_arg();
  *arg = *src_it;
  arg->set_name(argName);
}

bool removeStopGradientForInference(repr::NNModule* nn, caffe2::Workspace* ws) {
  auto allNodes = nn->dataFlow.getMutableNodes();
  for (int i = 0; i < allNodes.size(); ++i) {
    auto node = allNodes[i];
    if (!isOpType(node, "StopGradient")) {
      continue;
    }

    auto stopGradInput = repr::nn::getInputs(node).front();
    auto stopGradOutput = repr::nn::getOutputs(node).front();
    auto inputName = repr::nn::get<repr::Tensor>(stopGradInput)->getName();
    auto outputName = repr::nn::get<repr::Tensor>(stopGradOutput)->getName();
    if (inputName == outputName) {
      nn->dataFlow.replaceNode(stopGradOutput, stopGradInput);
      nn->dataFlow.deleteNode(node);
      return true;
    }
  }
  return false;
}

bool fuseConvBNAndAffCh(repr::NNModule* nn, caffe2::Workspace* ws) {
  for (auto node_pair : repr::nn::dataIterator<repr::Conv>(nn->dataFlow)) {
    bool no_bias = false;
    repr::NNGraph::NodeRef convNode;
    repr::Conv* conv;
    std::tie(conv, convNode) = node_pair;

    if (!isOnIdeepDevice(*conv)) {
      LOG(WARNING) << "Not a IDEEP operator";
      continue;
    }

    const auto& convOp = getOpDef(*conv);
    if (convOp.type() == "ConvFusion") {
      continue;
    }

    auto convOutput = repr::nn::getOutputs(convNode).front();
    auto consumers = repr::nn::getConsumers(convOutput);
    // convOutput is NOT referenced by sequential ops after BN.
    if (consumers.size() != 1) {
      continue;
    }

    bool isBN;
    auto consumer = consumers.front();
    if (repr::nn::is<repr::BatchNormalization>(consumer)) {
      isBN = true;
    } else if (isOpType(consumer, "AffineChannel")) {
      isBN = false;
    } else {
      continue;
    }

    auto bnOrAffChNode = consumer;
    auto bn =
        isBN ? repr::nn::get<repr::BatchNormalization>(bnOrAffChNode) : nullptr;
    auto bnOrAffChOutput = repr::nn::getOutputs(bnOrAffChNode).front();

    auto convInputs = repr::nn::getInputs(convNode);
    if (convInputs.size() < 2) {
      LOG(WARNING) << "Invalid convolution input size";
      continue;
    }

    auto bnOrAffChInputs = repr::nn::getInputs(bnOrAffChNode);
    int numInputs = isBN ? 5 : 3;
    if (bnOrAffChInputs.size() < numInputs) {
      LOG(WARNING) << "Invalid input size: " << bnOrAffChInputs.size()
                   << ", expect " << numInputs;
      continue;
    }

    // When no bias, borrow BN bias
    if (convInputs.size() < 3) {
      no_bias = true;
      nn->dataFlow.createEdge(bnOrAffChInputs[2], convNode);
      convInputs = repr::nn::getInputs(convNode);
    }

#define EXPOSE_TENSOR_DATA(name, index, nodes, need_init)                  \
  itensor* name = nullptr;                                                 \
  itensor name##Tensor;                                                    \
  float* name##Data = nullptr;                                             \
  if (need_init) {                                                         \
    name = getMutableTensor<itensor>(getBlob(nodes[index], ws));           \
    if (name == nullptr) {                                                 \
      LOG(WARNING) << #name " not a IDEEP tensor";                         \
      continue;                                                            \
    }                                                                      \
    name##Tensor.resize(name->get_dims(), name->get_data_type());          \
    name##Tensor.feed_from(*name);                                         \
    CAFFE_ENFORCE(                                                         \
        name##Tensor.is_public_format(), #name " not with public format"); \
    name##Data = static_cast<float*>(name##Tensor.get_data_handle());      \
  }

    EXPOSE_TENSOR_DATA(filter, 1, convInputs, true);
    EXPOSE_TENSOR_DATA(biasConv, 2, convInputs, true);

    EXPOSE_TENSOR_DATA(scale, 1, bnOrAffChInputs, true);
    EXPOSE_TENSOR_DATA(biasBNOrAffCh, 2, bnOrAffChInputs, true);
    EXPOSE_TENSOR_DATA(mean, 3, bnOrAffChInputs, isBN);
    EXPOSE_TENSOR_DATA(variance, 4, bnOrAffChInputs, isBN);

#undef EXPOSE_TENSOR_DATA

    // Assume M{CHW,HWC}
    auto chwDim = filterTensor.get_dim(1) * filterTensor.get_dim(2) *
        filterTensor.get_dim(3);
    for (auto c = 0; c < filterTensor.get_dim(0); ++c) {
      float mean_val = 0;
      float variance_val = 1;
      if (isBN) {
        mean_val = meanData[c];
        variance_val = std::sqrt(varianceData[c] + bn->getEpsilon());
      }
      float coeff = scaleData[c] / variance_val;
      for (auto i = 0; i < chwDim; ++i) {
        filterData[c * chwDim + i] *= coeff;
      }

      if (no_bias) {
        biasConvData[c] = biasBNOrAffChData[c] - mean_val * coeff;
      } else {
        biasConvData[c] =
            biasBNOrAffChData[c] + (biasConvData[c] - mean_val) * coeff;
      }
    }

    filter->feed_from(filterTensor);
    biasConv->feed_from(biasConvTensor);
    nn->dataFlow.replaceNode(convOutput, bnOrAffChOutput);

    nn->dataFlow.deleteNode(bnOrAffChNode);
    nn->dataFlow.deleteNode(convOutput);

    return true;
  }
  return false;
}

bool fuseConvSum(repr::NNModule* nn, caffe2::Workspace* ws) {
  CAFFE_ENFORCE(cpuinfo_initialize(), "failed to initialize cpuinfo");
  // Assume the order of nodes from getMutableNodes conforms to
  // the original topo order of operators
  auto allNodes = nn->dataFlow.getMutableNodes();
  for (int i = allNodes.size() - 1; i > 0; i--) {
    auto sumNode = allNodes[i];
    if (!repr::nn::hasInputs(sumNode)) {
      continue;
    }

    // [Caution] on IDEEP device, only element-wise Add operator is
    // supported yet. It totally works as element-wise sum without scalar
    // broadcast.
    bool is_dnnlowp_sum = false;
    if (isOpType(sumNode, "Int8Sum") || isOpType(sumNode, "Int8Add") ||
        isOpType(sumNode, "Int8SumRelu") || isOpType(sumNode, "Int8AddRelu")) {
      is_dnnlowp_sum = true;
    } else if (!repr::nn::is<repr::Sum>(sumNode) && !isOpType(sumNode, "Add")) {
      continue;
    }

    auto sum = repr::nn::get<repr::NeuralNetOperator>(sumNode);
    if (!isOnIdeepDevice(*sum)) {
      LOG(WARNING) << "Not a IDEEP operator";
      continue;
    }

    auto sumInputs = repr::nn::getInputs(sumNode);
    if (sumInputs.size() != 2) {
      continue;
    }

    int sum_idx = i;
    repr::NNGraph::NodeRef convNode = nullptr;
    while (--i >= 0) {
      if (repr::nn::is<repr::NeuralNetOperator>(allNodes[i])) {
        // Find the nearest conv Op before Sum
        if (repr::nn::is<repr::Conv>(allNodes[i]) ||
            isOpType(allNodes[i], "Int8Conv")) {
          convNode = allNodes[i];
          break;
        }
      }
    }
    if (convNode == nullptr || isConvFusion(convNode, FUSION_MAX)) {
      continue;
    }
    int conv_idx = i;

    auto conv = repr::nn::get<repr::NeuralNetOperator>(convNode);
    if (!isOnIdeepDevice(*conv)) {
      LOG(WARNING) << "Not a IDEEP operator";
      continue;
    }

    auto group = 1;
    auto* convOp = getMutableOpDef(*conv);
    for (const auto& arg : convOp->arg()) {
      if (arg.name() == "group") {
        group = arg.i();
        break;
      }
    }
    if (group > 1 && !cpuinfo_has_x86_avx512f()) {
      LOG(WARNING) << "Not support conv sum fusion with grouped filter";
      continue;
    }

    auto convOutput = repr::nn::getOutputs(convNode).front();
    if (convOutput != sumInputs[0] && convOutput != sumInputs[1]) {
      continue;
    }
    repr::NNGraph::NodeRef sumInputX =
        (sumInputs[0] == convOutput ? sumInputs[1] : sumInputs[0]);
    CAFFE_ENFORCE(sumInputX != nullptr, "Invalid sum inputs");
    if (sumInputX->getInEdges().size() <= 0) {
      continue;
    }

    auto preNode = repr::nn::getProducer(sumInputX);
    if (preNode == nullptr || !repr::nn::is<repr::NeuralNetOperator>(preNode)) {
      LOG(WARNING) << "Can not fuse Conv Sum";
      continue;
    }
    int pre_idx = sum_idx - 1;
    while (pre_idx >= 0) {
      if (preNode == allNodes[pre_idx]) {
        break;
      }
      pre_idx--;
    }

    bool should_fuse = true;
    auto convInput = repr::nn::getInputs(convNode).front();
    for (int idx = conv_idx + 1; idx < allNodes.size() - 1; ++idx) {
      if (idx == sum_idx ||
          !repr::nn::is<repr::NeuralNetOperator>(allNodes[idx])) {
        continue;
      }

      auto checkNode = allNodes[idx];
      auto checkInputs = repr::nn::getInputs(checkNode);
      // Conv output should not be used by other ops after Conv node (except the
      // fused Sum) The other Sum input (sumInputX) should not be used by the
      // other ops after Sum node due to the Sum output is inplace with
      // sumInputX
      for (size_t input_idx = 0; input_idx < checkInputs.size(); ++input_idx) {
        if (convOutput == checkInputs[input_idx] ||
            (idx > sum_idx && sumInputX == checkInputs[input_idx])) {
          should_fuse = false;
          break;
        }
      }
      if (!should_fuse) {
        break;
      }

      // If fuse Conv with Sum, the Conv op will be pulled down between preNode
      // and Sum Check Conv input tensor buffer has been re-written by other ops
      // between Conv and preNode
      if (idx <= pre_idx) {
        auto checkOutputs = repr::nn::getOutputs(checkNode);
        for (size_t output_idx = 0; output_idx < checkOutputs.size();
             ++output_idx) {
          auto check_output_tensor =
              repr::nn::get<repr::Tensor>(checkOutputs[output_idx]);
          auto conv_input_tensor = repr::nn::get<repr::Tensor>(convInput);
          if (conv_input_tensor->getName() == check_output_tensor->getName()) {
            should_fuse = false;
            break;
          }
        }
      }
      if (!should_fuse) {
        break;
      }
    }
    if (!should_fuse) {
      continue;
    }

    nn->dataFlow.createEdge(sumInputX, convNode);
    auto newOutputName = repr::nn::get<repr::Tensor>(sumInputX)->getName() +
        "_fusion_fix_" + std::to_string(i);

    auto newInputTensor = c10::guts::make_unique<repr::Tensor>(newOutputName);
    auto newInput = nn->dataFlow.createNode(
        unique_dyn_cast<repr::NeuralNetData>(newInputTensor));

    nn->dataFlow.replaceNode(sumInputX, newInput);
    nn->dataFlow.deleteNode(sumInputX);

    auto newOutputTensor = c10::guts::make_unique<repr::Tensor>(newOutputName);
    auto newOutput = nn->dataFlow.createNode(
        unique_dyn_cast<repr::NeuralNetData>(newOutputTensor));

    auto sumOutput = repr::nn::getOutputs(sumNode).front();
    nn->dataFlow.replaceNode(sumOutput, newOutput);
    nn->dataFlow.createEdge(convNode, newOutput);

    if (!is_dnnlowp_sum) {
      resetConvForFusion(convNode, FUSION_CONV_SUM);
    } else {
      moveOpArg(ws, "Y_scale", sum, conv);
      moveOpArg(ws, "Y_zero_point", sum, conv);

      if (isOpType(sumNode, "Int8Sum") || isOpType(sumNode, "Int8Add")) {
        convOp->set_type("Int8ConvSum");
      } else if (
          isOpType(sumNode, "Int8SumRelu") ||
          isOpType(sumNode, "Int8AddRelu")) {
        convOp->set_type("Int8ConvSumRelu");
      } else {
        CAFFE_THROW("Unsupport operator in conv fusion");
      }
    }

    nn->dataFlow.deleteNode(sumNode);
    nn->dataFlow.deleteNode(sumOutput);
    nn->dataFlow.deleteNode(convOutput);
    return true;
  }
  return false;
}

bool fuseActivation(repr::NNModule* nn, caffe2::Workspace* ws) {
  // Conv+Relu fusion
  for (auto node_pair : repr::nn::dataIterator<repr::Conv>(nn->dataFlow)) {
    repr::NNGraph::NodeRef conv_node;
    repr::Conv* conv;
    std::tie(conv, conv_node) = node_pair;

    // Check topological feasibility
    auto conv_outputs = repr::nn::getOutputs(conv_node);
    if (conv_outputs.size() != 1) {
      continue;
    }
    auto conv_output = conv_outputs.front();

    auto consumers = repr::nn::getConsumers(conv_output);
    if (consumers.size() != 1) {
      continue;
    }
    if (!repr::nn::is<repr::Relu>(consumers.front())) {
      continue;
    }
    auto relu_node = consumers.front();
    auto relu = repr::nn::get<repr::Relu>(relu_node);

    auto relu_outputs = repr::nn::getOutputs(relu_node);
    if (relu_outputs.size() != 1) {
      continue;
    }

    // Check feasibility with application specific logic
    if (!isOnIdeepDevice(*conv)) {
      continue;
    }

    // Ready to fuse
    auto relu_output = relu_outputs.front();
    auto output_tensor = repr::nn::get<repr::Tensor>(relu_output);
    auto output_node = relu_output;
    auto input_tensor =
        repr::nn::get<repr::Tensor>(repr::nn::getInputs(conv_node).front());

    if (isConvFusion(conv_node, FUSION_CONV_SUM)) {
      nn->dataFlow.replaceNode(relu_output, conv_output);
      nn->dataFlow.deleteNode(relu_node);
      nn->dataFlow.deleteNode(relu_output);
    } else {
      // Conv cannot be in-place
      if (output_tensor->getName() != input_tensor->getName()) {
        nn->dataFlow.replaceNode(conv_output, relu_output);
        nn->dataFlow.deleteNode(relu_node);
        nn->dataFlow.deleteNode(conv_output);
      } else {
        nn->dataFlow.replaceNode(relu_output, conv_output);
        output_tensor = repr::nn::get<repr::Tensor>(conv_output);
        output_node = conv_output;
        nn->dataFlow.deleteNode(relu_node);
        nn->dataFlow.deleteNode(relu_output);
      }

      // We may have accidentally made the next op in-place
      // In future iterations of transformations this won't be an issue,
      // but current caffe2 predictor usage requires things like
      // external_input and output to be unchanged.
      bool rectify_inplace = false;
      for (auto& consumer : repr::nn::getConsumers(output_node)) {
        for (auto& consumer_output : repr::nn::getOutputs(consumer)) {
          auto co_name =
              repr::nn::get<repr::Tensor>(consumer_output)->getName();
          if (co_name == output_tensor->getName()) {
            rectify_inplace = true;
          }
        }
      }
      if (rectify_inplace) {
        auto new_output = nn->dataFlow.createNode(make_unique<repr::Tensor>(
            output_tensor->getName() + "_fusion_fix"));
        nn->dataFlow.replaceNode(output_node, new_output);
      }
    }

    resetConvForFusion(conv_node, FUSION_CONV_RELU);
    return true;
  }
  return false;
}

bool enforceFusionInplace(repr::NNModule* nn, caffe2::Workspace* ws) {
  // For fusions of Conv+Sum or Conv+Sum+ReLU, the last input and output must
  // be inplaced. To enforce inplace, here to re-check whole graph and correct
  // the ConvFusion Ops.
  auto allNodes = nn->dataFlow.getMutableNodes();
  for (int i = allNodes.size() - 1; i > 0; i--) {
    auto convNode = allNodes[i];
    if (convNode == nullptr ||
        !repr::nn::is<repr::NeuralNetOperator>(convNode)) {
      continue;
    }

    auto conv = repr::nn::get<repr::NeuralNetOperator>(convNode);
    if (!isOnIdeepDevice(*conv)) {
      LOG(WARNING) << "Not a IDEEP operator";
      continue;
    }

    if (repr::nn::is<repr::Conv>(convNode)) {
      if (!isConvFusion(convNode, FUSION_CONV_SUM) &&
          !isConvFusion(convNode, FUSION_CONV_SUM_RELU))
        continue;
    } else if (
        !isOpType(convNode, "Int8ConvSum") &&
        !isOpType(convNode, "Int8ConvSumRelu")) {
      continue;
    }

    auto convInput = repr::nn::getInputs(convNode).back();
    auto inputName = repr::nn::get<repr::Tensor>(convInput)->getName();
    auto convOutput = repr::nn::getOutputs(convNode).front();
    auto outputName = repr::nn::get<repr::Tensor>(convOutput)->getName();
    if (inputName == outputName) {
      continue;
    }

    auto consumer = repr::nn::getConsumers(convInput).back();
    if (consumer != convNode) {
      LOG(ERROR) << "Can not enforce to inplace for fusion";
      return false;
    }

    auto newOutputTensor = c10::guts::make_unique<repr::Tensor>(inputName);
    auto newOutput = nn->dataFlow.createNode(
        unique_dyn_cast<repr::NeuralNetData>(newOutputTensor));
    nn->dataFlow.replaceNode(convOutput, newOutput);
    nn->dataFlow.deleteNode(convOutput);

    return true;
  }
  return false;
}

bool fuseOrderSwitchToQuantizeOp(repr::NNModule* nn, caffe2::Workspace* ws) {
  // In INT8 module, the quantize/dequantize op always appears
  // along with corresponding order switch op, which aims to switch
  // between INT8 computation domain and others.
  // Here we assume they always obey below combination and order:
  // NCHW2NHWC followed by Int8Quantize, or Int8Dequantize followed by NHWC2NCHW
  // On iDEEP, there is chance to fuse the order switch op into the
  // quantize/dequantize op, in order to improve the module performance.
  auto allNodes = nn->dataFlow.getMutableNodes();
  for (int i = 0; i < allNodes.size(); ++i) {
    auto osNode = allNodes[i];
    if (osNode == nullptr || !repr::nn::is<repr::NeuralNetOperator>(osNode)) {
      continue;
    }

    if (isOpType(osNode, "NCHW2NHWC")) {
      auto output = repr::nn::getOutputs(osNode).front();
      auto consumers = repr::nn::getConsumers(output);
      if (consumers.size() != 1) {
        continue;
      }

      auto seqNode = consumers.front();
      if (!isOpType(seqNode, "Int8Quantize")) {
        continue;
      }

      auto seq = repr::nn::get<repr::NeuralNetOperator>(seqNode);
      removeArg(*seq, "output_order");

      auto* seqOp = getMutableOpDef(*seq);
      auto* arg = seqOp->add_arg();
      arg->set_name("output_order");
      arg->set_i(iformat::nhwc);

      auto input = repr::nn::getInputs(osNode).front();
      nn->dataFlow.replaceNode(output, input);

      nn->dataFlow.deleteNode(osNode);
      nn->dataFlow.deleteNode(output);
      return true;
    } else if (isOpType(osNode, "NHWC2NCHW")) {
      auto input = repr::nn::getInputs(osNode).front();
      if (input->getInEdges().size() <= 0) {
        continue;
      }

      auto preNode = repr::nn::getProducer(input);
      if (!isOpType(preNode, "Int8Dequantize")) {
        continue;
      }

      auto pre = repr::nn::get<repr::NeuralNetOperator>(preNode);
      removeArg(*pre, "output_order");

      auto* preOp = getMutableOpDef(*pre);
      auto* arg = preOp->add_arg();
      arg->set_name("output_order");
      arg->set_i(iformat::nchw);

      auto output = repr::nn::getOutputs(osNode).front();
      nn->dataFlow.replaceNode(input, output);

      nn->dataFlow.deleteNode(osNode);
      nn->dataFlow.deleteNode(input);
      return true;
    }
  }
  return false;
}

bool fusePreConvertOp(repr::NNModule* nn, caffe2::Workspace* ws) {
  // 1. Int8Sum has been fallbacked to FP32 in current impl
  //    It can handle inputs with diff format and data type
  // 2. FC is able to convert input format and data type by itself
  // 3. The fallback wrapper can handle the conversion of format and data type
  static vector<string> op_list = {
      "FC",
      "Python",
      "Softmax",
      "Sigmoid",
      "RoIAlign",
      "UpsampleNearest",
      "BatchPermutation",
      "Int8Sum",
      "Int8SumRelu",
  };

  auto allNodes = nn->dataFlow.getMutableNodes();
  for (int i = 0; i < allNodes.size(); ++i) {
    auto opNode = allNodes[i];
    if (opNode == nullptr || !repr::nn::is<repr::NeuralNetOperator>(opNode)) {
      continue;
    }

    if (!isOpType(opNode, "NCHW2NHWC") && !isOpType(opNode, "NHWC2NCHW") &&
        !isOpType(opNode, "Int8Quantize") &&
        !isOpType(opNode, "Int8Dequantize")) {
      continue;
    }

    auto op = repr::nn::get<repr::NeuralNetOperator>(opNode);
    if (!isOnIdeepDevice(*op)) {
      LOG(WARNING) << "Not a IDEEP operator";
      continue;
    }

    auto output = repr::nn::getOutputs(opNode).front();
    auto consumers = repr::nn::getConsumers(output);
    if (consumers.size() != 1) {
      continue;
    }

    bool is_op_found = false;
    auto seqNode = consumers.front();
    for (int j = 0; j < op_list.size(); j++) {
      if (isOpType(seqNode, op_list[j])) {
        is_op_found = true;
        break;
      }
    }
    if (!is_op_found) {
      continue;
    }

    auto seqOp = repr::nn::get<repr::NeuralNetOperator>(seqNode);
    if (!isOnIdeepDevice(*seqOp)) {
      LOG(WARNING) << "Not a IDEEP operator";
      continue;
    }

    auto input = repr::nn::getInputs(opNode).front();

    if (isOpType(opNode, "Int8Dequantize") &&
        repr::nn::hasSingleOutputAndConsumer(opNode)) {
      auto preNode = repr::nn::getProducer(input);
      if (isOpType(preNode, "Int8FC") &&
          repr::nn::hasSingleOutputAndConsumer(preNode)) {
        auto predOp = repr::nn::get<repr::NeuralNetOperator>(preNode);
        removeArg(*predOp, "Y_scale");
        removeArg(*predOp, "Y_zero_point");
      }
    }

    nn->dataFlow.replaceNode(output, input);

    nn->dataFlow.deleteNode(opNode);
    nn->dataFlow.deleteNode(output);
    return true;
  }
  return false;
}

void setPoolingInferenceMode(repr::NNModule* nn) {
  auto setTrainingMode = [](repr::NeuralNetOperator& pool) {
    if (!isOnIdeepDevice(pool)) {
      LOG(WARNING) << "Not a IDEEP operator";
      return;
    }
    auto* op = getMutableOpDef(pool);
    bool found_training_mode = false;
    for (auto& arg : *op->mutable_arg()) {
      if (arg.name() == "training_mode") {
        arg.set_i(0);
        found_training_mode = true;
        break;
      }
    }
    if (!found_training_mode) {
      auto* arg = op->add_arg();
      arg->set_name("training_mode");
      arg->set_i(0);
    }
  };

  auto allNodes = nn->dataFlow.getMutableNodes();
  for (int i = 0; i < allNodes.size(); ++i) {
    auto poolNode = allNodes[i];
    if (poolNode == nullptr ||
        !repr::nn::is<repr::NeuralNetOperator>(poolNode)) {
      continue;
    }

    if (isOpType(poolNode, "FC") || isOpType(poolNode, "Conv") ||
        isOpType(poolNode, "ConvFusion") || isOpType(poolNode, "MaxPool") ||
        isOpType(poolNode, "AveragePool") || isOpType(poolNode, "Int8FC") ||
        isOpType(poolNode, "Int8Conv") || isOpType(poolNode, "Int8ConvRelu") ||
        isOpType(poolNode, "Int8ConvSum") ||
        isOpType(poolNode, "Int8ConvSumRelu") ||
        isOpType(poolNode, "Int8MaxPool") ||
        isOpType(poolNode, "Int8AveragePool")) {
      auto pool = repr::nn::get<repr::NeuralNetOperator>(poolNode);
      setTrainingMode(*pool);
    }
  }
}

// Pre-convert filters format to expected one here
// in order to avoid boring conversions during computations
void preConvertFiltersFormat(repr::NNModule* nn, caffe2::Workspace* ws) {
  for (auto& node : nn->dataFlow.getMutableNodes()) {
    if (!repr::nn::is<repr::ConvTranspose>(node) &&
        !repr::nn::is<repr::Conv>(node) && !repr::nn::is<repr::FC>(node)) {
      continue;
    }

    auto* nnOp = repr::nn::get<repr::NeuralNetOperator>(node);
    if (!isOnIdeepDevice(*nnOp)) {
      LOG(INFO) << "Not a IDEEP operator";
      continue;
    }

    auto inputs = repr::nn::getInputs(node);
    if (inputs.size() < 2) {
      LOG(WARNING) << "Invalid input size";
      continue;
    }

    auto* filterBlob = getBlob(inputs[1], ws);
    auto* filter = getMutableTensor<itensor>(filterBlob);
    if (filter == nullptr) {
      continue;
    }

    itensor::descriptor expectedDesc;
    if (repr::nn::is<repr::ConvTranspose>(node)) {
      if (filter->get_public_format() == ideep::format::iohw)
        continue;
      auto convTranspose = repr::nn::get<repr::ConvTranspose>(node);
      auto initValue = [](vector<int>& v, vector<int> i) {
        if (v.empty())
          v = i;
      };
      auto strides = convTranspose->getStrides();
      initValue(strides, {1, 1});
      auto pads = convTranspose->getPads();
      initValue(pads, {0, 0, 0, 0});
      auto* op = getMutableOpDef(*convTranspose);
      auto aalgorithm = ialgo::deconvolution_direct;
      auto dataType = filter->get_data_type();
      ideep::tensor::dims filter_dims_mkldnn{filter->get_dim(1),
                                             filter->get_dim(0),
                                             filter->get_dim(2),
                                             filter->get_dim(3)};
      expectedDesc =
          ideep::convolution_transpose_forward::expected_weights_descriptor(
              filter_dims_mkldnn,
              dataType,
              strides,
              {pads[0], pads[1]},
              {pads[2], pads[3]});

      if (filter->get_descriptor() != expectedDesc) {
        filter->set_public_format(ideep::format::iohw);
        itensor newFilter;
        newFilter.init(expectedDesc);
        newFilter.feed_from(*filter);
        newFilter.set_public_format(ideep::format::iohw);
        filterBlob->Reset<itensor>(new itensor(std::move(newFilter)));
      }
    } else if (repr::nn::is<repr::Conv>(node)) {
      auto conv = repr::nn::get<repr::Conv>(node);
      auto initValue = [](vector<int>& v, vector<int> i) {
        if (v.empty())
          v = i;
      };
      auto strides = conv->getStrides();
      initValue(strides, {1, 1});
      auto pads = conv->getPads();
      initValue(pads, {0, 0, 0, 0});
      auto dilations = conv->getDilations();
      initValue(dilations, {1, 1});

      auto* op = getMutableOpDef(*conv);
      auto aalgorithm = ialgo::convolution_direct;
      for (auto& arg : *op->mutable_arg()) {
        if ((arg.name() == "conv_algorithm") &&
            (arg.i() == CONV_ALGORITHM_WINOGRAD)) {
          aalgorithm = ialgo::convolution_winograd;
        }
      }
      auto dataType = filter->get_data_type();

      filter->make_group(conv->getGroup());
      expectedDesc = ideep::convolution_forward::expected_weights_descriptor(
          filter->get_dims(),
          dataType,
          strides,
          {pads[0], pads[1]},
          {pads[2], pads[3]},
          dilations,
          conv->getGroup(),
          aalgorithm);

      if (filter->get_descriptor() != expectedDesc) {
        itensor newFilter;
        newFilter.init(expectedDesc);
        newFilter.feed_from(*filter);
        filterBlob->Reset<itensor>(new itensor(std::move(newFilter)));
      }
      // convert weights for FC
    } else if (repr::nn::is<repr::FC>(node)) {
      auto fc = repr::nn::get<repr::FC>(node);
      auto axis_w = fc->getAxisW();
      if (axis_w != 1) {
        auto f_dims = filter->get_dims();
        auto f_dim0 = std::accumulate(
            f_dims.begin(),
            f_dims.begin() + axis_w,
            1,
            std::multiplies<itensor::dim_t>());
        auto f_dim1 = std::accumulate(
            f_dims.begin() + axis_w,
            f_dims.end(),
            1,
            std::multiplies<itensor::dim_t>());
        filter->reshape({f_dim0, f_dim1});
      }

      expectedDesc = ideep::inner_product_forward::expected_weights_descriptor(
          filter->get_dims());

      if (filter->get_descriptor() != expectedDesc) {
        itensor newFilter;
        newFilter.init(expectedDesc);
        newFilter.feed_from(filter->as_weights());
        filterBlob->Reset<itensor>(new itensor(std::move(newFilter)));
      }
    }
  }
}

// Fusers for ideep to parse the graph and apply operator fusion
using Fuser = bool (*)(repr::NNModule* nn, caffe2::Workspace* ws);
static Fuser fusers[] = {
    removeStopGradientForInference,
    fuseConvBNAndAffCh,
    fuseConvSum,
    fuseActivation,
    enforceFusionInplace,
    fuseOrderSwitchToQuantizeOp,
    fusePreConvertOp,
};

void OptimizeForMkldnn(
    repr::NNModule* nn,
    caffe2::Workspace* ws,
    bool training_mode) {
  if (training_mode) {
    preConvertFiltersFormat(nn, ws);
    return;
  }

  for (auto fuser : fusers) {
    while (fuser(nn, ws)) {
    }
  }

  setPoolingInferenceMode(nn);
}

#endif // CAFFE2_USE_MKLDNN

} // namespace opt
} // namespace caffe2
