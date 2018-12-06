#include "caffe2/opt/optimize_ideep.h"
#include "caffe2/opt/converter.h"
#include "caffe2/opt/fusion.h"

#ifdef CAFFE2_USE_MKLDNN
#include "caffe2/ideep/ideep_utils.h"
#endif

namespace caffe2 {
namespace opt {

using namespace nom;

#ifndef CAFFE2_USE_MKLDNN
void OptimizeForIdeep(
    repr::NNModule* nn,
    caffe2::Workspace* ws,
    bool training_mode) {
  LOG(WARNING) << "Only support optimizations for IDEEP";
}

#else
USE_IDEEP_DEF_ALIASES();

Blob* getBlob(repr::NNGraph::NodeRef node, caffe2::Workspace* ws) {
  auto tensor = repr::nn::get<repr::Tensor>(node);
  CAFFE_ENFORCE(ws->HasBlob(tensor->getName()), "Blob not in workspace");
  return ws->GetBlob(tensor->getName());
}

template <class T>
T* getTensor(Blob* blob) {
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

bool isOnIdeepDevice(const repr::NeuralNetOperator& nnOp) {
  // We only want to fuse for IDEEP convs
  const auto& op = getOpDef(nnOp);
  return op.device_option().device_type() == DeviceTypeProto::PROTO_IDEEP;
}

bool shouldFuseConv(const repr::Conv& conv) {
  return isOnIdeepDevice(conv) ? (conv.getGroup() <= 1) : false;
}

void resetConvForFusion(repr::NNGraph::NodeRef convNode, int fusion_type) {
  // Fusion types:
  // FUSION_CONV_RELU = 1
  // FUSION_CONV_SUM = 2
  // FUSION_CONV_SUM_RELU = 3
  auto conv = repr::nn::get<repr::Conv>(convNode);
  auto annotation = conv->getMutableAnnotation();
  if (!annotation || !isa<Caffe2Annotation>(annotation)) {
    return;
  }

  auto* op = getMutableOpDef(*conv);
  if (op == nullptr) {
    return;
  }

  if (op->type() == "ConvFusion") {
    CAFFE_ENFORCE(fusion_type == 1, "Invalid nest fusion");
    for (auto& arg : *op->mutable_arg()) {
      if (arg.name() == "fusion_type") {
        // Only from FUSION_CONV_SUM to FUSION_CONV_SUM_RELU
        CAFFE_ENFORCE(arg.i() == 2, "Invalid nest fusion");
        arg.set_i(3);
        return;
      }
    }
    return;
  }

  CAFFE_ENFORCE(fusion_type < 3, "Invalid fusion type");
  op->set_type("ConvFusion");
  auto* arg = op->add_arg();
  arg->set_name("fusion_type");
  arg->set_i(fusion_type);
}

bool fuseConvBNAndAffChHelperForIdeep(repr::NNModule* nn, caffe2::Workspace* ws) {
  auto isAffineChannelNode = [](const repr::NNGraph::NodeRef& node) {
    if (!repr::nn::is<repr::NeuralNetOperator>(node)) {
      return false;
    }
    auto maybeAffCh = repr::nn::get<repr::NeuralNetOperator>(node);
    auto maybeAffChDef = getOpDef(*maybeAffCh);
    return maybeAffChDef.type() == "AffineChannel";
  };

  for (auto node_pair : repr::nn::dataIterator<repr::Conv>(nn->dataFlow)) {
    bool no_bias = false;
    repr::NNGraph::NodeRef convNode;
    repr::Conv* conv;
    std::tie(conv, convNode) = node_pair;

    if (!isOnIdeepDevice(*conv)) {
      LOG(WARNING) << "Not a IDEEP operator";
      continue;
    }

    const auto& op = getOpDef(*conv);
    if (op.type() == "ConvFusion") {
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
    } else if (isAffineChannelNode(consumer)) {
      isBN = false;
    } else {
      continue;
    }
    auto bnOrAffChNode = consumer;
    auto bn = isBN ? repr::nn::get<repr::BatchNormalization>(bnOrAffChNode) : nullptr;
    auto bnOrAffChOutput = repr::nn::getOutputs(bnOrAffChNode).front();

    auto convInputs = repr::nn::getInputs(convNode);
    if (convInputs.size() < 2) {
      LOG(WARNING) << "Invalid convolution input size";
      continue;
    }

    auto bnOrAffChInputs = repr::nn::getInputs(bnOrAffChNode);
    int numInputs = isBN ? 5 : 3;
    if (bnOrAffChInputs.size() < numInputs) {
      LOG(WARNING) << "Invalid input size: "
                   << bnOrAffChInputs.size()
                   << ", expect " << numInputs;
      continue;
    }

    // When no bias, borrow BN bias
    if (convInputs.size() < 3) {
      no_bias = true;
      nn->dataFlow.createEdge(bnOrAffChInputs[2], convNode);
      convInputs = repr::nn::getInputs(convNode);
    }

#define EXPOSE_TENSOR_DATA(name, index, nodes, need_init)                \
  itensor* name = nullptr;                                               \
  itensor name##Tensor;                                                  \
  float* name##Data = nullptr;                                           \
  if (need_init) {                                                       \
    name = getTensor<itensor>(getBlob(nodes[index], ws));                \
    if (name == nullptr) {                                               \
      LOG(WARNING) << #name " not a IDEEP tensor";                       \
      continue;                                                          \
    }                                                                    \
    name##Tensor.resize(name->get_dims(), name->get_data_type());        \
    name##Tensor.reorder_from(*name);                                    \
    CAFFE_ENFORCE(                                                       \
      name##Tensor.is_public_format(), #name " not with public format"); \
    name##Data = static_cast<float*>(name##Tensor.get_data_handle());    \
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

    filter->reorder_from(filterTensor);
    biasConv->reorder_from(biasConvTensor);
    nn->dataFlow.replaceNode(convOutput, bnOrAffChOutput);

    nn->dataFlow.deleteNode(bnOrAffChNode);
    nn->dataFlow.deleteNode(convOutput);

    return true;
  }

  return false;
}

void fuseConvBNAndAffChForIdeep(repr::NNModule* nn, caffe2::Workspace* ws) {
  while (fuseConvBNAndAffChHelperForIdeep(nn, ws)) {
  }
}

void fuseConvSumForIdeep(repr::NNModule* nn, caffe2::Workspace* ws) {
  // Assume the order of nodes from getMutableNodes conforms to
  // the original topo order of operators
  auto allNodes = nn->dataFlow.getMutableNodes();
  for (int i = 0; i < allNodes.size(); i++) {
    auto sumNode = allNodes[i];
    if (!repr::nn::hasInputs(sumNode)) {
      continue;
    }

    if (!repr::nn::is<repr::Sum>(sumNode)) {
      continue;
    }

    auto sum = repr::nn::get<repr::Sum>(sumNode);
    if (!isOnIdeepDevice(*sum)) {
      LOG(WARNING) << "Not a IDEEP operator";
      continue;
    }

    auto sumInputs = repr::nn::getInputs(sumNode);
    if (sumInputs.size() != 2) {
      continue;
    }

    bool should_fuse = true;
    for (auto input : sumInputs) {
      auto consumer = repr::nn::getConsumers(input).back();
      if (consumer != sumNode) {
        should_fuse = false;
        break;
      }
    }
    // Sum inputs should not be referenced by sequential ops.
    if (!should_fuse) {
      continue;
    }

    int j = i - 1;
    repr::NNGraph::NodeRef convNode = nullptr;
    while (j-- >= 0) {
      if (!repr::nn::hasInputs(sumNode)) {
        continue;
      }

      // Find the nearest Op before Sum
      if (repr::nn::is<repr::NeuralNetOperator>(allNodes[j])) {
        // The Op must be a Conv
        if (repr::nn::is<repr::Conv>(allNodes[j])) {
          convNode = allNodes[j];
        }
        break;
      }
    }
    if (convNode == nullptr) {
      continue;
    }

    auto conv = repr::nn::get<repr::Conv>(convNode);
    if (!shouldFuseConv(*conv)) {
      LOG(WARNING) << "Not a IDEEP operator";
      continue;
    }

    auto convOutput = repr::nn::getOutputs(convNode).front();
    repr::NNGraph::NodeRef sumInputX =
        (sumInputs[0] == convOutput ? sumInputs[1] : sumInputs[0]);
    CAFFE_ENFORCE(sumInputX != nullptr, "Invalid sum inputs");

    auto preNode = repr::nn::getProducer(sumInputX);
    if (preNode == nullptr || !repr::nn::is<repr::NeuralNetOperator>(preNode)) {
      LOG(WARNING) << "Can not fuse Conv Sum";
      continue;
    }

    auto newOutputName = repr::nn::get<repr::Tensor>(sumInputX)->getName();
    auto newOutputTensor = util::make_unique<repr::Tensor>(newOutputName);
    auto newOutput = nn->dataFlow.createNode(
        unique_dyn_cast<repr::NeuralNetData>(newOutputTensor));

    auto sumOutput = repr::nn::getOutputs(sumNode).front();
    nn->dataFlow.replaceNode(sumOutput, newOutput);

    // 2 means FUSION_CONV_SUM
    resetConvForFusion(convNode, 2);
    nn->dataFlow.createEdge(sumInputX, convNode);
    nn->dataFlow.createEdge(convNode, newOutput);

    nn->dataFlow.deleteNode(sumNode);
    nn->dataFlow.deleteNode(sumOutput);
    nn->dataFlow.deleteNode(convOutput);
  }
}

void fuseActivationForIdeep(repr::NNModule* nn) {
  // Conv+Relu fusion
  auto should_fuse = shouldFuseConv;
  auto postprocess = std::bind(resetConvForFusion, std::placeholders::_1, 1);
  fuseActivation<repr::Conv, repr::Relu>(nn, should_fuse, postprocess);
}

void enforceFusionInplaceForIdeep(repr::NNModule* nn) {
  // For fusions of Conv+Sum or Conv+Sum+ReLU, the last input and output must
  // be inplaced. To enforce inplace, here to re-check whole graph and correct
  // the ConvFusion Ops.
  for (auto node_pair : repr::nn::dataIterator<repr::Conv>(nn->dataFlow)) {
    repr::NNGraph::NodeRef convNode;
    repr::Conv* conv;
    std::tie(conv, convNode) = node_pair;

    if (!isOnIdeepDevice(*conv)) {
      LOG(WARNING) << "Not a IDEEP operator";
      continue;
    }

    const auto& op = getOpDef(*conv);
    if (op.type() != "ConvFusion") {
      continue;
    }

    bool enforce_inplace = false;
    for (const auto& arg : op.arg()) {
      // Only check FUSION_SUM & FUSION_SUM_RELU
      if (arg.name() == "fusion_type" && (arg.i() == 2 || arg.i() == 3)) {
        enforce_inplace = true;
        break;
      }
    }

    if (!enforce_inplace) {
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
      return;
    }

    auto newOutputTensor = util::make_unique<repr::Tensor>(inputName);
    auto newOutput = nn->dataFlow.createNode(
        unique_dyn_cast<repr::NeuralNetData>(newOutputTensor));
    nn->dataFlow.replaceNode(convOutput, newOutput);

    nn->dataFlow.deleteNode(convOutput);
  }
}

void setPoolingInferenceMode(repr::NNModule *nn) {
  for (auto node_pair : repr::nn::dataIterator<repr::MaxPool>(nn->dataFlow)) {
    repr::NNGraph::NodeRef maxPoolNode;
    repr::MaxPool *maxPool;
    std::tie(maxPool, maxPoolNode) = node_pair;

    if (!isOnIdeepDevice(*maxPool)) {
      LOG(WARNING) << "Not a IDEEP operator";
      continue;
    }

    auto *op = getMutableOpDef(*maxPool);
    bool found_training_mode = false;
    for (auto &arg : *op->mutable_arg()) {
      if (arg.name() == "training_mode") {
        arg.set_i(0);
        found_training_mode = true;
        break;
      }
    }

    if (!found_training_mode) {
      auto *arg = op->add_arg();
      arg->set_name("training_mode");
      arg->set_i(0);
    }
  }
}

void OptimizeForIdeep(
    repr::NNModule* nn,
    caffe2::Workspace* ws,
    bool training_mode) {
  if (training_mode) {
    // Only support inference so far
    return;
  }

  fuseConvBNAndAffChForIdeep(nn, ws);

  fuseConvSumForIdeep(nn, ws);

  fuseActivationForIdeep(nn);

  enforceFusionInplaceForIdeep(nn);

  setPoolingInferenceMode(nn);
}

#endif // CAFFE2_USE_MKLDNN

} // namespace opt
} // namespace caffe2
