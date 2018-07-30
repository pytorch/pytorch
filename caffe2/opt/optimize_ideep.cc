#include "caffe2/opt/optimize_ideep.h"
#include "caffe2/opt/converter.h"

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

Blob *renameBlob(const std::string &old_name,
    const std::string &new_name, caffe2::Workspace *ws) {
  CAFFE_ENFORCE(!old_name.empty() && !new_name.empty(),
      "Invalide blob name");
  if (!ws->HasBlob(old_name) || ws->HasBlob(new_name)) return nullptr;
  return ws->RenameBlob(old_name, new_name);
}

Blob *getBlob(const std::string name, caffe2::Workspace *ws) {
  CAFFE_ENFORCE(ws->HasBlob(name), "Blob ", name, " not in workspace");
  return ws->GetBlob(name);
}

Blob *getBlob(repr::NNGraph::NodeRef node, caffe2::Workspace *ws) {
  auto tensor = repr::nn::get<repr::Tensor>(node);
  return getBlob(tensor->getName(), ws);
}

template <class T> T getTensor(Blob *blob) {
  CAFFE_ENFORCE(blob, "Blob is invalid");
  return blob->template Get<T>();
}

template <class T> T *getMutableTensor(Blob *blob) {
  CAFFE_ENFORCE(blob, "Blob is invalid");
  if (blob && blob->template IsType<T>()) {
    return blob->template GetMutable<T>();
  }
  return nullptr;
}

const caffe2::OperatorDef &getOpDef(const repr::NeuralNetOperator &nnOp) {
  auto annotation = nnOp.getAnnotation();
  if (annotation == nullptr) {
    CAFFE_THROW("Cannot get Operator annotation");
  }
  return dyn_cast<Caffe2Annotation>(annotation)->getOperatorDef();
}

caffe2::OperatorDef *getMutableOpDef(repr::NeuralNetOperator &nnOp) {
  auto annotation = nnOp.getMutableAnnotation();
  if (annotation == nullptr) {
    CAFFE_THROW("Cannot get Operator annotation");
  }
  return dyn_cast<Caffe2Annotation>(annotation)->getMutableOperatorDef();
}

bool isOnIdeepDevice(const repr::NeuralNetOperator &nnOp) {
  // We only want to fuse for IDEEP convs
  const auto &op = getOpDef(nnOp);
  return op.device_option().device_type() == DeviceTypeProto::PROTO_IDEEP;
}

bool shouldFuseConv(const repr::Conv& conv) {
  return isOnIdeepDevice(conv);
}

bool isConvFusion(repr::NNGraph::NodeRef convNode, int fusion_type) {
  auto conv = repr::nn::get<repr::Conv>(convNode);
  auto &op = getOpDef(*conv);

  if (op.type() == "ConvFusion") {
    for (const auto &arg : op.arg()) {
      if (arg.name() == "fusion_type") {
        return arg.i() == fusion_type;
      }
    }
  }

  return false;
}

void resetConvForFusion(repr::NNGraph::NodeRef convNode, int fusion_type) {
  auto conv = repr::nn::get<repr::Conv>(convNode);
  auto *op = getMutableOpDef(*conv);
  if (op == nullptr) {
    return;
  }

  if (op->type() == "ConvFusion") {
    CAFFE_ENFORCE(fusion_type == FUSION_CONV_RELU, "Invalid nest fusion");
    for (auto &arg : *op->mutable_arg()) {
      if (arg.name() == "fusion_type") {
        // Only from FUSION_CONV_SUM to FUSION_CONV_SUM_RELU
        CAFFE_ENFORCE(arg.i() == FUSION_CONV_SUM, "Invalid nest fusion");
        arg.set_i(FUSION_CONV_SUM_RELU);
        return;
      }
    }
    CAFFE_THROW("Can not find fusion type in ConvFusion");
  }

  CAFFE_ENFORCE(fusion_type < FUSION_CONV_SUM_RELU, "Invalid fusion type");
  op->set_type("ConvFusion");
  auto* arg = op->add_arg();
  arg->set_name("fusion_type");
  arg->set_i(fusion_type);
}

void reconnectStatInfo(
    caffe2::Workspace *ws,
    repr::NeuralNetOperator *srcOp,
    std::string srcScaleName,
    repr::NeuralNetOperator *dstOp,
    std::string dstScaleName) {
  if (srcOp == nullptr || dstOp == nullptr)
    return;
  if (srcOp == dstOp && srcScaleName == dstScaleName)
    return;
  if (dstScaleName.empty())
    return;

  auto *dst = getMutableOpDef(*dstOp);
  auto &dst_args = *dst->mutable_arg();
  auto remove_arg = [&](decltype(dst_args) &args, std::string &name) {
    for (auto it = args.begin(); it != args.end(); it ++) {
      if (it->name() == name) {
        args.erase(it);
        return true;
      }
    }
    return false;
  };
  while(remove_arg(dst_args, dstScaleName));

  if (srcScaleName.empty())
    return;

  auto &src = getOpDef(*srcOp);
  auto &src_args = src.arg();
  auto src_it = src_args.begin();
  for (; src_it != src_args.end(); src_it ++) {
    if (src_it->name() == srcScaleName) break;
  }
  if (src_it == src_args.end()) return;

  auto *arg = dst->add_arg();
  *arg = *src_it;
  arg->set_name(dstScaleName);
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

    const auto &convOp = getOpDef(*conv);
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
    name = getMutableTensor<itensor>(getBlob(nodes[index], ws));         \
    if (name == nullptr) {                                               \
      LOG(WARNING) << #name " not a IDEEP tensor";                       \
      continue;                                                          \
    }                                                                    \
    name##Tensor.resize(name->get_dims(), name->get_data_type());        \
    name##Tensor.feed_from(*name);                                       \
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

    filter->feed_from(filterTensor);
    biasConv->feed_from(biasConvTensor);
    nn->dataFlow.replaceNode(convOutput, bnOrAffChOutput);
    reconnectStatInfo(ws, bn, IDEEP_ABSMAX_O(0), conv, IDEEP_ABSMAX_O(0));

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
    if (preNode == nullptr ||
        !repr::nn::is<repr::NeuralNetOperator>(preNode)) {
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
    reconnectStatInfo(ws, sum, IDEEP_ABSMAX_O(0), conv, IDEEP_ABSMAX_O(0));

    nn->dataFlow.createEdge(sumInputX, convNode);
    nn->dataFlow.createEdge(convNode, newOutput);

    nn->dataFlow.deleteNode(sumNode);
    nn->dataFlow.deleteNode(sumOutput);
    nn->dataFlow.deleteNode(convOutput);
  }
}

void fuseActivationForIdeep(repr::NNModule *nn, caffe2::Workspace *ws) {
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
    if (!shouldFuseConv(*conv)) {
      continue;
    }

    reconnectStatInfo(ws, relu, IDEEP_ABSMAX_O(0), conv, IDEEP_ABSMAX_O(0));

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
          auto co_name = repr::nn::get<repr::Tensor>(consumer_output)->getName();
          if (co_name == output_tensor->getName()) {
            rectify_inplace = true;
          }
        }
      }
      if (rectify_inplace) {
        auto new_output = nn->dataFlow.createNode(
            make_unique<repr::Tensor>(output_tensor->getName() + "_fusion_fix"));
        nn->dataFlow.replaceNode(output_node, new_output);
      }
    }

    resetConvForFusion(conv_node, 1);
  }
}

void enforceFusionInplaceForIdeep(repr::NNModule *nn, caffe2::Workspace *ws) {
  // For fusions of Conv+Sum or Conv+Sum+ReLU, the last input and output must
  // be inplaced. To enforce inplace, here to re-check whole graph and correct
  // the ConvFusion Ops.
  for (auto node_pair : repr::nn::dataIterator<repr::Conv>(nn->dataFlow)) {
    repr::NNGraph::NodeRef convNode;
    repr::Conv *conv;
    std::tie(conv, convNode) = node_pair;

    if (!isOnIdeepDevice(*conv)) {
      LOG(WARNING) << "Not a IDEEP operator";
      continue;
    }

    const auto &op = getOpDef(*conv);
    if (op.type() != "ConvFusion") {
      continue;
    }

    // Only check FUSION_SUM & FUSION_SUM_RELU
    if(!isConvFusion(convNode, FUSION_CONV_SUM)
        && !isConvFusion(convNode, FUSION_CONV_SUM_RELU)) {
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
  auto setTrainingMode = [](repr::NeuralNetOperator &pool) {
    if (!isOnIdeepDevice(pool)) {
      LOG(WARNING) << "Not a IDEEP operator";
      return;
    }
    auto *op = getMutableOpDef(pool);
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
  };

  for (auto node_pair : repr::nn::dataIterator<repr::MaxPool>(nn->dataFlow)) {
    repr::NNGraph::NodeRef maxPoolNode;
    repr::MaxPool *maxPool;
    std::tie(maxPool, maxPoolNode) = node_pair;
    setTrainingMode(*maxPool);
  }

  for (auto node_pair : repr::nn::dataIterator<repr::AveragePool>(nn->dataFlow)) {
    repr::NNGraph::NodeRef avgPoolNode;
    repr::AveragePool *avgPool;
    std::tie(avgPool, avgPoolNode) = node_pair;
    setTrainingMode(*avgPool);
  }
}

// Pre-convert filters format to expected one here
// in order to avoid boring conversions during computations
void preConvertFiltersFormat(repr::NNModule *nn, caffe2::Workspace *ws) {
  for (auto &node : nn->dataFlow.getMutableNodes()) {
    if (!repr::nn::is<repr::Conv>(node) && !repr::nn::is<repr::FC>(node)) {
      continue;
    }

    auto *nnOp = repr::nn::get<repr::NeuralNetOperator>(node);
    if (!isOnIdeepDevice(*nnOp)) {
      LOG(WARNING) << "Not a IDEEP operator";
      continue;
    }

    auto inputs = repr::nn::getInputs(node);
    if (inputs.size() < 2) {
      LOG(WARNING) << "Invalid input size";
      continue;
    }

    auto *filterBlob = getBlob(inputs[1], ws);
    auto *filter = getMutableTensor<itensor>(filterBlob);
    if (filter == nullptr) {
      CAFFE_ENFORCE(filter, "Filter not a IDEEP tensor");
      continue;
    }

    itensor::descriptor expectedDesc;
    if (repr::nn::is<repr::Conv>(node)) {
      auto conv = repr::nn::get<repr::Conv>(node);
      auto initValue = [](vector<int>& v, vector<int> i) {
        if (v.empty()) v = i; };
      auto strides = conv->getStrides();
      initValue(strides, {1, 1});
      auto pads = conv->getPads();
      initValue(pads, {0, 0, 0, 0});
      auto dilations = conv->getDilations();
      initValue(dilations, {1, 1});

      iscale filterScale;
      auto *op = getMutableOpDef(*conv);
      auto aalgorithm = ialgo::convolution_direct;
      for (auto &arg : *op->mutable_arg()) {
        if ((arg.name() == "conv_algorithm")
            && (arg.i() == CONV_ALGORITHM_WINOGRAD)) {
          aalgorithm = ialgo::convolution_winograd;
        } else if (arg.name() == "need_quantize" && arg.i() != 0) {
          filterScale = filter->calculate_scale(idtype::s8, 0);
        }
      }
      auto dataType = filterScale.empty() ? filter->get_data_type() : idtype::s8;

      filter->make_group(conv->getGroup());
      expectedDesc = ideep::convolution_forward::expected_weights_descriptor(
          filter->get_dims(), dataType, strides, {pads[0], pads[1]},
          {pads[2], pads[3]}, dilations, conv->getGroup(), aalgorithm);

      if (filter->get_descriptor() != expectedDesc) {
        itensor *newFilter = new itensor(expectedDesc);
        if (filterScale.empty()) {
          ideep::reorder::compute(*filter, *newFilter);
        } else {
          int mask = (filterScale.size() > 1)
            ? ((conv->getGroup() > 1) ? 3 : 1) : 0;
          ideep::reorder::compute(filter->as_weights(),
              *newFilter, {mask, filterScale});
          newFilter->set_scale(filterScale);
        }
        filterBlob->Reset(newFilter);
      }
    // convert weights for FC
    } else if (repr::nn::is<repr::FC>(node)) {
      auto fc = repr::nn::get<repr::FC>(node);
      auto axis_w = fc->getAxisW();
      if (axis_w != 1) {
        auto f_dims = filter->get_dims();
        auto f_dim0 = std::accumulate(f_dims.begin(), f_dims.begin() + axis_w,
            1, std::multiplies<itensor::dim_t>());
        auto f_dim1 = std::accumulate(f_dims.begin() + axis_w, f_dims.end(),
            1, std::multiplies<itensor::dim_t>());
        filter->reshape({f_dim0, f_dim1});
      }

      expectedDesc = ideep::inner_product_forward::expected_weights_descriptor(
          filter->get_dims());

      if (filter->get_descriptor() != expectedDesc) {
        itensor *newFilter = new itensor(expectedDesc);
        ideep::reorder::compute(filter->as_weights(), *newFilter);
        filterBlob->Reset(newFilter);
      }
    }
  }
}

void OptimizeForIdeep(repr::NNModule *nn, caffe2::Workspace *ws,
                      bool training_mode) {
  if (training_mode) {
    // Only support inference so far
    return;
  }

  fuseConvBNAndAffChForIdeep(nn, ws);

  fuseConvSumForIdeep(nn, ws);

  fuseActivationForIdeep(nn, ws);

  enforceFusionInplaceForIdeep(nn, ws);

  setPoolingInferenceMode(nn);

  // Must be called behind all other optimizations
  preConvertFiltersFormat(nn, ws);
}

#endif // CAFFE2_USE_MKLDNN

} // namespace opt
} // namespace caffe2
