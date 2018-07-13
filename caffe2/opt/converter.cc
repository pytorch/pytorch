#include "caffe2/opt/converter.h"
#include "caffe2/core/logging.h"

#include "nomnigraph/Graph/Algorithms.h"

#include "nomnigraph/Support/Casting.h"
#include "nomnigraph/Support/Pointer.h"

using namespace nom;

namespace {

std::vector<int> getStrides(std::map<std::string, caffe2::Argument> argMap) {
  std::vector<int> strides;
  // TODO: include all the other ways of adding these args.
  // e.g. strides, stride_h, etc.
  if (argMap.count("stride")) {
    CAFFE_ENFORCE(argMap["stride"].has_i(), "Invalid stride argument");
    int stride = static_cast<int>(argMap["stride"].i());
    strides = {stride, stride};
  }
  return strides;
}

std::vector<int> getPads(std::map<std::string, caffe2::Argument> argMap) {
  std::vector<int> pads;
  if (argMap.count("pad")) {
    CAFFE_ENFORCE(argMap["pad"].has_i(), "Invalid pad argument");
    int pad = static_cast<int>(argMap["pad"].i());
    pads = {pad, pad, pad, pad};
  }
  return pads;
}

std::vector<int> getDilations(std::map<std::string, caffe2::Argument> argMap) {
  std::vector<int> dilations;
  if (argMap.count("dilation")) {
    CAFFE_ENFORCE(argMap["dilation"].has_i(), "Invalid dilation argument");
    int dilation = static_cast<int>(argMap["dilation"].i());
    dilations = {dilation, dilation};
  }
  return dilations;
}

} // namespace

namespace caffe2 {

CAFFE_DEFINE_REGISTRY(ConverterRegistry, Converter);

std::map<std::string, caffe2::Argument> Converter::getArgumentsFromOperator(
    caffe2::OperatorDef op) {
  std::map<std::string, caffe2::Argument> argMap;
  for (auto arg : op.arg()) {
    argMap[arg.name()] = arg;
  }
  return argMap;
}

repr::NeuralNetOperator::NNLayout getLayout(std::map<std::string, caffe2::Argument> argMap) {
  auto arg = argMap.find("order");
  if (arg != argMap.end()) {
    auto order = argMap["order"].s();
    if (order == "NCHW" || order == "nchw") {
      return repr::NeuralNetOperator::NNLayout::NCHW;
    } else if (order == "NHWC" || order == "nhwc") {
      return repr::NeuralNetOperator::NNLayout::NHWC;
    }
  }
  return repr::NeuralNetOperator::NNLayout::Undefined;
}

OperatorDef Converter::convertToOperatorDef(
    const nom::repr::NeuralNetOperator* nnOp) {
  auto* annotation = nnOp->getAnnotation();
  // Default to using the stored operator.
  if (isa<Caffe2Annotation>(annotation)) {
    return dyn_cast<Caffe2Annotation>(annotation)->getOperatorDef();
  }
  CAFFE_THROW("TODO: Cannot yet instantiate OperatorDef from nomnigraph");
}

std::vector<int> getKernelShape(std::map<std::string, caffe2::Argument> argMap) {
  // There are literally three ways to define shapes in Conv in Caffe2
  std::vector<int> kernelShape;
  if (argMap.count("kernel")) {
    CAFFE_ENFORCE(argMap["kernel"].has_i(), "Invalid kernel argument");
    int kernel = static_cast<int>(argMap["kernel"].i());
    kernelShape = {kernel, kernel};
  } else if (argMap.count("kernels")) {
    for (auto i : argMap["kernels"].ints()) {
      kernelShape.push_back(static_cast<int>(i));
    }
  } else if (argMap.count("kernel_h") && argMap.count("kernel_w")) {
    CAFFE_ENFORCE(argMap["kernel_h"].has_i(), "Invalid kernel argument");
    CAFFE_ENFORCE(argMap["kernel_w"].has_i(), "Invalid kernel argument");
    int kernelH = static_cast<int>(argMap["kernel_h"].i());
    int kernelW = static_cast<int>(argMap["kernel_w"].i());
    kernelShape = {kernelH, kernelW};
  }
  return kernelShape;
}

namespace {

class ConvConverter : public Converter {
  std::unique_ptr<nom::repr::NeuralNetOperator> convertToNeuralNetOperator(
      const OperatorDef& op) override {
    std::unique_ptr<repr::NeuralNetOperator> nnOp;
    auto argMap = getArgumentsFromOperator(op);
    auto kernelShape = getKernelShape(argMap);
    nnOp = util::make_unique<repr::Conv>(kernelShape);
    auto c = dyn_cast<repr::Conv>(nnOp.get());

    c->setStrides(getStrides(argMap));
    c->setPads(getPads(argMap));
    c->setDilations(getDilations(argMap));
    return nnOp;
  }
  // Does not override default converter to OperatorDef

  virtual ~ConvConverter() {}
};

REGISTER_CONVERTER(Conv, ConvConverter);

TRIVIAL_CONVERTER(Relu);
REGISTER_CONVERTER(Relu, ReluConverter);

TRIVIAL_CONVERTER(Sum);
REGISTER_CONVERTER(Sum, SumConverter);

TRIVIAL_CONVERTER(BatchNormalization);
REGISTER_CONVERTER(SpatialBN, BatchNormalizationConverter);

TRIVIAL_CONVERTER(Flatten);
REGISTER_CONVERTER(Flatten, FlattenConverter);

TRIVIAL_CONVERTER(BatchGather);
REGISTER_CONVERTER(BatchGather, BatchGatherConverter);

class AveragePoolConverter : public Converter {
  std::unique_ptr<nom::repr::NeuralNetOperator> convertToNeuralNetOperator(
      const OperatorDef& op) override {
    std::unique_ptr<repr::NeuralNetOperator> nnOp;
    auto argMap = getArgumentsFromOperator(op);
    auto kernelShape = getKernelShape(argMap);
    nnOp = util::make_unique<repr::AveragePool>(kernelShape);
    return nnOp;
  }
  // Does not override default converter to OperatorDef

  virtual ~AveragePoolConverter() {}
};
REGISTER_CONVERTER(AveragePool, AveragePoolConverter);

class MaxPoolConverter : public Converter {
  std::unique_ptr<nom::repr::NeuralNetOperator> convertToNeuralNetOperator(
      const OperatorDef& op) override {
    std::unique_ptr<repr::NeuralNetOperator> nnOp;
    auto argMap = getArgumentsFromOperator(op);
    auto kernelShape = getKernelShape(argMap);
    nnOp = util::make_unique<repr::MaxPool>(kernelShape);
    return nnOp;
  }
  // Does not override default converter to OperatorDef

  virtual ~MaxPoolConverter() {}
};
REGISTER_CONVERTER(MaxPool, MaxPoolConverter);

class ConcatConverter : public Converter {
  std::unique_ptr<nom::repr::NeuralNetOperator> convertToNeuralNetOperator(
      const OperatorDef& op) override {
    std::unique_ptr<repr::NeuralNetOperator> nnOp =
        util::make_unique<repr::Concat>();
    auto argMap = getArgumentsFromOperator(op);

    auto c = dyn_cast<repr::Concat>(nnOp.get());
    if (argMap.count("axis")) {
      CAFFE_ENFORCE(argMap["axis"].has_i(), "Invalid axis argument");
      int axis = static_cast<int>(argMap["axis"].i());
      c->setAxis(axis);
    }
    if (argMap.count("add_axis")) {
      CAFFE_ENFORCE(argMap["add_axis"].has_i(), "Invalid add_axis argument");
      int add_axis = static_cast<int>(argMap["add_axis"].i());
      c->setAddAxis(!!add_axis);
    }
    return nnOp;
  }
  // Does not override default converter to OperatorDef

  virtual ~ConcatConverter() {}
};
REGISTER_CONVERTER(Concat, ConcatConverter);

class BatchMatMulConverter : public Converter {
  std::unique_ptr<nom::repr::NeuralNetOperator> convertToNeuralNetOperator(
      const OperatorDef& op) override {
    std::unique_ptr<repr::NeuralNetOperator> nnOp =
        util::make_unique<repr::BatchMatMul>();
    auto argMap = getArgumentsFromOperator(op);

    auto c = dyn_cast<repr::BatchMatMul>(nnOp.get());
    if (argMap.count("trans_a")) {
      CAFFE_ENFORCE(argMap["trans_a"].has_i(), "Invalid axis argument");
      int trans_a = static_cast<int>(argMap["trans_a"].i());
      c->setTransA(!!trans_a);
    }
    if (argMap.count("trans_b")) {
      CAFFE_ENFORCE(argMap["trans_b"].has_i(), "Invalid add_axis argument");
      int trans_b = static_cast<int>(argMap["trans_b"].i());
      c->setTransB(!!trans_b);
    }
    if (argMap.count("broadcast")) {
      CAFFE_ENFORCE(argMap["broadcast"].has_i(), "Invalid add_axis argument");
      int broadcast = static_cast<int>(argMap["broadcast"].i());
      c->setBroadcast(!!broadcast);
    }
    return nnOp;
  }
  // Does not override default converter to OperatorDef

  virtual ~BatchMatMulConverter() {}
};
REGISTER_CONVERTER(BatchMatMul, BatchMatMulConverter);

} // namespace

std::unique_ptr<repr::NeuralNetOperator> convertToNeuralNetOperator(
    const caffe2::OperatorDef& op) {
  auto argMap = Converter::getArgumentsFromOperator(op);

  std::unique_ptr<repr::NeuralNetOperator> nnOp;

  if (ConverterRegistry()->Has(op.type())) {
    nnOp =
        ConverterRegistry()->Create(op.type())->convertToNeuralNetOperator(op);
  }

  if (!nnOp) {
    nnOp = util::make_unique<repr::GenericOperator>(op.type());
  }

  // Generic attributes associated with Ops here
  nnOp->setLayout(getLayout(argMap));

  auto annotation = util::make_unique<Caffe2Annotation>();
  annotation->setOperatorDef(op);

  auto device_name = op.device_option().node_name();
  if (device_name != "") {
    annotation->setDevice(device_name);
  }
  annotation->setDeviceType(op.device_option().device_type());

  nnOp->setAnnotation(std::move(annotation));

  return nnOp;
}

void handleWhileOp(
    repr::NNGraph& dfg,
    repr::NNCFGraph& cfg,
    repr::NNGraph::NodeRef& opNode,
    repr::NNCFGraph::NodeRef& bbNode,
    OperatorDef& op,
    std::unordered_map<std::string, repr::NNGraph::NodeRef>& blobMap
) {
  opNode->resetData(util::make_unique<repr::While>());
  auto argMap = Converter::getArgumentsFromOperator(op);
  std::string bodyNetSerialized = argMap["body"].s();
  auto bodyNet = caffe2::NetDef();
  bodyNet.ParseFromString(bodyNetSerialized);

  std::unordered_map<std::string, repr::NNGraph::NodeRef> bodyBlobMap;
  auto bodyNN = convertToNNModule(bodyNet, &bodyBlobMap);
  repr::NNGraph bodyGraph = std::move(bodyNN.dataFlow);
  repr::NNCFGraph bodyCFGraph = std::move(bodyNN.controlFlow);

  auto rev_sorted = algorithm::tarjans(&bodyGraph);

  for (auto& k : bodyBlobMap) {
    auto name = k.first;
    if (blobMap.count(name)) {
      auto oldNode = blobMap[name];
      printf("Exit tensor %s is in the parent scope, inserting Phi node...\n", k.first.c_str());
      auto phiNode = dfg.createNode(util::make_unique<repr::NNPhi>()); // NN variant of a Phi node
      // Clone the operator.
      auto tensor = dyn_cast<repr::NeuralNetData>(blobMap[name]->data().get());
      auto* clonedTensor = tensor->clone();
      auto phiOut = dfg.createNode(std::unique_ptr<repr::NeuralNetData>(clonedTensor));
      dfg.createEdge(phiNode, phiOut);
      dfg.createEdge(oldNode, phiNode);
      dfg.createEdge(bodyBlobMap[name], phiNode);
      blobMap[name] = phiOut;
      for (auto& inEdge : opNode->getInEdges()) {
        if (inEdge->tail() == oldNode) {
          dfg.deleteEdge(inEdge);
          dfg.createEdge(phiOut, opNode);
        }
      }
    }
  }

  // Dependencies simply have no producers
  std::unordered_map<repr::NNGraph::NodeRef, repr::NNGraph::NodeRef> inNodeMap;
  for (auto& n : bodyGraph.getMutableNodes()) {
    if (!isa<repr::NeuralNetData>(n->data())) { continue; }
    if (n->getInEdges().size() == 0) {
      auto name = dyn_cast<repr::NeuralNetData>(n->data().get())->getName();
      // TODO(bwasti): this may be needed, depending on constraints
      //assert(blobMap.count(name) != 0 && "Loop body takes undefined dependency.");
      if (blobMap.count(name)) {
        inNodeMap[n] = blobMap[name];
      }
    }
  }

  CAFFE_ENFORCE(rev_sorted.front().getNodes().size() == 1,
      "More than one exit node.");
  CAFFE_ENFORCE(rev_sorted.back().getNodes().size() == 1,
      "More than one entry node.");

  auto exit_tensor = *(rev_sorted.front().getNodes().begin());
  CAFFE_ENFORCE(isa<repr::NeuralNetData>(exit_tensor->data()),
      "Exit node is not a tensor.");

  auto bodyNodes = bodyGraph.getMutableNodes();
  auto bodyEdges = bodyGraph.getMutableEdges();

  for (auto node : bodyNodes) {
    bodyGraph.importNode(node, dfg);
  }

  for (auto edge : bodyEdges) {
    bodyGraph.importEdge(edge, dfg);
  }

  // Merge all dependencies
  for (auto node : dfg.getMutableNodes()) {
    if (inNodeMap.count(node)) {
      dfg.replaceNode(node, inNodeMap[node]);
      dfg.deleteNode(node);
    }
  }

  for (const auto& inEdge : opNode->getInEdges()) {
    auto* inputData = dyn_cast<repr::NeuralNetData>(inEdge->tail()->data().get());
    auto* exitData = dyn_cast<repr::NeuralNetData>(exit_tensor->data().get());
    if (inputData->getName() == exitData->getName()) {
      dfg.replaceNode(exit_tensor, inEdge->tail());
      dfg.deleteNode(exit_tensor);
    }
  }

  // CFG Handling
  auto bodyCFNodes = bodyCFGraph.getMutableNodes();
  auto bodyCFEdges = bodyCFGraph.getMutableEdges();

  // Create a while loop CFG node.
  auto whileBasicBlock = util::make_unique<repr::BasicBlockType<repr::NNGraph>>();
  for (auto& inEdge : opNode->getInEdges()) {
    auto node = inEdge->tail();
    for (auto& parentInEdge : node->getInEdges()) {
      auto parentNode = parentInEdge->tail();
      if (isa<repr::Phi>(parentNode->data().get())) {
        whileBasicBlock->pushInstructionNode(parentNode);
      }
    }
  }
  whileBasicBlock->pushInstructionNode(opNode);

  auto whileCFNode = cfg.createNode(std::move(whileBasicBlock));
  cfg.createEdge(bbNode, whileCFNode, 0);

  // The true path executes the body of the loop, so we
  // take that BB and point to it.
  for (auto cfNode : bodyCFNodes) {
    bodyCFGraph.importNode(cfNode, cfg);
    // If the CFG node has no children, we loop back to the top of the
    // while loop.
    if (cfNode->getOutEdges().size() == 0) {
      cfg.createEdge(cfNode, whileCFNode, 0);
    }
    // TODO check for a single entry point
    if (cfNode->getInEdges().size() == 0) {
      cfg.createEdge(whileCFNode, cfNode, 1);
    }
  }
  for (auto cfEdge : bodyCFEdges) {
    bodyCFGraph.importEdge(cfEdge, cfg);
  }

  // Now create the false case.
  bbNode =
    cfg.createNode(util::make_unique<repr::BasicBlockType<repr::NNGraph>>());
  cfg.createEdge(whileCFNode, bbNode, -1);
}


/// \brief Ingest a caffe2 protobuf model and output an NNModule.
/// \param net The caffe2 protobuf NetDef
/// \param blobMap [optional][output] A pointer to a blobMap to be populated with all the output blobs of the NetDef by name->NodeRef
repr::NNModule convertToNNModule(caffe2::NetDef &net, std::unordered_map<std::string, repr::NNGraph::NodeRef>* blobMapOut) {
  repr::NNGraph dfg;
  repr::NNCFGraph cfg;
  /// \brief We keep track of the producer of the blob.
  /// Because Caffe2 Nets are really just ordered operations
  /// we can just keep track of the most recent producer of
  /// a blob and draw and edge from that to any consumer we
  /// come by. If a new operator produces the blob we simply
  /// replace it in this map.
  std::unordered_map<std::string, repr::NNGraph::NodeRef> blobMap;

  /// \brief For the construction of the control flow graph we keep track
  /// of a current basic block, which we split up as we come accross control
  /// flow operations such as if and while.
  // std::unique_ptr<repr::BasicBlockType<repr::NNGraph>> currentBasicBlock =
  auto bbNode =
      cfg.createNode(util::make_unique<repr::BasicBlockType<repr::NNGraph>>());

  for (auto &op : *net.mutable_op()) {
    auto opNode = dfg.createNode(); // Create an empty node for the operator.
    // First calculate in-edges (data dependencies).
    for (const auto &input : op.input()) {
      // If we've never seen this tensor, make one.
      if (!blobMap.count(input)) {
        auto tensor = util::make_unique<repr::Tensor>(input);
        blobMap[input] =
            dfg.createNode(unique_dyn_cast<repr::NeuralNetData>(tensor));
      }

      auto tensorNode = blobMap[input];
      dfg.createEdge(tensorNode, opNode);
    }

    // Then save outputs into the blobMap for later consumption.
    for (const auto &output : op.output()) {
      auto tensor = util::make_unique<repr::Tensor>(output);
      auto tensorNode =
          dfg.createNode(unique_dyn_cast<repr::NeuralNetData>(tensor));
      dfg.createEdge(opNode, tensorNode);
      blobMap[output] = tensorNode;
    }

    if (op.type() == "While") {
      handleWhileOp(dfg, cfg, opNode, bbNode, op, blobMap);
    } else {
      opNode->resetData(convertToNeuralNetOperator(op));
      auto currentBasicBlock = bbNode->mutableData()->get();
      currentBasicBlock->pushInstructionNode(opNode);
    }
  }

  repr::NNModule module;
  module.dataFlow = std::move(dfg);
  module.controlFlow = std::move(cfg);
  if (blobMapOut) {
    *blobMapOut = blobMap;
  }
  return module;
}

caffe2::OperatorDef convertToOperatorDef(
    const repr::NNGraph::NodeRef& instrNode) {
  auto *nnOp = repr::nn::get<repr::NeuralNetOperator>(instrNode);
  auto *annotation = nnOp->getAnnotation();
  caffe2::OperatorDef op;

  if (ConverterRegistry()->Has(op.type())) {
    op = ConverterRegistry()->Create(op.type())->convertToOperatorDef(nnOp);
  } else if (!annotation) {
    op.set_type(nnOp->getName());
  } else {
    if (isa<Caffe2Annotation>(annotation)) {
      auto c2_annotation = dyn_cast<Caffe2Annotation>(annotation);
      op = c2_annotation->getOperatorDef();
      op.mutable_device_option()->set_device_type(
          c2_annotation->getDeviceType());
    } else {
      CAFFE_THROW(
          "Couldn't convert operator annotation to Caffe2 operator def");
    }
  }

  // We may have swapped out some of the edges.
  op.clear_input();
  op.clear_output();
  return op;
}

caffe2::NetDef convertToCaffe2Proto(repr::NNModule &m) {
  auto predictNet = caffe2::NetDef();
  return convertToCaffe2Proto(m, predictNet);
}

caffe2::NetDef convertToCaffe2Proto(repr::NNModule &m, const caffe2::NetDef& oldNet) {
  auto predictNet = caffe2::NetDef();
  // We copy the old net rather than mutate it.
  predictNet.CopyFrom(oldNet);
  predictNet.mutable_op()->Clear();

  repr::nn::coalesceInsertedDataDependencies(&m);

  // Simply iterate through the CFG and populate data dependencies
  // with the DFG
  for (const auto &bbNode : m.controlFlow.getMutableNodes()) {
    if (bbNode->getOutEdges().size() > 1) {
      CAFFE_THROW("Control flow not yet supported in Caffe2 converter.");
    }
    auto bb = bbNode->data().get();
    for (const auto &instrNode : bb->getInstructions()) {
      caffe2::OperatorDef op = convertToOperatorDef(instrNode);

      for (const auto &inEdge : instrNode->getInEdges()) {
        auto *tensorNode =
            dyn_cast<repr::NeuralNetData>(inEdge->tail()->data().get());
        *op.add_input() = tensorNode->getName();
      }
      for (const auto &outEdge : instrNode->getOutEdges()) {
        auto *tensorNode =
            dyn_cast<repr::NeuralNetData>(outEdge->head()->data().get());
        *op.add_output() = tensorNode->getName();
      }

      auto *nnOp = repr::nn::get<repr::NeuralNetOperator>(instrNode);
      if (nnOp->getLayout() != repr::NeuralNetOperator::NNLayout::Undefined) {

        caffe2::Argument* arg = nullptr;
        for (int i = 0; i < op.arg_size(); ++i) {
          auto arg_ = op.mutable_arg(i);
          if (arg_->name() == "order") {
            arg = arg_;
            break;
          }
        }

        if (!arg) {
          arg = op.add_arg();
          arg->set_name("order");
        }

        auto layout = nnOp->getLayout();
        if (layout == repr::NeuralNetOperator::NNLayout::NCHW) {
          arg->set_s("NCHW");
        }
        if (layout == repr::NeuralNetOperator::NNLayout::NHWC) {
          arg->set_s("NHWC");
        }
      }

      // Save the operator to the net.
      *predictNet.add_op() = op;
    }
  }

  return predictNet;
}

} // namespace caffe2
