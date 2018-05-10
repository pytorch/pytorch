#include "caffe2/opt/converter.h"
#include "nomnigraph/Graph/Algorithms.h"

#include "nomnigraph/Support/Casting.h"
#include "nomnigraph/Support/Pointer.h"

using namespace nom;

std::map<std::string, caffe2::Argument>
getArgumentsFromOperator(caffe2::OperatorDef op) {
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

std::vector<int> getKernelShape(std::map<std::string, caffe2::Argument> argMap) {
  // There are literally three ways to define shapes in Conv in Caffe2
  std::vector<int> kernelShape;
  if (argMap.count("kernel")) {
    assert(argMap["kernel"].has_i() && "Invalid kernel argument");
    int kernel = static_cast<int>(argMap["kernel"].i());
    kernelShape = {kernel, kernel};
  } else if (argMap.count("kernels")) {
    for (auto i : argMap["kernels"].ints()) {
      kernelShape.push_back(static_cast<int>(i));
    }
  } else if (argMap.count("kernel_h") && argMap.count("kernel_w")) {
    assert(argMap["kernel_h"].has_i() && "Invalid kernel argument");
    assert(argMap["kernel_w"].has_i() && "Invalid kernel argument");
    int kernelH = static_cast<int>(argMap["kernel_h"].i());
    int kernelW = static_cast<int>(argMap["kernel_w"].i());
    kernelShape = {kernelH, kernelW};
  } else {
    assert(0 && "Could not parse kernel argument");
  }
  return kernelShape;
}

std::vector<int> getStrides(std::map<std::string, caffe2::Argument> argMap) {
  std::vector<int> strides;
  // TODO: include all the other ways of adding these args.
  // e.g. strides, stride_h, etc.
  if (argMap.count("stride")) {
    assert(argMap["stride"].has_i() && "Invalid stride argument");
    int stride = static_cast<int>(argMap["stride"].i());
    strides = {stride, stride};
  }
  return strides;
}

std::vector<int> getPads(std::map<std::string, caffe2::Argument> argMap) {
  std::vector<int> pads;
  if (argMap.count("pad")) {
    assert(argMap["pad"].has_i() && "Invalid pad argument");
    int pad = static_cast<int>(argMap["pad"].i());
    pads = {pad, pad, pad, pad};
  }
  return pads;
}

std::vector<int> getDilations(std::map<std::string, caffe2::Argument> argMap) {
  std::vector<int> dilations;
  if (argMap.count("dilation")) {
    assert(argMap["dilation"].has_i() && "Invalid dilation argument");
    int dilation = static_cast<int>(argMap["dilation"].i());
    dilations = {dilation, dilation};
  }
  return dilations;
}

namespace caffe2 {

std::unique_ptr<repr::NeuralNetOperator>
convertToOperatorDef(caffe2::OperatorDef op) {
  auto argMap = getArgumentsFromOperator(op);

  std::unique_ptr<repr::NeuralNetOperator> nnOp;

  if (op.type() == "Conv") {
    auto kernelShape = getKernelShape(argMap);
    nnOp = util::make_unique<repr::Conv>(kernelShape);
    auto c = dyn_cast<repr::Conv>(nnOp.get());

    c->setStrides(getStrides(argMap));
    c->setPads(getPads(argMap));
    c->setDilations(getDilations(argMap));

  }

  if (op.type() == "Relu") {
    nnOp = util::make_unique<repr::Relu>();
  }

  if (op.type() == "AveragePool") {
    auto kernelShape = getKernelShape(argMap);
    nnOp = util::make_unique<repr::AveragePool>(kernelShape);
  }

  if (op.type() == "MaxPool") {
    auto kernelShape = getKernelShape(argMap);
    nnOp = util::make_unique<repr::MaxPool>(kernelShape);
  }

  if (op.type() == "Sum") {
    nnOp = util::make_unique<repr::Sum>();
  }

  if (op.type() == "SpatialBN") {
    nnOp = util::make_unique<repr::BatchNormalization>();
  }

  if (!nnOp) {
    nnOp = util::make_unique<repr::GenericOperator>(op.type());
  }

  // Generic attributes associated with Ops here
  nnOp->setLayout(getLayout(argMap));

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
  auto argMap = getArgumentsFromOperator(op);
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

  assert(rev_sorted.front().getNodes().size() == 1 &&
      "More than one exit node.");
  assert(rev_sorted.back().getNodes().size() == 1 &&
      "More than one entry node.");

  auto exit_tensor = *(rev_sorted.front().getNodes().begin());
  assert(isa<repr::NeuralNetData>(exit_tensor->data()) &&
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
      opNode->resetData(convertToOperatorDef(op));
      auto currentBasicBlock = bbNode->mutableData()->get();
      currentBasicBlock->pushInstructionNode(opNode);
    }
    auto opRef = dyn_cast<repr::NeuralNetOperator>(opNode->data().get());

    assert(opNode->data());

    auto annotation = util::make_unique<Caffe2Annotation>();
    annotation->setOperatorDef(&op);

    auto device_name = op.device_option().node_name();
    if (device_name != "") {
      annotation->setDevice(device_name);
    }

    opRef->setAnnotation(std::move(annotation));

  }

  repr::NNModule module;
  module.dataFlow = std::move(dfg);
  module.controlFlow = std::move(cfg);
  if (blobMapOut) {
    *blobMapOut = blobMap;
  }
  return module;
}

caffe2::OperatorDef convertToOperatorDef(repr::NNGraph::NodeRef instrNode) {
  auto *nnOp = repr::nn::get<repr::NeuralNetOperator>(instrNode);
  auto *annotation = nnOp->getAnnotation();
  caffe2::OperatorDef op;

  if (!annotation) {
    using NNKind = repr::NeuralNetOperator::NNKind;
    switch (nnOp->getKind()) {
      case NNKind::Send:
      {
        auto sendOp = dyn_cast<repr::Send>(nnOp);
        op.set_type("Placeholder:Send");
        {
          auto arg = op.add_arg();
          arg->set_name("dst_node");
          arg->set_s(sendOp->getDestination());
        }
        {
          auto out = repr::nn::getOutputs(instrNode).front();
          auto t = repr::nn::get<repr::Data>(out);
          auto arg = op.add_arg();
          arg->set_name("callsite_id");
          arg->set_i(t->getVersion());
        }
        break;
      }
      case NNKind::Receive:
      {
        auto recvOp = dyn_cast<repr::Receive>(nnOp);
        op.set_type("Placeholder:Receive");
        {
          auto arg = op.add_arg();
          arg->set_name("src_node");
          arg->set_s(recvOp->getSource());
        }
        {
          auto out = repr::nn::getOutputs(instrNode).front();
          auto t = repr::nn::get<repr::Data>(out);
          auto arg = op.add_arg();
          arg->set_name("callsite_id");
          arg->set_i(t->getVersion());
        }
        break;
      }
      default:
        op.set_type(nnOp->getName());
    }
  } else {
    switch (annotation->getKind()) {
      case repr::Annotation::AnnotationKind::Caffe2:
        op = *dyn_cast<Caffe2Annotation>(annotation)->getOperatorDef();
        break;
      default:
        op.set_type("__NOMNIGRAPH_CONVERSION_ERROR__");
        assert(0 && "Couldn't convert operator annotation to Caffe2 operator def");
        break;
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
      assert(0 && "Control flow not yet supported in Caffe2 converter.");
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
