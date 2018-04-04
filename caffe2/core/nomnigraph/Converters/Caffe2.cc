#include "nomnigraph/Converters/Caffe2.h"
#include "nomnigraph/Graph/Algorithms.h"

#include "nomnigraph/Support/Casting.h"
#include "nomnigraph/Support/Pointer.h"

std::map<std::string, caffe2::Argument>
getArgumentsFromOperator(caffe2::OperatorDef op) {
  std::map<std::string, caffe2::Argument> argMap;
  for (auto arg : op.arg()) {
    argMap[arg.name()] = arg;
  }
  return argMap;
}

namespace nom {
namespace converters {

std::unique_ptr<repr::NeuralNetOperator>
convertOperator(caffe2::OperatorDef op) {
  auto argMap = getArgumentsFromOperator(op);

  if (op.type() == "Conv") {
    // There are literally three ways to define shapes in Conv in Caffe2
    std::vector<int> kernelShape;
    if (argMap.count("kernel")) {
      assert(argMap["kernel"].has_i() && "Invalid kernel argument passed to Conv");
      int kernel = static_cast<int>(argMap["kernel"].i());
      kernelShape = {kernel, kernel};
    } else if (argMap.count("kernels")) {
      for (auto i : argMap["kernels"].ints()) {
        kernelShape.push_back(static_cast<int>(i));
      }
    } else if (argMap.count("kernel_h") && argMap.count("kernel_w")) {
      assert(argMap["kernel_h"].has_i() && "Invalid kernel argument passed to Conv");
      assert(argMap["kernel_w"].has_i() && "Invalid kernel argument passed to Conv");
      int kernelH = static_cast<int>(argMap["kernel_h"].i());
      int kernelW = static_cast<int>(argMap["kernel_w"].i());
      kernelShape = {kernelH, kernelW};
    } else {
      assert(0);
    }

    auto c = util::make_unique<repr::Conv>(kernelShape);

    if (argMap.count("order")) {
      auto order = argMap["order"].s();
      if (order == "NCHW") {
        c->setLayout(repr::Conv::NNLayout::NCHW);
      } else if (order == "NHWC") {
        c->setLayout(repr::Conv::NNLayout::NHWC);
      }
    }

    // TODO: include all the other ways of adding these args.
    // e.g. strides, stride_h, etc.
    if (argMap.count("stride")) {
      assert(argMap["stride"].has_i() && "Invalid stride argument");
      int stride = static_cast<int>(argMap["stride"].i());
      c->setStrides({stride, stride});
    }

    if (argMap.count("pad")) {
      assert(argMap["pad"].has_i() && "Invalid pad argument");
      int pad = static_cast<int>(argMap["pad"].i());
      c->setPads({pad, pad, pad, pad});
    }

    if (argMap.count("dilation")) {
      assert(argMap["dilation"].has_i() && "Invalid dilation argument");
      int dilation = static_cast<int>(argMap["dilation"].i());
      c->setDilations({dilation, dilation});
    }

    return std::move(c);
  }

  if (op.type() == "Relu") {
    auto relu = util::make_unique<repr::Relu>();
    return std::move(relu);
  }

  return util::make_unique<repr::GenericOperator>(op.type());
}


/// \brief Ingest a caffe2 protobuf model and output an NNModule.
/// \param net The caffe2 protobuf NetDef
/// \param blobMap [optional][output] A pointer to a blobMap to be populated with all the output blobs of the NetDef by name->NodeRef
repr::NNModule convertFromCaffe2Proto(const caffe2::NetDef &net, std::unordered_map<std::string, repr::NNGraph::NodeRef>* blobMapOut) {
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

  for (const auto &op : net.op()) {
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
      opNode->resetData(util::make_unique<repr::While>());
      auto argMap = getArgumentsFromOperator(op);
      std::string bodyNetSerialized = argMap["body"].s();
      auto bodyNet = caffe2::NetDef();
      bodyNet.ParseFromString(bodyNetSerialized);

      std::unordered_map<std::string, repr::NNGraph::NodeRef> bodyBlobMap;
      auto bodyNN = convertFromCaffe2Proto(bodyNet, &bodyBlobMap);
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
        bodyGraph.swapNode(node, dfg);
      }

      for (auto edge : bodyEdges) {
        bodyGraph.swapEdge(edge, dfg);
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
        bodyCFGraph.swapNode(cfNode, cfg);
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
        bodyCFGraph.swapEdge(cfEdge, cfg);
      }

      // Now create the false case.
      bbNode =
          cfg.createNode(util::make_unique<repr::BasicBlockType<repr::NNGraph>>());
      cfg.createEdge(whileCFNode, bbNode, -1);
    } else {
      opNode->resetData(convertOperator(op));
      auto currentBasicBlock = bbNode->mutableData()->get();
      currentBasicBlock->pushInstructionNode(opNode);
    }
    auto opRef = dyn_cast<repr::NeuralNetOperator>(opNode->data().get());

    assert(opNode->data());

    auto device_name = op.device_option().node_name();
    if (device_name != "") {
      auto device = util::make_unique<repr::DeviceAnnotation>(device_name);
      opRef->setAnnotation(std::move(device));
    } else {
      opRef->setAnnotation(util::make_unique<repr::Annotation>());
    }

    opRef->getMutableAnnotation()->setSaved((void *)&op);
  }

  repr::NNModule module;
  module.dataFlow = std::move(dfg);
  module.controlFlow = std::move(cfg);
  if (blobMapOut) {
    *blobMapOut = blobMap;
  }
  return module;
}

caffe2::NetDef convertToCaffe2Proto(repr::NNModule &m) {
  auto predictNet = caffe2::NetDef();

  repr::nn::coalesceInsertedDataDependencies(&m);

  // Simply iterate through the CFG and populate data dependencies
  // with the DFG
  for (const auto &bbNode : m.controlFlow.getMutableNodes()) {
    if (bbNode->getOutEdges().size() > 1) {
      assert(0 && "Control flow not yet supported in Caffe2 converter.");
    }
    auto bb = bbNode->data().get();
    for (const auto &instrNode : bb->getInstructions()) {
      auto *nnOp = dyn_cast<repr::NeuralNetOperator>(instrNode->data().get());
      auto *annotation = nnOp->getAnnotation();
      assert(annotation->getSaved() &&
             "Generating Caffe2 operators from IR not yet supported.\n");
      auto *op =
          reinterpret_cast<caffe2::OperatorDef *>(annotation->getSaved());

      // We may have swapped out some of the edges.
      op->clear_input();
      op->clear_output();
      for (const auto &inEdge : instrNode->getInEdges()) {
        auto *tensorNode =
            dyn_cast<repr::NeuralNetData>(inEdge->tail()->data().get());
        *op->add_input() = tensorNode->getName();
      }
      for (const auto &outEdge : instrNode->getOutEdges()) {
        auto *tensorNode =
            dyn_cast<repr::NeuralNetData>(outEdge->head()->data().get());
        *op->add_output() = tensorNode->getName();
      }
      // Save the operator to the net.
      *predictNet.add_op() = *op;
    }
  }

  return predictNet;
}

} // namespace converters
} // namespace nom
