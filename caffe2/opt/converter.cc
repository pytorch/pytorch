#include <limits>
#include <memory>
#include <utility>

#include "caffe2/core/logging.h"
#include "caffe2/opt/converter.h"

#include "nomnigraph/Graph/Algorithms.h"

#include "nomnigraph/Support/Casting.h"

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

int getGroup(std::map<std::string, caffe2::Argument>& argMap) {
  if (argMap.count("group")) {
    CAFFE_ENFORCE(argMap["group"].has_i() && "Invalid group argument");
    return static_cast<int>(argMap["group"].i());
  }
  return 1;
}

} // namespace

namespace caffe2 {

C10_DEFINE_REGISTRY(ConverterRegistry, Converter);

std::map<std::string, caffe2::Argument> Converter::getArgumentsFromOperator(
    caffe2::OperatorDef op) {
  std::map<std::string, caffe2::Argument> argMap;
  // NOLINTNEXTLINE(performance-for-range-copy)
  for (auto arg : op.arg()) {
    argMap[arg.name()] = arg;
  }
  return argMap;
}

repr::NeuralNetOperator::NNLayout getLayout(
    std::map<std::string, caffe2::Argument> argMap) {
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
  if (annotation && isa<Caffe2Annotation>(annotation)) {
    return dyn_cast<Caffe2Annotation>(annotation)->getOperatorDef();
  }
  LOG(WARNING)
      << "Cannot instantiate this OperatorDef from nomnigraph, falling back";
  caffe2::OperatorDef op;
  op.set_type(nnOp->getName());
  return op;
}

DeviceOption Converter::getDeviceOption(
    const nom::repr::NeuralNetOperator* nnOp) const {
  auto* annotation = nnOp->getAnnotation();
  // Default to using the stored operator.
  if (annotation && isa<Caffe2Annotation>(annotation)) {
    return dyn_cast<Caffe2Annotation>(annotation)
        ->getOperatorDef()
        .device_option();
  }
  caffe2::DeviceOption opt;
  return opt;
}

std::vector<int> getKernelShape(
    std::map<std::string, caffe2::Argument> argMap) {
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
    nnOp = std::make_unique<repr::Conv>(kernelShape);
    auto c = dyn_cast<repr::Conv>(nnOp.get());

    c->setStrides(getStrides(argMap));
    c->setPads(getPads(argMap));
    c->setDilations(getDilations(argMap));
    c->setGroup(getGroup(argMap));

    return nnOp;
  }
  // Does not override default converter to OperatorDef

  // NOLINTNEXTLINE(modernize-use-equals-default)
  ~ConvConverter() override {}
};

class ConvTransposeConverter : public Converter {
  std::unique_ptr<nom::repr::NeuralNetOperator> convertToNeuralNetOperator(
      const OperatorDef& op) override {
    std::unique_ptr<repr::NeuralNetOperator> nnOp;
    auto argMap = getArgumentsFromOperator(op);
    auto kernelShape = getKernelShape(argMap);
    nnOp = std::make_unique<repr::ConvTranspose>(kernelShape);
    auto c = dyn_cast<repr::ConvTranspose>(nnOp.get());

    c->setStrides(getStrides(argMap));
    c->setPads(getPads(argMap));
    c->setGroup(getGroup(argMap));

    return nnOp;
  }
  // Does not override default converter to OperatorDef

  // NOLINTNEXTLINE(modernize-use-override,modernize-use-equals-default)
  virtual ~ConvTransposeConverter() {}
};

REGISTER_CONVERTER(Conv, ConvConverter);

REGISTER_CONVERTER(ConvTranspose, ConvTransposeConverter);

TRIVIAL_CONVERTER(Relu);
REGISTER_CONVERTER(Relu, ReluConverter);

TRIVIAL_CONVERTER(Sum);
REGISTER_CONVERTER(Sum, SumConverter);

TRIVIAL_CONVERTER(BatchNormalization);
REGISTER_CONVERTER(SpatialBN, BatchNormalizationConverter);

TRIVIAL_CONVERTER(Flatten);
REGISTER_CONVERTER(Flatten, FlattenConverter);

class ClipConverter : public Converter {
  std::unique_ptr<nom::repr::NeuralNetOperator> convertToNeuralNetOperator(
      const OperatorDef& op) override {
    auto argMap = getArgumentsFromOperator(op);
    float min = std::numeric_limits<float>::lowest();
    float max = std::numeric_limits<float>::max();

    if (argMap.count("min")) {
      CAFFE_ENFORCE(argMap["min"].has_f(), "Invalid 'min' argument");
      min = static_cast<float>(argMap["min"].f());
    }

    if (argMap.count("max")) {
      CAFFE_ENFORCE(argMap["max"].has_f(), "Invalid 'max' argument");
      max = static_cast<float>(argMap["max"].f());
    }

    return std::make_unique<repr::Clip>(min, max);
  }
  // Does not override default converter to OperatorDef

  // NOLINTNEXTLINE(modernize-use-equals-default)
  ~ClipConverter() override {}
};
REGISTER_CONVERTER(Clip, ClipConverter);

class AveragePoolConverter : public Converter {
  std::unique_ptr<nom::repr::NeuralNetOperator> convertToNeuralNetOperator(
      const OperatorDef& op) override {
    std::unique_ptr<repr::NeuralNetOperator> nnOp;
    auto argMap = getArgumentsFromOperator(op);
    auto kernelShape = getKernelShape(argMap);
    nnOp = std::make_unique<repr::AveragePool>(kernelShape);
    return nnOp;
  }
  // Does not override default converter to OperatorDef

  // NOLINTNEXTLINE(modernize-use-equals-default)
  ~AveragePoolConverter() override {}
};
REGISTER_CONVERTER(AveragePool, AveragePoolConverter);

class MaxPoolConverter : public Converter {
  std::unique_ptr<nom::repr::NeuralNetOperator> convertToNeuralNetOperator(
      const OperatorDef& op) override {
    std::unique_ptr<repr::NeuralNetOperator> nnOp;
    auto argMap = getArgumentsFromOperator(op);
    auto kernelShape = getKernelShape(argMap);
    nnOp = std::make_unique<repr::MaxPool>(kernelShape);
    return nnOp;
  }
  // Does not override default converter to OperatorDef

  // NOLINTNEXTLINE(modernize-use-equals-default)
  ~MaxPoolConverter() override {}
};
REGISTER_CONVERTER(MaxPool, MaxPoolConverter);

class ConcatConverter : public Converter {
  std::unique_ptr<nom::repr::NeuralNetOperator> convertToNeuralNetOperator(
      const OperatorDef& op) override {
    std::unique_ptr<repr::NeuralNetOperator> nnOp =
        std::make_unique<repr::Concat>();
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

  // NOLINTNEXTLINE(modernize-use-equals-default)
  ~ConcatConverter() override {}
};
REGISTER_CONVERTER(Concat, ConcatConverter);

class FCConverter : public Converter {
  std::unique_ptr<nom::repr::NeuralNetOperator> convertToNeuralNetOperator(
      const OperatorDef& op) override {
    std::unique_ptr<repr::NeuralNetOperator> nnOp =
        std::make_unique<repr::FC>();
    auto argMap = getArgumentsFromOperator(op);

    auto c = dyn_cast<repr::FC>(nnOp.get());
    if (argMap.count("axis")) {
      CAFFE_ENFORCE(argMap["axis"].has_i(), "Invalid axis argument");
      int axis = static_cast<int>(argMap["axis"].i());
      c->setAxis(axis);
    }
    if (argMap.count("axis_w")) {
      CAFFE_ENFORCE(argMap["axis_w"].has_i(), "Invalid axis_w argument");
      int axis_w = static_cast<int>(argMap["axis_w"].i());
      c->setAxisW(axis_w);
    }

    return nnOp;
  }
  // Does not override default converter to OperatorDef

  // NOLINTNEXTLINE(modernize-use-equals-default)
  ~FCConverter() override {}
};
REGISTER_CONVERTER(FC, FCConverter);

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
    nnOp = std::make_unique<repr::GenericOperator>(op.type());
  }

  // Generic attributes associated with Ops here
  nnOp->setLayout(getLayout(argMap));

  auto annotation = std::make_unique<Caffe2Annotation>();
  annotation->setOperatorDef(op);

  auto device_name = op.device_option().node_name();
  if (device_name != "") {
    annotation->setDevice(device_name);
  }
  annotation->setDeviceType(op.device_option().device_type());

  nnOp->setAnnotation(std::move(annotation));

  return nnOp;
}

/// \brief Ingest a caffe2 protobuf model and output an NNModule.
/// \param net The caffe2 protobuf NetDef
repr::NNModule convertToNNModule(
    const caffe2::NetDef& net,
    bool strict,
    std::vector<repr::NNGraph::NodeRef>* opNodeVec) {
  repr::NNModule module;
  repr::NNGraph& dfg = module.dataFlow;
  repr::NNCFGraph& cfg = module.controlFlow;
  /// \brief We keep track of the producer of the blob.
  /// Because Caffe2 Nets are really just ordered operations
  /// we can just keep track of the most recent producer of
  /// a blob and draw and edge from that to any consumer we
  /// come by. If a new operator produces the blob we simply
  /// replace it in this map.
  std::unordered_map<std::string, repr::NNGraph::NodeRef> blobMap;

  std::unordered_set<std::string> externalInputNames;
  for (const auto& inputName : net.external_input()) {
    externalInputNames.insert(inputName);
  }

  /// \brief For the construction of the control flow graph we keep track
  /// of a current basic block, which we split up as we come across control
  /// flow operations such as if and while.
  auto bbNode = cfg.createNamedFunction("main");

  for (const auto& op : net.op()) {
    auto opNode = dfg.createNode(); // Create an empty node for the operator.
    // First calculate in-edges (data dependencies).
    for (const auto& input : op.input()) {
      // If we've never seen this tensor, make one.
      if (!blobMap.count(input)) {
        auto tensor = std::make_unique<repr::Tensor>(input);
        blobMap[input] =
            dfg.createNode(unique_dyn_cast<repr::NeuralNetData>(tensor));
        if (externalInputNames.count(input)) {
          module.inputs.insert(blobMap[input]);
          externalInputNames.erase(input);
        }
      }

      auto tensorNode = blobMap[input];
      dfg.createEdge(tensorNode, opNode);
    }

    // Then save outputs into the blobMap for later consumption.
    for (const auto& output : op.output()) {
      auto tensor = std::make_unique<repr::Tensor>(output);
      auto tensorNode =
          dfg.createNode(unique_dyn_cast<repr::NeuralNetData>(tensor));
      dfg.createEdge(opNode, tensorNode);
      blobMap[output] = tensorNode;
    }

    opNode->resetData(convertToNeuralNetOperator(op));
    if (opNodeVec) {
      opNodeVec->emplace_back(opNode);
    }
    auto currentBasicBlock = bbNode->mutableData();
    currentBasicBlock->pushInstructionNode(opNode);
  }

  if (externalInputNames.size()) {
    // In strict mode we ensure the input names are valid
    if (strict) {
      std::ostringstream os;
      for (const auto& inputName : externalInputNames) {
        os << "\"" << inputName << "\" ";
      }

      CAFFE_ENFORCE(
          externalInputNames.size() == 0,
          "Attempting to convert an ill-formed network: ",
          "external_input contains ",
          externalInputNames.size(),
          " unused blobs: ",
          os.str());
      // Otherwise, we add the blobs to the graph as no-ops
    } else {
      for (const auto& input : externalInputNames) {
        blobMap[input] = dfg.createNode(std::make_unique<repr::Tensor>(input));
      }
    }
  }

  for (const auto& outputName : net.external_output()) {
    CAFFE_ENFORCE(
        !strict || blobMap.count(outputName),
        "NetDef has ill-formed external_output:",
        outputName);
    if (!blobMap.count(outputName)) {
      LOG(ERROR) << "NetDef has ill-formed external_output: " << outputName;
      continue;
    }
    module.outputs.insert(blobMap[outputName]);
  }

  return module;
}

caffe2::OperatorDef convertToOperatorDef(
    const repr::NNGraph::NodeRef& instrNode) {
  auto* nnOp = repr::nn::get<repr::NeuralNetOperator>(instrNode);
  auto op_type = nnOp->getName();
  auto* annotation = nnOp->getAnnotation();
  caffe2::OperatorDef op;

  if (ConverterRegistry()->Has(op_type)) {
    op = ConverterRegistry()->Create(op_type)->convertToOperatorDef(nnOp);
  } else if (!annotation) {
    op.set_type(op_type);
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

Caffe2Annotation* getOrAddCaffe2Annotation(
    nom::repr::NNGraph::NodeRef& instrNode) {
  auto* nnOp = repr::nn::get<repr::NeuralNetOperator>(instrNode);
  auto* annotation = nnOp->getMutableAnnotation();
  if (!annotation) {
    auto new_annot = std::make_unique<Caffe2Annotation>();
    new_annot->setOperatorDef(convertToOperatorDef(instrNode));
    nnOp->setAnnotation(std::move(new_annot));
    annotation = nnOp->getMutableAnnotation();
  }
  CAFFE_ENFORCE(isa<Caffe2Annotation>(annotation));
  auto c2_annotation = dyn_cast<Caffe2Annotation>(annotation);
  return c2_annotation;
}

caffe2::NetDef convertToCaffe2Proto(repr::NNModule& m) {
  auto predictNet = caffe2::NetDef();
  return convertToCaffe2Proto(m, predictNet);
}

std::vector<std::string> mergeExternalTensors(
    const std::unordered_set<repr::NNGraph::NodeRef>& currExternal,
    const std::vector<std::string>& oldExternal) {
  std::vector<std::string> out;

  // Maximally preserve the order of external inputs and outputs.
  std::unordered_set<std::string> newExternal;
  for (const auto& tensorNode : currExternal) {
    CAFFE_ENFORCE(
        repr::nn::is<repr::NeuralNetData>(tensorNode),
        "A non-tensor node was added to external inputs/outputs of the NNModule");
    auto name = repr::nn::get<repr::NeuralNetData>(tensorNode)->getName();
    newExternal.insert(name);
  }

  for (const auto& tensorName : oldExternal) {
    if (newExternal.count(tensorName)) {
      out.emplace_back(tensorName);
      newExternal.erase(tensorName);
    }
  }
  for (const auto& tensorName : newExternal) {
    out.emplace_back(tensorName);
  }

  return out;
}

caffe2::NetDef convertToCaffe2Proto(
    repr::NNModule& m,
    const caffe2::NetDef& oldNet) {
  auto predictNet = caffe2::NetDef();
  // We copy the old net rather than mutate it.
  predictNet.CopyFrom(oldNet);
  predictNet.mutable_op()->Clear();

  repr::nn::coalesceInsertedDataDependencies(&m);

  // Simply iterate through the CFG and populate data dependencies
  // with the DFG
  for (const auto& bbNode : m.controlFlow.getMutableNodes()) {
    if (bbNode->getOutEdges().size() > 1) {
      CAFFE_THROW("Control flow not yet supported in Caffe2 converter.");
    }
    auto& bb = bbNode->data();
    for (const auto& instrNode : bb.getInstructions()) {
      caffe2::OperatorDef op = convertToOperatorDef(instrNode);

      for (const auto& inEdge : instrNode->getInEdges()) {
        auto* tensorNode =
            dyn_cast<repr::NeuralNetData>(inEdge->tail()->data().get());
        *op.add_input() = tensorNode->getName();
      }
      for (const auto& outEdge : instrNode->getOutEdges()) {
        auto* tensorNode =
            dyn_cast<repr::NeuralNetData>(outEdge->head()->data().get());
        *op.add_output() = tensorNode->getName();
      }

      auto* nnOp = repr::nn::get<repr::NeuralNetOperator>(instrNode);
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

  // Maximally preserve the order of external inputs and outputs.
  std::vector<std::string> oldExternalInputs;
  std::vector<std::string> oldExternalOutputs;

  for (const auto& inputName : predictNet.external_input()) {
    oldExternalInputs.emplace_back(inputName);
  }
  for (const auto& outputName : predictNet.external_output()) {
    oldExternalOutputs.emplace_back(outputName);
  }

  auto newExternalInputs = mergeExternalTensors(m.inputs, oldExternalInputs);
  auto newExternalOutputs = mergeExternalTensors(m.outputs, oldExternalOutputs);

  predictNet.clear_external_input();
  predictNet.clear_external_output();

  for (const auto& inputName : newExternalInputs) {
    predictNet.add_external_input(inputName);
  }

  for (const auto& outputName : newExternalOutputs) {
    predictNet.add_external_output(outputName);
  }

  return predictNet;
}

void pushOpToFront(caffe2::OperatorDef& op, caffe2::NetDef* net) {
  *net->add_op() = op;
  google::protobuf::RepeatedPtrField<caffe2::OperatorDef>* op_list(
      net->mutable_op());
  // Reverse iterate, swapping new element in front each time
  for (int i(net->op_size() - 1); i > 0; --i) {
    op_list->SwapElements(i, i - 1);
  }
}

void injectDataEdgeIndicators(caffe2::NetDef* net) {
  for (const auto& input : net->external_input()) {
    caffe2::OperatorDef op;
    op.set_type("Declare");
    op.add_output(input);
    pushOpToFront(op, net);
  }
  for (const auto& output : net->external_output()) {
    caffe2::OperatorDef op;
    op.set_type("Export");
    op.add_input(output);
    *net->add_op() = std::move(op);
  }
  net->clear_external_input();
  net->clear_external_output();
}

void removeDataEdgeIndicators(caffe2::NetDef* net) {
  google::protobuf::RepeatedPtrField<caffe2::OperatorDef>* op_list(
      net->mutable_op());
  for (auto i = 0; i < net->op_size(); ++i) {
    auto op = net->op(i);
    if (op.type() == "Declare") {
      net->add_external_input(op.output(0));
    } else if (op.type() == "Export") {
      net->add_external_output(op.input(0));
    } else {
      continue;
    }
    // Note that this compensates for modifying the list inplace
    op_list->DeleteSubrange(i--, 1);
  }
}

} // namespace caffe2
