#include "caffe2/opt/custom/concat_elim.h"
#include "caffe2/core/logging.h"
#include "caffe2/opt/nql/graphmatcher.h"
#include "caffe2/opt/passes.h"
#include "nomnigraph/Representations/NeuralNet.h"
#include "nomnigraph/Support/Common.h"
#include "nomnigraph/Transformations/SubgraphMatcher.h"

using namespace nom::repr;

namespace caffe2 {
namespace opt {

void concatElim(NNModule* nn) {
  using namespace nom::matcher;
  using namespace nom::repr::nn;
  using namespace nom::repr;

  auto mg = NNMatchGraph();

  auto matchConcatInputs =
      mg.createNode(std::move(matchExternalTensorNode().starCount()));
  auto matchConcat = mg.createNode([](NNGraph::NodeRef nodeRef) {
    NOM_REQUIRE_OR_RET_FALSE(nn::is<Concat>(nodeRef));
    NOM_REQUIRE_OR_RET_FALSE(nn::hasUniqueConsumer(nodeRef));
    auto node = nn::get<Concat>(nodeRef);
    return node->getAxis() == 1 && node->getAddAxis();
  });
  mg.createEdge(matchConcatInputs, matchConcat);

  auto matchConcatOutput = mg.createNode(nn::is<nom::repr::Tensor>);
  mg.createEdge(matchConcat, matchConcatOutput);

  auto matchBatchMatmul = mg.createNode([](NNGraph::NodeRef nodeRef) {
    NOM_REQUIRE_OR_RET_FALSE(nn::is<BatchMatMul>(nodeRef));
    NOM_REQUIRE_OR_RET_FALSE(nn::hasSingleOutputAndConsumer(nodeRef));
    auto node = nn::get<BatchMatMul>(nodeRef);
    return node->getTransA() == 0 && node->getTransB() == 1 &&
        node->getBroadcast() == 0;
  });

  mg.createEdge(matchConcatOutput, matchBatchMatmul);
  mg.createEdge(matchConcatOutput, matchBatchMatmul);

  auto matchBmmOutput = mg.createNode(nn::is<nom::repr::Tensor>);
  mg.createEdge(matchBatchMatmul, matchBmmOutput);

  auto matchFlatten = mg.createNode([](NNGraph::NodeRef nodeRef) {
    NOM_REQUIRE_OR_RET_FALSE(nn::is<Flatten>(nodeRef));
    return nn::hasSingleOutputAndConsumer(nodeRef);
  });
  mg.createEdge(matchBmmOutput, matchFlatten);

  auto matchFlattenOutput = mg.createNode(nn::is<nom::repr::Tensor>);
  mg.createEdge(matchFlatten, matchFlattenOutput);

  auto matchIndices = mg.createNode(matchExternalTensorNode());
  auto matchBatchGather = mg.createNode(nn::is<BatchGather>);
  mg.createEdge(matchFlattenOutput, matchBatchGather);
  mg.createEdge(matchIndices, matchBatchGather);

  mg.replaceSubgraph(
      nn->dataFlow,
      matchBatchGather,
      [matchConcatOutput](
          NNGraph& g,
          NNGraph::NodeRef batchGatherNode,
          const NNMatchGraph::SubgraphMatchResultType& matchResult) {
        auto fusedNode =
            g.createNode(make_unique<ConcatBatchMatMulBatchGatherOp>());

        auto batchGatherNodeOutputs = nn::getOutputs(batchGatherNode);
        for (const auto& output : batchGatherNodeOutputs) {
          auto tensor = nn::get<nom::repr::Tensor>(output);
          // Handle cases where blob names are reused - D9113128.
          auto newOutput = g.createNode(
              make_unique<nom::repr::Tensor>(tensor->getName() + "_cc_bmm_bg"));
          g.createEdge(fusedNode, newOutput);
          g.replaceOutEdges(output, newOutput);
        }

        auto concatNode =
            getProducer(matchResult.getMatchNodeMap()->at(matchConcatOutput));
        g.replaceInEdges(batchGatherNode, fusedNode);
        g.replaceInEdges(concatNode, fusedNode);
        g.deleteNodes(matchResult.getMatchedSubgraph()->getNodes());
        return true;
      });
}

REGISTER_OPT_PASS_FROM_FUNC(ConcatElim, concatElim);

void concatAddMulNaNClipElim(NNModule* nn) {
  using namespace nom::repr;

  nom::nql::GraphMatcher gm;
  gm.initFromString(R"NQL(def query {
    %concat_out, %split_info = Concat(*)
    %add = Add(%concat_out, %add_in)
    %mul = Mul(%add, %mul_in)
    %replace = ReplaceNaN(%mul)
    %out = Clip(%replace)
  })NQL");
  CAFFE_ENFORCE(gm.getMatcher(), "Unable to parse NQL query.");

  // Iterate through each match and replace them
  for (const auto& match : gm.getMatches(nn->dataFlow)) {
    // Various attributes we care about for this fusion
    NOM_REQUIRE_OR_CONT(nn::get<Concat>(match["Concat"])->getAxis() == 1);
    NOM_REQUIRE_OR_CONT(nn::get<Add>(match["Add"])->getBroadcast() == 1);
    NOM_REQUIRE_OR_CONT(nn::get<Mul>(match["Mul"])->getBroadcast() == 1);
    NOM_REQUIRE_OR_CONT(
        std::abs(nn::get<ReplaceNaN>(match["ReplaceNaN"])->getValue()) < 0.01);

    // Figure out the input/output order (creating new nodes if needed)
    std::vector<NNGraph::NodeRef> inputs, outputs;

    // First set up the inputs
    inputs.emplace_back(match["\%add_in"]);
    inputs.emplace_back(match["\%mul_in"]);
    for (const auto& concat_input : nn::getInputs(match["Concat"])) {
      inputs.emplace_back(concat_input);
    }

    // Set up the outputs
    outputs.emplace_back(match["\%out"]);
    // TODO(duc): The subgraph matcher doesn't yet handle patterns
    // that are not trees, meaning the %split_info node is not yet
    // matched.
    outputs.emplace_back(nn::getOutputs(match["Concat"]).at(1));

    auto min = nn::get<Clip>(match["Clip"])->getMin();
    auto max = nn::get<Clip>(match["Clip"])->getMax();
    // This will do all the work
    nn->replaceSubgraphWithOperator<ConcatAddMulReplaceNaNClip>(
        match.subgraph, inputs, outputs, min, max);
  }
}

REGISTER_OPT_PASS_FROM_FUNC(ConcatAddMulNaNClipElim, concatAddMulNaNClipElim);

void gatherFuse8BitRowwiseQuantFloatMulLengthsSumElim(NNModule* nn) {
  using namespace nom::repr;

  nom::nql::GraphMatcher gm;
  gm.initFromString(R"NQL(def query {
    %gather = Gather(%a, %b)
    %ff = Fused8BitRowwiseQuantizedToFloat(%gather)
    %mu = Mul(%ff, %mul_in)
    %out = LengthsSum(%mu, %len_in)
  })NQL");
  CAFFE_ENFORCE(gm.getMatcher(), "Unable to parse NQL query.");

  // Iterate through each match and replace them
  for (const auto& match : gm.getMatches(nn->dataFlow)) {
    NOM_REQUIRE_OR_CONT(nn::get<Mul>(match["Mul"])->getBroadcast() == 1);
    NOM_REQUIRE_OR_CONT(nn::get<Mul>(match["Mul"])->getAxis() == 0);
    // Figure out the input/output order (creating new nodes if needed)
    std::vector<NNGraph::NodeRef> inputs, outputs;

    // First set up the inputs
    const auto& gather_inputs = nn::getInputs(match["Gather"]);
    inputs.emplace_back(gather_inputs.at(0));
    inputs.emplace_back(match["\%mul_in"]);
    inputs.emplace_back(gather_inputs.at(1));
    inputs.emplace_back(match["\%len_in"]);

    // Set up the outputs
    outputs.emplace_back(match["\%out"]);

    // Check if outputs of the subgraph contain intermediate tensors
    // If so, abort fusion.
    std::unordered_set<NNGraph::NodeRef> internal;
    for (const auto& output : nn::getOutputs(match["Gather"])) {
      internal.emplace(output);
    }
    for (const auto& output :
         nn::getOutputs(match["Fused8BitRowwiseQuantizedToFloat"])) {
      internal.emplace(output);
    }
    for (const auto& output : nn::getOutputs(match["Mul"])) {
      internal.emplace(output);
    }
    for (const auto& output : nn::getOutputs(match.subgraph)) {
      if (internal.count(output)) {
        LOG(INFO) << "Skip fusing Gather-Fused8BitRowwiseQuantizedToFloat"
                  << "-Mul-LengthsSum as internal tensor "
                  << nn::getName(output)
                  << " is used as external output of the subgraph.";
        return;
      }
    }

    // This will do all the work
    nn->replaceSubgraphWithOperator<SparseLengthsWeightedSumFused8BitRowwise>(
        match.subgraph, inputs, outputs);
  }
}

REGISTER_OPT_PASS_FROM_FUNC(
    GatherFuse8BitRowwiseQuantFloatMulLengthsSumElim,
    gatherFuse8BitRowwiseQuantFloatMulLengthsSumElim);

} // namespace opt
} // namespace caffe2
