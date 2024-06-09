#include "caffe2/core/operator.h"
#include "mpscnn.h"
#include "mpscnn_context.h"

#import <Metal/Metal.h>
#import <MetalPerformanceShaders/MetalPerformanceShaders.h>
#import <UIKit/UIDevice.h>

namespace caffe2 {
struct Analysis {
  struct SSA {
    using BlobVersions = std::unordered_map<std::string, size_t>;
    BlobVersions inVersions;
    BlobVersions outVersions;
  };
  std::vector<SSA> ssa;
  std::unordered_map<
      std::string,
      std::unordered_map<size_t, std::vector<size_t>>>
      inUsages;
};

Analysis analyzeNet(const NetDef& net) {
  Analysis::SSA::BlobVersions frontier;
  Analysis analysis;

  auto play = [&](size_t i, const OperatorDef& op) {
    Analysis::SSA::BlobVersions inVersions;
    for (const auto& s : op.input()) {
      inVersions[s] = frontier[s];
      analysis.inUsages[s][frontier[s]].push_back(i);
    }
    Analysis::SSA::BlobVersions outVersions;
    for (const auto& s : op.output()) {
      if (frontier.find(s) != frontier.end()) {
        frontier[s] += 1;
      }
      outVersions[s] = frontier[s];
    }
    analysis.ssa.push_back(Analysis::SSA{inVersions, outVersions});
  };

  for (auto i = 0; i < net.op_size(); ++i) {
    play(i, net.op(i));
  }
  return analysis;
}

static void rewriteInput(OperatorDef* op, int i) {
  auto input = op->input(i);
  op->set_input(i, input + "_M");
}

static void rewriteOutput(OperatorDef* op, int i) {
  auto output = op->output(i);
  op->set_output(i, output + "_M");
}

static void insertOutputCopyFromMPSCNNOp(
    NetDef& predictNet,
    const std::vector<std::string>& cpu_blobs) {
  auto* op = predictNet.add_op();
  op->set_type("CopyFromMPSCNN");
  for (int i = 0; i < cpu_blobs.size(); ++i) {
    op->add_input(cpu_blobs[i] + "_M");
    op->add_output(cpu_blobs[i]);
  }
}

NetDef insertInputOutputCopyOps(const NetDef& def) {
  // Do some validation of the outputs. For this version, we require:
  // - a single input (first element of external_input()) is consumed by the
  // NetDef - a single output (first element of external_output()) is produced
  // by the NetDef. - the input is consumed by def.op(0), and this is the only
  // consumer. - the output is produced by def.op(-1).
  CAFFE_ENFORCE_GE(def.external_input_size(), 1);
  CAFFE_ENFORCE_GE(def.external_output_size(), 1);
  auto analysis = analyzeNet(def);
  // enforce a single use of the input blob.
  CAFFE_ENFORCE_GE(def.op_size(), 1);
  const auto& inputBlob = def.external_input(0);
  // Enforce that the input blob has a single usage - in the first operator.
  CAFFE_ENFORCE(analysis.inUsages[inputBlob][0] == (std::vector<size_t>{0}));
  // Enforce that the external_output(0) blob is produced by the last operator
  // in this sequence.
  const auto& outputBlob = def.external_output(0);
  CAFFE_ENFORCE(
      analysis.ssa.back().outVersions.find(outputBlob) !=
      analysis.ssa.back().outVersions.end());
  const auto& outputBlobVersion = analysis.ssa.back().outVersions[outputBlob];
  // This should hold true by definition of the SSA analysis.
  CAFFE_ENFORCE(
      analysis.inUsages[outputBlob].find(outputBlobVersion) ==
      analysis.inUsages[outputBlob].end());
  NetDef mdef;
  mdef.CopyFrom(def);
  mdef.clear_op();

  {
    auto& op = *(mdef.add_op());
    op.set_type("CopyToMPSCNN");
    op.add_input(def.external_input(0));
    op.add_output("__METAL_INPUT_COPY__");
  }

  std::unordered_set<std::string> output_set;

  for (auto i = 0; i < def.op_size(); ++i) {
    const auto& ogOp = def.op(i);
    auto op = mdef.add_op();
    op->CopyFrom(ogOp);
    if (i == 0) {
      CAFFE_ENFORCE_EQ(op->input(0), def.external_input(0));
      op->set_input(0, "__METAL_INPUT_COPY__");
    }
    /*
     * Let's say we have a Blob called "X" that is both the external output
     * and will be used in the later operators. And it's on Metal. First, we'll
     * rename the output of the operator to "X_M", therefore all the following
     * operators that referenced this blob will need to change the input name
     * and then we will copy "X_M" to CPU as "X" in the end.
     *
     */
    for (auto j = 0; j < op->input_size(); ++j) {
      if (output_set.find(op->input(j)) != output_set.end()) {
        rewriteInput(op, j);
        // we'll add one CopyFromMPSCNN operator in the end
        // to copy all the output blobs from MPSCNN to CPU
      }
    }
    // if the output is in external output, copy from metal when necessary
    for (auto j = 0; j < op->output_size(); ++j) {
      for (auto k = 0; k < def.external_output_size(); ++k) {
        // Assuming external output blob has unique name, e.g. only version 0
        // of the blob is used as the output
        if (op->output(j) == def.external_output(k)) {
          output_set.insert(op->output(j));
          // rewrite output to output_M for the operator
          rewriteOutput(op, j);
        }
      }
    }
  }

  // We copy all the output from Metal to CPU at once in the end
  std::vector<std::string> external_outputs;
  for (int i = 0; i < def.external_output_size(); ++i) {
    external_outputs.push_back(def.external_output(i));
  }
  insertOutputCopyFromMPSCNNOp(mdef, external_outputs);

  return mdef;
}

bool nextIsOnlyUserOfCurrent(
    const Analysis& analysis,
    size_t currentIdx,
    const OperatorDef& currentOp,
    const OperatorDef& nextOp) {
  CAFFE_ENFORCE_EQ(currentOp.output_size(), 1);
  CAFFE_ENFORCE_GE(nextOp.input_size(), 1);
  CAFFE_ENFORCE_EQ(currentOp.output(0), nextOp.input(0));
  const auto outputName = currentOp.output(0);
  // Find the version of the output name we are currently looking at.
  // This is guaranteed to exist by SSA analysis.
  const auto currentOutputVersion =
      analysis.ssa.at(currentIdx).outVersions.at(outputName);
  VLOG(2) << "Blob: " << outputName << ", idx: " << currentOutputVersion;
  // Find the usages of this in the SSA analysis.

  // Has this blob every been used?
  if (analysis.inUsages.find(outputName) == analysis.inUsages.end()) {
    return false;
  }

  // Has this version of the blob ever been used?
  if (analysis.inUsages.at(outputName).find(currentOutputVersion) ==
      analysis.inUsages.at(outputName).end()) {
    return false;
  }
  const auto currentOutputUsages =
      analysis.inUsages.at(outputName).at(currentOutputVersion);
  VLOG(2) << "Blob: " << outputName << ", idx: " << currentOutputVersion
          << ", usages[0]: " << currentOutputUsages[0];

  return currentOutputUsages == std::vector<size_t>{currentIdx + 1};
}
bool tryFuseAdjacentOps(
    const Analysis& analysis,
    size_t currentIdx,
    const OperatorDef& currentOp,
    const OperatorDef& nextOp,
    OperatorDef* fusedOp) {
  // Check for possible invalid opportunities.
  // Must be identical outputs, with either in-place usage for nextOp, *or* the
  // only use of the output of currentOp is the consumption by nextOp.
  if (currentOp.output_size() != 1 || !nextOp.input_size() ||
      nextOp.output_size() != 1) {
    return false;
  }

  if (currentOp.output(0) != nextOp.input(0)) {
    return false;
  }

  if (!nextIsOnlyUserOfCurrent(analysis, currentIdx, currentOp, nextOp)) {
    return false;
  }

  // Can we autogenerate this at registration time instead?
  static const std::map<std::pair<std::string, std::string>, std::string>
      fusionOpportunities = {{
          {{"MPSCNNConv", "MPSCNNRelu"}, "MPSCNNConvRelu"},
          {{"MPSCNNConv", "MPSCNNSigmoid"}, "MPSCNNConvSigmoid"},
          {{"MPSCNNFC", "MPSCNNRelu"}, "MPSCNNFCRelu"},
          {{"MPSCNNInstanceNorm", "MPSCNNPRelu"}, "MPSCNNInstanceNormPRelu"},
      }};
  auto it = fusionOpportunities.find({currentOp.type(), nextOp.type()});
  if (it == fusionOpportunities.end()) {
    return false;
  }
  // MPSCNNConvRelu and MPSCNNConvSigmoid cannot be in-place
  if (currentOp.type() == "MPSCNNConv" &&
      currentOp.input(0) == nextOp.output(0)) {
    return false;
  }
  LOG(INFO) << "Found a fusion between adjacent ops: (" << currentOp.type()
            << ", " << nextOp.type() << ") -> " << it->second;
  fusedOp->CopyFrom(currentOp);
  fusedOp->set_type(it->second);
  for (auto i = 1; i < nextOp.input_size(); ++i) {
    fusedOp->add_input(nextOp.input(i));
  }
  fusedOp->set_output(0, nextOp.output(0));
  return true;
}

NetDef runMPSCNNFusion(const NetDef& def) {
  CAFFE_ENFORCE_GE(def.op_size(), 1);
  NetDef mdef;
  mdef.CopyFrom(def);
  mdef.clear_op();
  auto i = 0;
  auto analysis = analyzeNet(def);

  while (i < def.op_size()) {
    if (i == def.op_size() - 1) {
      VLOG(2) << "Last operator, skipping";
      auto* op = mdef.add_op();
      op->CopyFrom(def.op(i));
      i += 1;
      continue;
    }

    const auto& currentOp = def.op(i);
    const auto& nextOp = def.op(i + 1);
    OperatorDef fusedOp;
    if (tryFuseAdjacentOps(analysis, i, currentOp, nextOp, &fusedOp)) {
      VLOG(2) << "Found an adjacent fusion at: " << i;
      // We can fuse.
      auto* op = mdef.add_op();
      op->CopyFrom(fusedOp);
      i += 2;
      continue;
    }
    VLOG(2) << "No fusion available";
    // Just emit the current type.
    auto* op = mdef.add_op();
    op->CopyFrom(currentOp);
    i += 1;
  }
  return mdef;
}

NetDef rewriteForMetal(const NetDef& def) {
  NetDef mdef;
  mdef.CopyFrom(def);

  const auto& opKeyList = CPUOperatorRegistry()->Keys();
  const auto& opKeySet =
      std::set<std::string>(opKeyList.begin(), opKeyList.end());
  for (auto i = 0; i < mdef.op_size(); ++i) {
    auto* op = mdef.mutable_op(i);
    const auto mpscnnOp = std::string("MPSCNN") + op->type();
    CAFFE_ENFORCE(opKeySet.find(mpscnnOp) != opKeySet.end());
    op->set_type(mpscnnOp);
  }

  mdef = runMPSCNNFusion(mdef);
  static std::set<std::string> mpscnnInputOps = {
      "CopyToMPSCNN", "MPSCNNPackedInt8BGRANHWCToNCHWCStylizerPreprocess"};
  static std::set<std::string> mpscnnOutputOps = {
      "CopyFromMPSCNN", "MPSCNNBRGNCHWCToPackedInt8BGRAStylizerDeprocess"};

  if (mpscnnInputOps.find(mdef.op(0).type()) == mpscnnInputOps.end() &&
      mpscnnOutputOps.find(mdef.op(mdef.op_size() - 1).type()) ==
          mpscnnOutputOps.end()) {
    mdef = insertInputOutputCopyOps(mdef);
  }
  CAFFE_ENFORCE_GE(mdef.op_size(), 2);
  CAFFE_ENFORCE(mpscnnInputOps.find(mdef.op(0).type()) != mpscnnInputOps.end());
  CAFFE_ENFORCE(
      mpscnnOutputOps.find(mdef.op(mdef.op_size() - 1).type()) !=
      mpscnnOutputOps.end());
  return mdef;
}

void dumpDef(const NetDef& d) {
  for (const auto& op : d.op()) {
    LOG(INFO) << op.input(0) << " -> " << op.type() << " -> " << op.output(0);
  }
}

NetDef annotateDefWithReadCounts(const NetDef& net) {
  // Now we have usage versions, we want to compute, for each blob version, the
  // number of usages of each blob version. ReadCount
  auto analysis = analyzeNet(net);
  using ReadCount = std::unordered_map<std::string, size_t>;
  std::vector<ReadCount> readCounts;

  auto computeReadCount = [&](size_t i, const OperatorDef& op) {
    ReadCount rcs;
    for (const auto bv : analysis.ssa[i].outVersions) {
      const auto versionUsages = analysis.inUsages[bv.first][bv.second];
      rcs[bv.first] = versionUsages.size();
    }
    readCounts.push_back(rcs);
  };
  for (auto i = 0; i < net.op_size(); ++i) {
    computeReadCount(i, net.op(i));
  }

  NetDef annotatedNet;
  annotatedNet.CopyFrom(net);
  for (auto i = 0; i < annotatedNet.op_size(); ++i) {
    auto* op = annotatedNet.mutable_op(i);
    // TODO - relax this? CAFFE_ENFORCE_EQ(op->output_size(), 1);
    const auto& blob = op->output(0);
    const size_t readCount = readCounts[i][blob];
    if (readCount > 1) {
      auto* arg = op->add_arg();
      arg->set_name(kMPSCNNReadCountArg);
      arg->set_i(readCount);
      LOG(INFO) << "Op: " << i << ", ty: " << op->type() << ", blob: " << blob
                << ", read count: " << readCount;
    }
  }
  return annotatedNet;
}

bool tryConvertToMPSCNN(
    const NetDef& initNet,
    const NetDef& predictNet,
    NetDef* metalPredictNet) {
  // iOS 10.0 and above.

#define SYSTEM_VERSION_GREATER_THAN_OR_EQUAL_TO(v) \
  ([[[UIDevice currentDevice] systemVersion]       \
       compare:v                                   \
       options:NSNumericSearch] != NSOrderedAscending)
  if (!SYSTEM_VERSION_GREATER_THAN_OR_EQUAL_TO(@"11.0")) {
    LOG(ERROR) << "MPSCNN is only supported for ios version above 11.0.";
    return false;
  }
#undef SYSTEM_VERSION_GREATER_THAN_OR_EQUAL_TO
  // The iOS GPU Family 3 v2 feature set. Introduced with the Apple A9 GPU and
  // iOS 10.0. Don't instantiate the MPSCNNContext, as that compiles the kernel
  // source.
  if (![MTLCreateSystemDefaultDevice()
          supportsFeatureSet:MTLFeatureSet_iOS_GPUFamily3_v2]) {
    LOG(ERROR) << "The iOS GPU is less than an A9, so MPSCNN is not available";
    return false;
  }

  try {
    // Instantiating the net and catching failures allows us to
    Workspace ws;
    ws.RunNetOnce(initNet);
    // Throws if unsupported operators are found.
    *metalPredictNet = rewriteForMetal(predictNet);
    *metalPredictNet = annotateDefWithReadCounts(*metalPredictNet);
    // Throws if unsupported parameters are found.
    ws.CreateNet(*metalPredictNet);
    LOG(INFO) << "MPSCNN is successfully enabled";
    return true;
  } catch (const std::exception& e) {
    LOG(ERROR) << "Caught exception trying to convert NetDef to MPSCNN: "
               << e.what();
    return false;
  }
}

void mpscnnRecordExecutionFinish() {
  [getMPSCNNContext().commandQueue insertDebugCaptureBoundary];
}
}
