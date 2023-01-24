#include "mpscnn_graph_mask.h"
#include "caffe2/core/operator.h"
#include "mpscnn_context.h"

#import <Metal/Metal.h>
#import <MetalPerformanceShaders/MetalPerformanceShaders.h>
#import <UIKit/UIDevice.h>

namespace caffe2 {

namespace {
enum class StorageType {
  MPSTEMPORARYIMAGE, /* Default for MPSCNN */
  MPSIMAGE,
  CPU,
  INVALID
};

string asString(StorageType st) {
  switch (st) {
  case StorageType::MPSTEMPORARYIMAGE:
    return "MPSTEMPORARYIMAGE";
  case StorageType::MPSIMAGE:
    return "MPSIMAGE";
  case StorageType::CPU:
    return "CPU";
  case StorageType::INVALID:
    return "INVALID";
  }
}

bool isImage(StorageType type) {
  return type == StorageType::MPSTEMPORARYIMAGE ||
      type == StorageType::MPSIMAGE;
}

std::unordered_map<string, std::vector<StorageType>> inputStorageTypeMap = {
    {"MPSCNNGenerateProposalsCPP",
     std::vector<StorageType>{StorageType::CPU,
                              StorageType::CPU,
                              StorageType::CPU,
                              StorageType::CPU}},
    {"MPSCNNRoIWarp",
     std::vector<StorageType>{StorageType::MPSTEMPORARYIMAGE,
                              StorageType::CPU}},
    {"MPSCNNConvRelu",
     std::vector<StorageType>{StorageType::MPSTEMPORARYIMAGE,
                              StorageType::CPU,
                              StorageType::CPU}},
    {"MPSCNNFC",
     std::vector<StorageType>{StorageType::MPSTEMPORARYIMAGE,
                              StorageType::CPU,
                              StorageType::CPU}},
    {"MPSCNNConv",
     std::vector<StorageType>{StorageType::MPSTEMPORARYIMAGE,
                              StorageType::CPU,
                              StorageType::CPU}},
    {"MPSCNNConvTranspose",
     std::vector<StorageType>{StorageType::MPSTEMPORARYIMAGE,
                              StorageType::CPU,
                              StorageType::CPU}},
    {"MPSCNNMul",
     std::vector<StorageType>{StorageType::MPSTEMPORARYIMAGE,
                              StorageType::CPU}},
    {"MPSCNNSub",
     std::vector<StorageType>{StorageType::MPSTEMPORARYIMAGE,
                              StorageType::CPU}},
    {"MPSCNNNormalizePlanarYUV",
     std::vector<StorageType>{StorageType::MPSTEMPORARYIMAGE,
                              StorageType::CPU,
                              StorageType::CPU}}};
std::unordered_map<string, std::vector<StorageType>> outputStorageTypeMap = {
    {"MPSCNNGenerateProposalsCPP", std::vector<StorageType>{StorageType::CPU, StorageType::CPU}}};
std::vector<string> opsNeedsSync = {"MPSCNNGenerateProposalsCPP", "CopyFromMPSCNN", "CopyToMPSCNN"};

struct Analysis {
  struct SSA {
    using BlobVersions = std::unordered_map<std::string, size_t>;
    BlobVersions inVersions;
    BlobVersions outVersions;
  };
  struct BlobInfo {
    std::vector<size_t> inUsages; // ids for operator that used the blob
    StorageType storageType = StorageType::INVALID; // storage type of the blob
    int commandBufferId; // the id for command buffer used by the blob
  };
  std::vector<SSA> ssa;
  // blob name -> blob version -> blob information
  std::unordered_map<std::string, std::unordered_map<size_t, BlobInfo>> blobInfoMap;
  int currentCommandBufferId = 0;
};

void ssaAnalysis(Analysis& analysis, const NetDef& net) {
  Analysis::SSA::BlobVersions frontier;

  auto play = [&](size_t i, const OperatorDef& op) {
    Analysis::SSA::BlobVersions inVersions;
    for (const auto& s : op.input()) {
      inVersions[s] = frontier[s];
      analysis.blobInfoMap[s][frontier[s]].inUsages.push_back(i);
    }
    Analysis::SSA::BlobVersions outVersions;
    auto isTemporaryImages = std::vector<int>();
    for (auto j = 0; j < op.arg_size(); ++j) {
      if (op.arg(j).name() == kMPSCNNOutputIsTempImageArg) {
        for (auto k = 0; k < op.arg(j).ints_size(); ++k) {
          isTemporaryImages.push_back(op.arg(j).ints(k));
        }
      }
    }

    for (auto j = 0; j < op.output_size(); j++) {
      auto s = op.output(j);
      if (frontier.find(s) != frontier.end()) {
        frontier[s] += 1;
      }
      outVersions[s] = frontier[s];
      if (outputStorageTypeMap.find(op.type()) != outputStorageTypeMap.end()) {
        analysis.blobInfoMap[s][frontier[s]].storageType = outputStorageTypeMap[op.type()][j];
      } else if (op.type() == "CopyFromMPSCNN") {
        analysis.blobInfoMap[s][frontier[s]].storageType = StorageType::CPU;
      } else if (isTemporaryImages.size() > 0) {
        if (isTemporaryImages.at(j)) {
          analysis.blobInfoMap[s][frontier[s]].storageType = StorageType::MPSTEMPORARYIMAGE;
        } else {
          analysis.blobInfoMap[s][frontier[s]].storageType = StorageType::MPSIMAGE;
        }
      } else if (op.type().find("MPSCNN") != std::string::npos) {
        analysis.blobInfoMap[s][frontier[s]].storageType = StorageType::MPSTEMPORARYIMAGE;
      } else {
        analysis.blobInfoMap[s][frontier[s]].storageType = StorageType::CPU;
      }
      VLOG(2) << op.type() << " outputBlobTypes:" << s << " " << frontier[s]
              << " "
              << asString(analysis.blobInfoMap[s][frontier[s]].storageType);
    }
    analysis.ssa.push_back(Analysis::SSA{inVersions, outVersions});
  };

  for (auto i = 0; i < net.op_size(); ++i) {
    play(i, net.op(i));
  }
}

static void rewriteOutput(OperatorDef* op, int i) {
  auto output = op->output(i);
  op->set_output(i, output + "_M");
}

static void rewriteInput(OperatorDef* op, int i) {
  auto input = op->input(i);
  op->set_input(i, input + "_I");
}

static void insertOutputCopyFromMPSCNNOp(NetDef& predictNet, const std::string& cpu_blob) {
  auto* op = predictNet.add_op();
  op->set_type("CopyFromMPSCNN");
  op->add_input(cpu_blob + "_M");
  op->add_output(cpu_blob);
}

static void insertInputCopyFromMPSCNNOp(NetDef& predictNet, const std::string& cpu_blob) {
  auto* op = predictNet.add_op();
  op->set_type("CopyFromMPSCNN");
  op->add_input(cpu_blob);
  op->add_output(cpu_blob + "_I");
}

static void insertInputCopyToMPSCNNOp(NetDef& predictNet, const std::string& gpu_blob) {
  auto* op = predictNet.add_op();
  op->set_type("CopyToMPSCNN");
  op->add_input(gpu_blob);
  op->add_output(gpu_blob + "_I");
}

void commandBufferAnalysis(Analysis& analysis, NetDef& def) {
  analysis.currentCommandBufferId = 0;
  analysis.blobInfoMap[def.op(0).input(0)][0].commandBufferId = analysis.currentCommandBufferId;
  for (auto i = 0; i < def.op_size(); ++i) {
    auto op = def.op(i);
    if (std::find(opsNeedsSync.begin(), opsNeedsSync.end(), op.type()) != opsNeedsSync.end()) {
      analysis.currentCommandBufferId += 1;
      for (auto j = 0; j < op.output_size(); ++j) {
        auto outputBlob = op.output(j);
        auto version = analysis.ssa[i].outVersions[outputBlob];
        analysis.blobInfoMap[outputBlob][version].commandBufferId = analysis.currentCommandBufferId;
      }
    } else {
      int inputCommandBufferId = 0;
      for (auto j = 0; j < op.input_size(); ++j) {
        auto inputBlob = op.input(j);
        auto version = analysis.ssa[i].inVersions[inputBlob];
        if (analysis.blobInfoMap.find(inputBlob) != analysis.blobInfoMap.end() &&
            analysis.blobInfoMap[inputBlob][version].storageType == StorageType::MPSIMAGE) {
          analysis.currentCommandBufferId += 1;
          inputCommandBufferId = analysis.currentCommandBufferId;
        } else {
          inputCommandBufferId =
              fmax(inputCommandBufferId, analysis.blobInfoMap[inputBlob][version].commandBufferId);
        }
      }
      // command buffer same as input
      for (auto j = 0; j < op.output_size(); ++j) {
        auto outputBlob = op.output(j);
        auto version = analysis.ssa[i].outVersions[outputBlob];
        analysis.blobInfoMap[outputBlob][version].commandBufferId = inputCommandBufferId;
      }
    }
    for (auto j = 0; j < op.output_size(); ++j) {
      auto outputBlob = op.output(j);
      auto version = analysis.ssa[i].outVersions[outputBlob];
      VLOG(2) << "command buffer analysis: " << outputBlob << " " << version << " "
              << analysis.blobInfoMap[outputBlob][version].commandBufferId;
    }
  }
}

void analyzeNet(Analysis& analysis, NetDef& net) {
  analysis.ssa.clear();
  analysis.blobInfoMap.clear();
  ssaAnalysis(analysis, net);
  commandBufferAnalysis(analysis, net);
}

NetDef mergeCopyFromMPSCNN(Analysis& analysis, NetDef& def) {
  analyzeNet(analysis, def);
  // command buffer id -> op id
  std::unordered_map<int, std::vector<size_t>> commandBufferToOps;
  // For CopyFromMPSCNN, find the command buffer id each input blob uses. and
  // aggreagate the ops with the same command buffer
  for (auto i = 0; i < def.op_size(); ++i) {
    auto op = def.op(i);
    if (op.type() == "CopyFromMPSCNN") {
      auto blobName = op.input(0);
      auto version = analysis.ssa[i].inVersions[blobName];
      auto commandId = analysis.blobInfoMap[blobName][version].commandBufferId;
      VLOG(2) << "Command buffer to ops:" << blobName << " " << version << " " << commandId;
      if (commandBufferToOps.find(commandId) == commandBufferToOps.end()) {
        commandBufferToOps[commandId] = std::vector<size_t>();
      }
      commandBufferToOps[commandId].push_back(i);
    }
  }

  std::vector<size_t> opsToRemove;
  for (auto item : commandBufferToOps) {
    auto commandBufferId = item.first;
    auto ops = item.second;
    if (ops.size() > 1) {
      VLOG(2) << "Merging for command buffer:" << commandBufferId;
      // Let's use the first input as an indicator whether the data is for
      // external output or internal use, if the data used by intermediate node,
      // we want to keep the first operator, otherwise, we want to keep
      // the last operator.
      // [LATER]There might be cases when some of the data is for external output and
      // others used by intermediate node, we'll need to have better heuristics
      // for these cases.
      auto externalUse = false;
      auto firstCopy = def.op(ops[0]);
      auto firstOutput = firstCopy.output(0);
      for (auto i = 0; i < def.external_output_size(); ++i) {
        if (def.external_output(i) == firstOutput) {
          externalUse = true;
        }
      }
      int removeStart, removeEnd, keepIndex;
      if (externalUse) {
        // change the last op into the new op and remove the other ops;
        removeStart = 0;
        removeEnd = ops.size() - 1;
        keepIndex = ops[removeEnd];
      } else {
        removeStart = 1;
        removeEnd = ops.size();
        keepIndex = ops[removeStart - 1];
      }
      auto* op = def.mutable_op(keepIndex);
      auto inputOutputs = std::set<std::pair<string, string>>();
      for (auto i = removeStart; i < removeEnd; ++i) {
        auto op0 = def.op(ops[i]);
        if (op0.input(0) != op->input(0)) {
          inputOutputs.insert(make_pair(op0.input(0), op0.output(0)));
        }
      }
      for (auto inputOutput : inputOutputs) {
        op->add_input(inputOutput.first);
        op->add_output(inputOutput.second);
      }
      for (auto i = removeStart; i < removeEnd; ++i) {
        opsToRemove.push_back(ops[i]);
      }
    }
  }

  NetDef mdef;
  mdef.CopyFrom(def);
  mdef.clear_op();
  for (auto i = 0; i < def.op_size(); ++i) {
    if (std::find(opsToRemove.begin(), opsToRemove.end(), i) == opsToRemove.end()) {
      const auto& ogOp = def.op(i);
      auto op = mdef.add_op();
      op->CopyFrom(ogOp);
    }
  }
  return mdef;
}

/* Remove the CopyToMPSCNN ops that has the same input/output version
 */
NetDef mergeCopyToMPSCNN(Analysis& analysis, NetDef& def) {
  std::vector<size_t> opsToRemove;
  std::set<std::pair<string, size_t>> copiedBlobs;
  for (auto i = 0; i < def.op_size(); ++i) {
    auto op = def.op(i);
    if (def.op(i).type() == "CopyToMPSCNN") {
      auto blobName = op.input(0);
      auto version = analysis.ssa[i].inVersions[blobName];
      auto pair = make_pair(blobName, version);
      if (std::find(copiedBlobs.begin(), copiedBlobs.end(), pair) == copiedBlobs.end()) {
        copiedBlobs.insert(pair);
      } else {
        opsToRemove.push_back(i);
      }
    }
  }
  NetDef mdef;
  mdef.CopyFrom(def);
  mdef.clear_op();
  for (auto i = 0; i < def.op_size(); ++i) {
    if (std::find(opsToRemove.begin(), opsToRemove.end(), i) == opsToRemove.end()) {
      const auto& ogOp = def.op(i);
      auto op = mdef.add_op();
      op->CopyFrom(ogOp);
    }
  }
  return mdef;
}

bool addTempImageArgs(Analysis& analysis, NetDef& def) {
  analyzeNet(analysis, def);

  std::vector<int> synced; // synced command buffer ids;
  std::set<std::pair<string, size_t>> mpsImageBlobs; // blobname, version

  // We want to add temp arg one by one since it changes the command buffer id
  // for later operators.
  bool found = false;
  // identify the images that the command buffer is synced before
  for (auto i = 0; i < def.op_size(); ++i) {
    auto op = def.op(i);
    if (op.type().find("MPSCNN") == string::npos) {
      continue;
    }
    for (auto j = 0; j < op.input_size(); ++j) {
      auto inputBlob = op.input(j);
      auto version = analysis.ssa[i].inVersions[inputBlob];
      auto commandId = analysis.blobInfoMap[inputBlob][version].commandBufferId;
      if (std::find(opsNeedsSync.begin(), opsNeedsSync.end(), op.type()) != opsNeedsSync.end()) {
        synced.push_back(commandId);
        break;
      }
      if (std::find(synced.begin(), synced.end(), commandId) != synced.end() &&
          analysis.blobInfoMap.find(inputBlob) != analysis.blobInfoMap.end() &&
          analysis.blobInfoMap[inputBlob][version].storageType == StorageType::MPSTEMPORARYIMAGE) {
        VLOG(2) << "mpsimage blob:" << inputBlob << " " << version << " "
                << "input " << j << " command: " << commandId;
        mpsImageBlobs.insert(make_pair(inputBlob, version));
        found = true;
      }
    }
    if (found) {
      break;
    }
  }
  // find the blob and add argument
  if (found) {
    for (auto i = 0; i < def.op_size(); ++i) {
      auto op = def.mutable_op(i);
      std::vector<int> isTempImages;
      bool setArg = false;
      for (auto j = 0; j < op->output_size(); ++j) {
        auto outputBlob = op->output(j);
        auto version = analysis.ssa[i].outVersions[outputBlob];
        if (mpsImageBlobs.find(make_pair(outputBlob, version)) != mpsImageBlobs.end()) {
          setArg = true;
          isTempImages.push_back(0);
        } else {
          isTempImages.push_back(1);
        }
      }
      if (setArg) {
        auto& arg = *(op->add_arg());
        arg.set_name(kMPSCNNOutputIsTempImageArg);
        for (auto j = 0; j < isTempImages.size(); ++j) {
          arg.add_ints(isTempImages[j]);
        }
      }
    }
  }
  return found;
}

NetDef insertCopies(const NetDef& def) {
  // For this version, we insert CopyFromMPSCNN both for
  // intermediate nodes and the output node when necessary
  CAFFE_ENFORCE_GE(def.external_input_size(), 1);
  CAFFE_ENFORCE_GE(def.external_output_size(), 1);

  Analysis analysis;
  ssaAnalysis(analysis, def);

  CAFFE_ENFORCE_GE(def.op_size(), 1);

  const auto& outputBlob = def.external_output(0);
  const auto& outputBlobVersion = analysis.ssa.back().outVersions[outputBlob];

  // This should hold true by definition of the SSA analysis.
  CAFFE_ENFORCE(analysis.blobInfoMap[outputBlob].find(outputBlobVersion) ==
                    analysis.blobInfoMap[outputBlob].end() ||
                analysis.blobInfoMap[outputBlob][outputBlobVersion].inUsages.size() == 0);
  NetDef mdef;
  mdef.CopyFrom(def);
  mdef.clear_op();

  const auto& opKeyList = CPUOperatorRegistry()->Keys();
  const auto& opKeySet = std::set<std::string>(opKeyList.begin(), opKeyList.end());

  for (auto i = 0; i < def.op_size(); ++i) {
    const auto& ogOp = def.op(i);
    auto inputsToRewrite = std::vector<int>();

    for (auto j = 0; j < ogOp.input_size(); j++) {
      // The blob storage type accepted by the operator
      auto expectedBlobType = StorageType::MPSTEMPORARYIMAGE;
      // The storage type for blob produced by previous operators
      // if it's not produced by previous operators, then it should be network
      // parameters which are stored in CPU
      auto actualBlobType = StorageType::CPU;
      // For non-mpscnn operators, we assume the expected storage type to be CPU
      if (ogOp.type().find("MPSCNN") == std::string::npos) {
        expectedBlobType = StorageType::CPU;
      }
      auto inputBlob = ogOp.input(j);
      auto version = analysis.ssa[i].inVersions[inputBlob];
      // Check whether the blob is produced by previous operators
      if (analysis.blobInfoMap.find(inputBlob) != analysis.blobInfoMap.end() &&
          analysis.blobInfoMap[inputBlob][version].storageType != StorageType::INVALID) {
        actualBlobType = analysis.blobInfoMap[inputBlob][version].storageType;
        VLOG(2) << "Found " << inputBlob << " " << j << " with type"
                << asString(actualBlobType);
      }
      if (inputStorageTypeMap.find(ogOp.type()) != inputStorageTypeMap.end()) {
        expectedBlobType = inputStorageTypeMap[ogOp.type()][j];
      }
      if (expectedBlobType != actualBlobType) {
        if (expectedBlobType == StorageType::CPU && (isImage(actualBlobType))) {
          // copy input(MPSCNN) to input_I(CPU)
          insertInputCopyFromMPSCNNOp(mdef, ogOp.input(j));
          // rewrite input to input_I for the operator
          inputsToRewrite.push_back(j);
        } else if (
            isImage(expectedBlobType) && actualBlobType == StorageType::CPU) {
          insertInputCopyToMPSCNNOp(mdef, ogOp.input(j));
          inputsToRewrite.push_back(j);
        } // We don't need to insert copies in other cases
      }
    }

    auto op = mdef.add_op();
    op->CopyFrom(ogOp);

    for (auto j = 0; j < inputsToRewrite.size(); ++j) {
      rewriteInput(op, inputsToRewrite[j]);
    }

    // rewrite name for (single) external input
    if (op->type().find("MPSCNN") != std::string::npos &&
        opKeySet.find(op->type()) != opKeySet.end()) {
      // input used by multiple ops
      const auto& inputBlob = def.external_input(0);
      if (std::find(analysis.blobInfoMap[inputBlob][0].inUsages.begin(),
                    analysis.blobInfoMap[inputBlob][0].inUsages.end(),
                    i) != analysis.blobInfoMap[inputBlob][0].inUsages.end()) {
        for (auto j = 0; j < op->input_size(); ++j) {
          if (op->input(j) == def.external_input(0)) {
            op->set_input(j, "__METAL_INPUT_COPY__");
          }
        }
      }
    }

    // if the output is in external output, copy from metal when necessary
    for (auto j = 0; j < op->output_size(); ++j) {
      for (auto k = 0; k < def.external_output_size(); ++k) {
        // Assuming external output blob has unique name, e.g. only version 0
        // of the blob is used as the output
        if (op->output(j) == def.external_output(k) &&
            analysis.blobInfoMap[op->output(j)][0].storageType != StorageType::CPU) {
          // copy output_M(MPSCNN) to output(CPU)
          insertOutputCopyFromMPSCNNOp(mdef, op->output(j));
          // rewrite output to output_M for the operator
          rewriteOutput(op, j);
        }
      }
    }
  }

  // Since adding temp image arg changes the result for command buffer analysis,
  // which is the analysis the function is based on, we'll add one temp image
  // arg at a time and re-run ssa analysis after each and repeat the process
  // until convergence
  int i = 0;
  while (addTempImageArgs(analysis, mdef) && i < 3 * mdef.op_size()) {
    i++;
  };

  mdef = mergeCopyFromMPSCNN(analysis, mdef);
  mdef = mergeCopyToMPSCNN(analysis, mdef);

  return mdef;
}

NetDef rewriteForMetalI(const NetDef& def) {
  NetDef mdef;
  mdef.CopyFrom(def);

  const auto& opKeyList = CPUOperatorRegistry()->Keys();
  const auto& opKeySet = std::set<std::string>(opKeyList.begin(), opKeyList.end());
  for (auto i = 0; i < mdef.op_size(); ++i) {
    auto* op = mdef.mutable_op(i);
    const auto mpscnnOp = std::string("MPSCNN") + op->type();
    if (opKeySet.find(mpscnnOp) != opKeySet.end()) {
      op->set_type(mpscnnOp);
    }
  }

  static std::set<std::string> mpscnnInputOps = {
      "CopyToMPSCNN", "MPSCNNPackedInt8BGRANHWCToNCHWCStylizerPreprocess"};

  mdef = insertCopies(mdef);

  mdef = runMPSCNNFusion(mdef);

  mdef = setSpecialArgs(mdef);

  CAFFE_ENFORCE_GE(mdef.op_size(), 2);
  CAFFE_ENFORCE(mpscnnInputOps.find(mdef.op(0).type()) != mpscnnInputOps.end());
  return mdef;
}
} // namespace

NetDef setSpecialArgs(const NetDef& def) {
  NetDef mdef;
  mdef.CopyFrom(def);
  for (auto i = 0; i < mdef.op_size(); ++i) {
    auto* op = mdef.mutable_op(i);
    // setting post_nms_top_N for MPSCNNGenerateProposalsCPP to 36 due to the
    // texture array length constraint in RoIWarp
    if (op->type() == "MPSCNNGenerateProposalsCPP" || op->type() == "GenerateProposalsCPP") {
      auto* arg = op->mutable_arg(0);
      arg->set_i(36);
    }
  }
  return mdef;
}

bool tryConvertToMPSCNNIntermediateCopies(const NetDef& initNet,
                                          const NetDef& predictNet,
                                          NetDef* metalPredictNet) {
// iOS 10.0 and above.
#define SYSTEM_VERSION_GREATER_THAN_OR_EQUAL_TO(v)                                 \
  ([[[UIDevice currentDevice] systemVersion] compare:v options:NSNumericSearch] != \
   NSOrderedAscending)
#define SYSTEM_VERSION_EQUAL_TO(v) \
  ([[[UIDevice currentDevice] systemVersion] compare:v options:NSNumericSearch] == NSOrderedSame)

  if (!SYSTEM_VERSION_GREATER_THAN_OR_EQUAL_TO(@"11.0")) {
    LOG(ERROR) << "MPSCNN is only supported for ios version above 11.0.";
    return false;
  }
#undef SYSTEM_VERSION_GREATER_THAN_OR_EQUAL_TO
#undef SYSTEM_VERSION_EQUAL_TO

  // The iOS GPU Family 3 v2 feature set. Introduced with the Apple A9 GPU and iOS 10.0.
  // Don't instantiate the MPSCNNContext, as that compiles the kernel source.
  if (![MTLCreateSystemDefaultDevice() supportsFeatureSet:MTLFeatureSet_iOS_GPUFamily3_v2]) {
    LOG(ERROR) << "The iOS GPU is less than an A9, so MPSCNN is not available";
    return false;
  }

  try {
    // Instantiating the net and catching failures allows us to
    Workspace ws;
    ws.RunNetOnce(initNet);
    // Throws if unsupported operators are found.
    *metalPredictNet = rewriteForMetalI(predictNet);
    *metalPredictNet = annotateDefWithReadCounts(*metalPredictNet);
    // Throws if unsupported parameters are found.
    ws.CreateNet(*metalPredictNet);
    LOG(INFO) << "MPSCNN is successfully enabled";
    return true;
  } catch (const std::exception& e) {
    LOG(ERROR) << "Caught exception trying to convert NetDef to MPSCNN: " << e.what();
    return false;
  }
}
} // caffe2
