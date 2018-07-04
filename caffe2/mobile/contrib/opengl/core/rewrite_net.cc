
#include "rewrite_net.h"
#include "caffe2/core/operator.h"
#include "caffe2/utils/proto_utils.h"
#include <unordered_map>
#include <unordered_set>

#ifdef CAFFE2_ANDROID
#include "../android/AndroidGLContext.h"
#endif

namespace caffe2 {

struct Analysis {
  struct SSA {
    using BlobVersions = std::unordered_map<std::string, size_t>;
    BlobVersions inVersions;
    BlobVersions outVersions;
  };
  std::vector<SSA> ssa;
  std::unordered_map<std::string, std::unordered_map<size_t, std::vector<size_t>>> inUsages;
};

static Analysis analyzeNet(const NetDef& net) {
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

static void insertCopyToGPUOp(NetDef& predictNet, const std::string& cpu_blob) {
  auto* op = predictNet.add_op();
  op->set_name("CopyToOpenGL");
  op->set_type("CopyToOpenGL");
  op->add_input(cpu_blob);
  op->add_output(cpu_blob + "_M");
}

static void insertCopyFromGPUOp(NetDef& predictNet, const std::string& cpu_blob) {
  // add argument "is_last" to the last op to signal this is the last operator before the
  // CopyFromOpenGL op
  auto* last_op = predictNet.mutable_op(predictNet.op_size() - 1);
  auto* arg = last_op->add_arg();
  arg->set_name("is_last");
  arg->set_i(1);

  auto* op = predictNet.add_op();
  op->set_name("CopyFromOpenGL");
  op->set_type("CopyFromOpenGL");
  op->add_input(cpu_blob + "_M");
  op->add_output(cpu_blob);
}

static NetDef insertInputOutputCopyOps(const NetDef& def, std::unordered_set<std::string>& glOps) {
  // Do some validation of the outputs. For this version, we require:
  // - a single input (first element of external_input()) is consumed by the NetDef
  // - a single output (first element of external_output()) is produced by the NetDef.
  // - the input is consumed by def.op(0), and this is the only consumer.
  // - the output is produced by def.op(-1).
  CAFFE_ENFORCE_GE(def.external_input_size(), 1);
  CAFFE_ENFORCE_GE(def.external_output_size(), 1);
  auto analysis = analyzeNet(def);
  // enforce a single use of the input blob.
  CAFFE_ENFORCE_GE(def.op_size(), 1);

  const auto& inputBlob = def.external_input(0);
  // Enforce that the input blob has a single usage - in the first operator.
  CAFFE_ENFORCE(analysis.inUsages[inputBlob][0] == (std::vector<size_t>{0}));
  // Enforce that the external_output(0) blob is produced by the last operator in this sequence.
  const auto& outputBlob = def.external_output(0);
  CAFFE_ENFORCE(analysis.ssa.back().outVersions.find(outputBlob) !=
                analysis.ssa.back().outVersions.end());
  const auto& outputBlobVersion = analysis.ssa.back().outVersions[outputBlob];
  // This should hold true by definition of the SSA analysis.
  CAFFE_ENFORCE(analysis.inUsages[outputBlob].find(outputBlobVersion) ==
                analysis.inUsages[outputBlob].end());

  NetDef mdef;
  mdef.CopyFrom(def);
  mdef.clear_op();

  std::unordered_map<std::string, std::set<size_t>> cpu_blobs, gpu_blobs;
  cpu_blobs[def.external_input(0)].insert(0);

  for (auto i = 0; i < def.op_size(); i++) {
    const auto& currentOp = def.op(i);
    if (glOps.count(currentOp.type()) > 0) {
      // OpenGL Op
      // insert copyToOpenGLOp
      for (auto j = 0; j < currentOp.input_size(); j++) {
        auto& input = currentOp.input(j);
        auto version = analysis.ssa[i].inVersions[input];
        if (cpu_blobs[input].count(version) > 0) {
          insertCopyToGPUOp(mdef, input);
          gpu_blobs[input].insert(version);
          cpu_blobs[input].erase(version);
        }
        // Only the first input should be OpenGL texture
        // Otherwise, copyToOpenGLOp will be inserted for the weights,
        // which are outputs of QuantDecode
        if (currentOp.type().find("OpenGLConv") == 0) {
          if (j == 0) {
            break;
          }
        }
      }

      auto* op = mdef.add_op();
      op->CopyFrom(currentOp);

      // swap input blob
      for (auto j = 0; j < currentOp.input_size(); j++) {
        auto& input = currentOp.input(j);
        auto version = analysis.ssa[i].inVersions[input];
        if (gpu_blobs[input].count(version) > 0) {
          op->set_input(j, input + "_M");
        }
      }

      // swap output blob
      for (auto j = 0; j < currentOp.output_size(); j++) {
        auto& output = currentOp.output(j);
        auto version = analysis.ssa[i].outVersions[output];
        op->set_output(j, output + "_M");
        gpu_blobs[output].insert(version);
      }
      // insert copyFromOpenGLOp after the last op if the last op is an OpenGL op
      if (i == def.op_size() - 1) {
        insertCopyFromGPUOp(mdef, currentOp.output(0));
      }
    } else {
      // CPU Op
      // insert copyFromOpenGLOp
      for (auto j = 0; j < currentOp.input_size(); j++) {
        auto& input = currentOp.input(j);
        auto version = analysis.ssa[i].inVersions[input];
        if (gpu_blobs[input].count(version) > 0) {
          insertCopyFromGPUOp(mdef, input);
        }
      }
      auto* op = mdef.add_op();
      op->CopyFrom(currentOp);
      for (auto j = 0; j < currentOp.output_size(); j++) {
        auto& output = currentOp.output(j);
        auto version = analysis.ssa[i].outVersions[output];
        cpu_blobs[output].insert(version);
      }
    }
  }
  return mdef;
}

static bool tryFuseAdjacentOps(const OperatorDef& currentOp,
                               const OperatorDef& nextOp,
                               OperatorDef* fusedOp,
                               std::unordered_set<std::string>& glOps) {
  // Check for possible invalid opportunities.
  if (currentOp.output_size() != 1 || nextOp.output_size() != 1) {
    return false;
  }
  // The fused op cannot be inplace
  if (currentOp.output(0) != nextOp.input(0) || currentOp.input(0) == nextOp.output(0)) {
    return false;
  }

  static const std::map<std::pair<std::string, std::string>, std::string> fusionOpportunities = {
      {{"OpenGLInstanceNorm", "OpenGLPRelu"}, "OpenGLInstanceNormPRelu"},
      {{"OpenGLConv", "OpenGLPRelu"}, "OpenGLConvPRelu"},
      {{"OpenGLConv", "OpenGLRelu"}, "OpenGLConvRelu"},
      {{"OpenGLConvTranspose", "OpenGLPRelu"}, "OpenGLConvTransposePRelu"}};
  auto it = fusionOpportunities.find({currentOp.type(), nextOp.type()});
  if (it == fusionOpportunities.end()) {
    return false;
  }

  glOps.insert(it->second);
  fusedOp->CopyFrom(currentOp);
  fusedOp->set_output(0, nextOp.output(0));
  fusedOp->set_type(it->second);
  for (auto i = 1; i < nextOp.input_size(); i++) {
    fusedOp->add_input(nextOp.input(i));
  }
  return true;
}

static NetDef runOpenGLFusion(const NetDef& def, std::unordered_set<std::string>& glOps) {
  CHECK_GE(def.op_size(), 1);
  NetDef mdef;
  mdef.CopyFrom(def);
  mdef.clear_op();
  auto i = 0;

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
    if (tryFuseAdjacentOps(currentOp, nextOp, &fusedOp, glOps)) {
      VLOG(2) << "Found an adjacent fusion for: " << currentOp.type() << ", " << nextOp.type();
      // We can fuse.
      auto* op = mdef.add_op();
      op->CopyFrom(fusedOp);
      i += 2;
      continue;
    }
    VLOG(2) << "No fusion available for: " << currentOp.type() << ", " << nextOp.type();
    // Just emit the current type.
    auto* op = mdef.add_op();
    op->CopyFrom(currentOp);
    i += 1;
  }
  return mdef;
}

void dumpDefForOpenGL(const NetDef& d) {
  for (const auto& op : d.op()) {
    LOG(INFO) << op.input(0) << " -> " << op.type() << " -> " << op.output(0);
  }
}

// // For debugging
// void dumpDefForOpenGL(const NetDef &net) {
//  for (const auto &op : net.op()) {
//    printf("***Operator: %s\n", op.type().c_str());
//    for (auto input : op.input()) {
//      printf("\tInput: %s\n", input.c_str());
//    }
//
//    for (auto output : op.output()) {
//      printf("\tOutput: %s\n", output.c_str());
//    }
//  }
//}

NetDef rewritePredictNetForOpenGL(const NetDef& predictNet, bool useTextureInput, bool useTiling, bool runFusion) {
  CAFFE_ENFORCE_GE(predictNet.op_size(), 1);
  NetDef net;
  net.CopyFrom(predictNet);

  std::unordered_map<std::string, std::string> replacements(
      {{"OpenGLPackedInt8BGRANHWCToNCHWCStylizerPreprocess",
        useTextureInput ? "OpenGLTextureToTextureStylizerPreprocess"
                        : "OpenGLTensorToTextureStylizerPreprocess"},
       {"OpenGLBRGNCHWCToPackedInt8BGRAStylizerDeprocess",
        useTextureInput ? "OpenGLTextureToTextureStylizerDeprocess"
                        : "OpenGLTextureToTensorStylizerDeprocess"}});

  std::unordered_set<std::string> openGLOps; // Used to insert copy ops
  bool needCopyOps = false;

  const auto& opKeyList = CPUOperatorRegistry()->Keys();
  auto opKeySet = std::set<std::string>(opKeyList.begin(), opKeyList.end());

#ifdef CAFFE2_ANDROID
  // TODO: debug InstanceNorm models on Mali devices
  AndroidGLContext* context = (AndroidGLContext*)GLContext::getGLContext();
  if (context->get_platform() == Mali) {
    opKeySet.erase("OpenGLInstanceNorm");
    opKeySet.erase("OpenGLInstanceNormPRelu");
  }
#endif
  for (auto i = 0; i < net.op_size(); ++i) {
    auto* op = net.mutable_op(i);
    string openGLOp = std::string("OpenGL") + op->type();
    if (replacements.count(openGLOp) > 0) {
      openGLOp = replacements[openGLOp];
    }

    if (opKeySet.find(openGLOp) != opKeySet.end()) {
      op->set_type(openGLOp);
      openGLOps.insert(openGLOp);

      if (useTiling) {
        auto* arg = op->add_arg();
        arg->set_name("tiling");
        arg->set_i(1);
      }
    } else {
      needCopyOps = true;
    }
  }

  if (useTextureInput && needCopyOps) {
    CAFFE_THROW("OpenGL operator missing");
  }

  if (runFusion) {
    net = runOpenGLFusion(net, openGLOps);
  }

  if (net.op(0).type() == replacements["OpenGLPackedInt8BGRANHWCToNCHWCStylizerPreprocess"]) {
    // For end-to-end testing
    if (net.op(net.op_size() - 1).type() !=
        replacements["OpenGLBRGNCHWCToPackedInt8BGRAStylizerDeprocess"]) {
      auto* last_op = net.mutable_op(net.op_size() - 1);
      auto output = last_op->output(0) + "M";
      last_op->set_output(0, output);
      auto* copy_op = net.add_op();
      copy_op->set_name("CopyFromOpenGL");
      copy_op->set_type("CopyFromOpenGL");
      copy_op->add_input(output);
      // rename output blob in case input and output blob has the same name
      copy_op->add_output(net.external_output(0));
    }
  } else {
    if (!useTextureInput) {
      needCopyOps = true;
    }
  }

  // copy ops are needed when the input is not a texture
  if (needCopyOps) {
    // For non style transfer cases
    net = insertInputOutputCopyOps(net, openGLOps);
  }

  return net;
}

bool tryConvertToOpenGL(const NetDef& initNet,
                        const NetDef& predictNet,
                        NetDef* glPredictNet,
                        bool useTextureInput,
                        bool useTiling,
                        bool runFusion) {
  try {
    // Throws if unsupported operators are found.
    *glPredictNet = rewritePredictNetForOpenGL(predictNet, useTextureInput, useTiling, runFusion);
    dumpDefForOpenGL(*glPredictNet);
    // Throws if unsupported parameters are found.
    Workspace ws;
    ws.RunNetOnce(initNet);
    ws.CreateNet(*glPredictNet);
    LOG(INFO) << "OpenGL is successfully enabled";
    return true;
  } catch (const std::exception& e) {
    LOG(ERROR) << "Caught exception trying to convert NetDef to OpenGL: " << e.what();
    return false;
  }
}
} // namespace caffe2
