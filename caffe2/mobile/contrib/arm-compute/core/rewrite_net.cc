
#include "rewrite_net.h"
#include "caffe2/core/operator.h"
#include "caffe2/utils/proto_utils.h"
#include <unordered_map>

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

static void insertCopyFromGLOp(NetDef& predictNet, const std::string& cpu_blob) {
  auto* op = predictNet.add_op();
  op->set_name("CopyFromGL");
  op->set_type("CopyFromGL");
  op->add_input(cpu_blob + "_M");
  op->add_output(cpu_blob);
}

static NetDef insertInputOutputCopyOps(const NetDef& def, std::unordered_set<std::string>& cpuOp) {
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
    if (cpuOp.count(currentOp.type()) > 0) {
      // CPU Op
      // insert copyFromOpenGLOp
      for (auto j = 0; j < currentOp.input_size(); j++) {
        auto& input = currentOp.input(j);
        auto version = analysis.ssa[i].inVersions[input];
        if (gpu_blobs[input].count(version) > 0) {
          insertCopyFromGLOp(mdef, input);
        }
      }
      auto* op = mdef.add_op();
      op->CopyFrom(currentOp);
      for (auto j = 0; j < currentOp.output_size(); j++) {
        auto& output = currentOp.output(j);
        auto version = analysis.ssa[i].outVersions[output];
        cpu_blobs[output].insert(version);
      }
    } else {
      // OpenGL Op
      auto* op = mdef.add_op();
      op->CopyFrom(currentOp);

     for (auto j = 0; j < op->input_size(); j++) {
        auto* input = op->mutable_input(j);
        auto version = analysis.ssa[i].inVersions[*input];
        if (gpu_blobs[*input].count(version) > 0) {
          *input = *input + "_M";
        }
      }

      for (auto j = 0; j < currentOp.output_size(); j++) {
        auto& output = currentOp.output(j);
        auto version = analysis.ssa[i].outVersions[output];
        gpu_blobs[output].insert(version);
        // add _M to intermediate OpenGL op outputs
        auto* output_ = op->mutable_output(j);
        bool inter = true;
        for(auto k = 0; k < def.external_output_size(); k++) {
          if (*output_ == def.external_output(k)) {
            inter = false;
          }
        }
        if (inter) {
          *output_ = *output_ + "_M";
        }
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

NetDef rewritePredictNetForOpenGL(const NetDef& predictNet, bool runFusion, std::unordered_set<std::string> cpuOps) {
  CAFFE_ENFORCE_GE(predictNet.op_size(), 1);
  NetDef net;
  net.CopyFrom(predictNet);

  // if (runFusion) {
  //   net = runOpenGLFusion(net, openGLOps);
  // }

  net = insertInputOutputCopyOps(net, cpuOps);
  net.set_type("opengl");

  for (auto i = 0; i < net.op().size(); ++i) {
    auto op = net.mutable_op(i);
    if (std::find(cpuOps.begin(), cpuOps.end(), op->type()) == cpuOps.end()) {
      op->mutable_device_option()->set_device_type(OPENGL);
    }
  }

  return net;
}

bool tryConvertToOpenGL(const NetDef& predictNet,
                        NetDef* glPredictNet,
                        bool runFusion,
                        std::unordered_set<std::string> cpuOps) {
  try {
    // Throws if unsupported operators are found.
    *glPredictNet = rewritePredictNetForOpenGL(predictNet, runFusion, cpuOps);
    dumpDefForOpenGL(*glPredictNet);
    // Throws if unsupported parameters are found.
    LOG(INFO) << "OpenGL is successfully enabled";
    return true;
  } catch (const std::exception& e) {
    LOG(ERROR) << "Caught exception trying to convert NetDef to OpenGL: " << e.what();
    return false;
  }
}
} // namespace caffe2
