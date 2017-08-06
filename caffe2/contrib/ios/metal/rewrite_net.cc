// Copyright 2004-present Facebook. All Rights Reserved.

#include "rewrite_net.h"
#include "caffe2/utils/proto_utils.h"

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

Analysis analyzeNet(const NetDef &net) {
  Analysis::SSA::BlobVersions frontier;
  Analysis analysis;

  auto play = [&](size_t i, const OperatorDef &op) {
    Analysis::SSA::BlobVersions inVersions;
    for (const auto &s : op.input()) {
      inVersions[s] = frontier[s];
      analysis.inUsages[s][frontier[s]].push_back(i);
    }
    Analysis::SSA::BlobVersions outVersions;
    for (const auto &s : op.output()) {
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
  
void insertCopyToMetalGPUOp(NetDef &predictNet, const std::string &cpu_blob) {
  auto *op = predictNet.add_op();
  op->set_name("CopyFromCPUToGPU");
  op->set_type("CopyToMetalGPU");
  op->set_engine("METAL");
  op->add_input(cpu_blob);
  op->add_output(cpu_blob + "_M");
}

void insertCopyFromMetalGPUOp(NetDef &predictNet, const std::string &cpu_blob) {
  auto *op = predictNet.add_op();
  op->set_name("CopyFromGPUToCPU");
  op->set_type("CopyFromMetalGPU");
  op->add_input(cpu_blob + "_M");
  op->add_output(cpu_blob);
}

NetDef insertInputOutputCopyOps(const NetDef &def, std::string engine) {
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

  const auto &inputBlob = def.external_input(0);
  // Enforce that the input blob has a single usage - in the first operator.
  CAFFE_ENFORCE(analysis.inUsages[inputBlob][0] == (std::vector<size_t>{0}));
  // Enforce that the external_output(0) blob is produced by the last operator in this sequence.
  const auto &outputBlob = def.external_output(0);
  CAFFE_ENFORCE(analysis.ssa.back().outVersions.find(outputBlob) != analysis.ssa.back().outVersions.end());
  const auto &outputBlobVersion = analysis.ssa.back().outVersions[outputBlob];
  // This should hold true by definition of the SSA analysis.
  CAFFE_ENFORCE(analysis.inUsages[outputBlob].find(outputBlobVersion) == analysis.inUsages[outputBlob].end());

  NetDef mdef;
  mdef.CopyFrom(def);
  mdef.clear_op();

  std::unordered_map<std::string, std::set<size_t>> cpu_blobs, metal_blobs;
  cpu_blobs[def.external_input(0)].insert(0);

  for (auto i = 0; i < def.op_size(); i++) {
    const auto &currentOp = def.op(i);
    if (currentOp.engine() == engine) {
      // Metal Op
      // insert copyToMetalOp
      for (auto j = 0; j < currentOp.input_size(); j++) {
        auto &input = currentOp.input(j);
        auto version = analysis.ssa[i].inVersions[input];
        if (cpu_blobs[input].count(version) > 0) {
          insertCopyToMetalGPUOp(mdef, input);
          metal_blobs[input].insert(version);
          cpu_blobs[input].erase(version);
        }
      }

      auto *op = mdef.add_op();
      op->CopyFrom(currentOp);

      // swap input blob
      for (auto j = 0; j < currentOp.input_size(); j++) {
        auto &input = currentOp.input(j);
        auto version = analysis.ssa[i].inVersions[input];
        if (metal_blobs[input].count(version) > 0) {
          op->set_input(j, input + "_M");
        }
      }

      // swap output blob
      for (auto j = 0; j < currentOp.output_size(); j++) {
        auto &output = currentOp.output(j);
        auto version = analysis.ssa[i].outVersions[output];
        op->set_output(j, output + "_M");
        metal_blobs[output].insert(version);
      }
      // insert copyFromMetalOp after the last op if the last op is a metal op
      if (i == def.op_size() - 1) {
        insertCopyFromMetalGPUOp(mdef, currentOp.output(0));
      }
    } else {
      // CPU Op
      // insert copyFromMetalOp
      for (auto j = 0; j < currentOp.input_size(); j++) {
        auto &input = currentOp.input(j);
        auto version = analysis.ssa[i].inVersions[input];
        if (metal_blobs[input].count(version) > 0) {
          insertCopyFromMetalGPUOp(mdef, input);
        }
      }
      auto *op = mdef.add_op();
      op->CopyFrom(currentOp);
      for (auto j = 0; j < currentOp.output_size(); j++) {
        auto &output = currentOp.output(j);
        auto version = analysis.ssa[i].outVersions[output];
        cpu_blobs[output].insert(version);
      }
    }
  }
  return mdef;
}

bool tryFuseAdjacentOps(const OperatorDef &currentOp, const OperatorDef &nextOp, OperatorDef *fusedOp) {
  // Check for possible invalid opportunities.
  // Must be identical outputs, with in-place usage for nextOp.
  if (currentOp.output_size() != 1 || nextOp.output_size() != 1) {
    return false;
  }
  if (currentOp.output(0) != nextOp.input(0) || nextOp.input(0) != nextOp.output(0)) {
    return false;
  }

  static const std::map<std::pair<std::string, std::string>, std::string> fusionOpportunities = {{
      {{"MetalInstanceNorm", "MetalPRelu"}, "MetalInstanceNormPRelu"},
  }};
  auto it = fusionOpportunities.find({currentOp.type(), nextOp.type()});
  if (it == fusionOpportunities.end()) {
    return false;
  }
  LOG(INFO) << "Found a fusion between adjacent ops: (" << currentOp.type() << ", " << nextOp.type() << ") -> "
            << it->second;
  fusedOp->CopyFrom(currentOp);
  fusedOp->set_type(it->second);
  for (auto i = 1; i < nextOp.input_size(); i++) {
    fusedOp->add_input(nextOp.input(i));
  }
  return true;
}

NetDef runMetalFusion(const NetDef &def) {
  CHECK_GE(def.op_size(), 1);
  NetDef mdef;
  mdef.CopyFrom(def);
  mdef.clear_op();
  auto i = 0;

  while (i < def.op_size()) {
    if (i == def.op_size() - 1) {
      VLOG(2) << "Last operator, skipping";
      auto *op = mdef.add_op();
      op->CopyFrom(def.op(i));
      i += 1;
      continue;
    }

    const auto &currentOp = def.op(i);
    const auto &nextOp = def.op(i + 1);
    OperatorDef fusedOp;
    if (tryFuseAdjacentOps(currentOp, nextOp, &fusedOp)) {
      VLOG(2) << "Found an adjacent fusion at: " << i;
      // We can fuse.
      auto *op = mdef.add_op();
      op->CopyFrom(fusedOp);
      i += 2;
      continue;
    }
    VLOG(2) << "No fusion available";
    // Just emit the current type.
    auto *op = mdef.add_op();
    op->CopyFrom(currentOp);
    i += 1;
  }
  return mdef;
}

void dumpDef(NetDef &net) {
  for (const auto &op : net.op()) {
    printf("***Operator: %s\n", op.type().c_str());
    for (auto input : op.input()) {
      printf("\tInput: %s\n", input.c_str());
    }

    for (auto output : op.output()) {
      printf("\tOutput: %s\n", output.c_str());
    }
  }
}

NetDef rewritePredictNetForMetal(const NetDef &predictNet, const std::string engine) {
  CAFFE_ENFORCE_GE(predictNet.op_size(), 1);
  NetDef net;
  net.CopyFrom(predictNet);

  std::unordered_map<std::string, std::string> replacements({
      {"Conv", "MetalConv"},
      {"InstanceNorm", "MetalInstanceNorm"},
      {"PRelu", "MetalPRelu"},
      {"ConvTranspose", "MetalConvTranspose"},
  });

  for (OperatorDef &op : *net.mutable_op()) {
    if (op.has_type() && replacements.count(op.type()) > 0) {
      op.set_type(replacements[op.type()]);
      op.set_engine("METAL");
    }
  }

  net = insertInputOutputCopyOps(net, engine);
  net = runMetalFusion(net);
  return net;
}

NetDef rewriteInitNetForMetal(const NetDef &initNet, const NetDef &predictNet, const std::string engine) {
  // Find the GivenTensorFill operators for weight tensors and change to Metal GivenTensorFill ops
  NetDef net;
  net.CopyFrom(initNet);
  std::set<std::string> conv_weights, conv_tranpose_weights, weights_and_biases;
  for (auto &op : predictNet.op()) {
    if (op.engine() == engine) {
      if (op.type() == "MetalConv") {
        conv_weights.insert(op.input(1));
      } else if (op.type() == "MetalConvTranspose") {
        conv_tranpose_weights.insert(op.input(1));
      } else {
        // Need to add support for operators with > 1 input tensors such as Add
        for (int i = 1; i < op.input_size(); i++) {
          weights_and_biases.insert(op.input(i));
        }
      }
    }
  }

  for (auto &op : *net.mutable_op()) {
    if (op.type() == "GivenTensorFill" && op.output_size() == 1) {
      if (conv_weights.count(op.output(0)) > 0) {
        op.set_type("GivenWeightTensorFill");
        op.set_engine(engine);
      } else if (conv_tranpose_weights.count(op.output(0)) > 0) {
        op.set_type("GivenTransposeWeightTensorFill");
        op.set_engine(engine);
      } else if (weights_and_biases.count(op.output(0)) > 0) {
        op.set_type("GivenTensorFloat16MetalFill");
        op.set_engine(engine);
      }
    }
  }
  return net;
}

bool tryConvertToMetal(const NetDef &initNet, const NetDef &predictNet, NetDef *metalInitNet, NetDef *metalPredictNet) {
  try {
    // Throws if unsupported operators are found.
    *metalPredictNet = rewritePredictNetForMetal(predictNet, "METAL");
    *metalInitNet = rewriteInitNetForMetal(initNet, *metalPredictNet, "METAL");

    // Throws if unsupported parameters are found.
    Workspace ws;
    ws.RunNetOnce(*metalInitNet);
    ws.CreateNet(*metalPredictNet);
    LOG(INFO) << "Metal is successfully enabled";
    return true;
  } catch (const std::exception &e) {
    LOG(ERROR) << "Caught exception trying to convert NetDef to Metal: " << e.what();
    return false;
  }
}
} // namespace caffe2
