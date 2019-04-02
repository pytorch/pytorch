#include <algorithm>
#include <ctime>

#include "caffe2/core/operator.h"
#include "caffe2/core/net.h"
#include "caffe2/core/workspace.h"
#include "caffe2/proto/caffe2.pb.h"

namespace caffe2 {

Blob* Workspace::CreateBlob(const string& name) {
  if (HasBlob(name)) {
    VLOG(1) << "Blob " << name << " already exists. Skipping.";
  } else {
    VLOG(1) << "Creating blob " << name;
    (*blob_map_)[name] = unique_ptr<Blob>(new Blob());
  }
  return (*blob_map_)[name].get();
}

const Blob* Workspace::GetBlob(const string& name) const {
  if (!HasBlob(name)) {
    LOG(WARNING) << "Blob " << name << " not in the workspace.";
    // TODO(Yangqing): do we want to always print out the list of blobs here?
    LOG(WARNING) << "Current blobs:";
    for (const auto& entry : *blob_map_) {
      LOG(WARNING) << entry.first;
    }
    return nullptr;
  } else {
    return (*blob_map_)[name].get();
  }
}

bool Workspace::CreateNet(const NetDef& net_def) {
  CHECK(net_def.has_name()) << "Net definition should have a name.";
  if (net_map_.count(net_def.name()) > 0) {
    LOG(WARNING) << "Overwriting existing network of the same name.";
    // Note(Yangqing): Why do we explicitly erase it here? Some components of
    // the old network, such as a opened LevelDB, may prevent us from creating a
    // new network before the old one is deleted. Thus we will need to first
    // erase the old one before the new one can be constructed.
    net_map_.erase(net_def.name());
  }
  // Create a new net with its name.
  LOG(INFO) << "Initializing network " << net_def.name();
  net_map_[net_def.name()] =
      unique_ptr<NetBase>(caffe2::CreateNet(net_def, this));
  if (net_map_[net_def.name()].get() == nullptr) {
    LOG(ERROR) << "Error when creating the network.";
    net_map_.erase(net_def.name());
    return false;
  }
  if (!net_map_[net_def.name()]->Verify()) {
    LOG(ERROR) << "Error when setting up network " << net_def.name();
    return false;
  }
  return true;
}

void Workspace::DeleteNet(const string& name) {
  if (net_map_.count(name)) {
    net_map_.erase(name);
  }
}

bool Workspace::RunNet(const string& name) {
  if (!net_map_.count(name)) {
    LOG(ERROR) << "Network " << name << " does not exist yet.";
    return false;
  }
  return net_map_[name]->Run();
}

bool Workspace::RunOperatorOnce(const OperatorDef& op_def) {
  std::unique_ptr<OperatorBase> op(CreateOperator(op_def, this));
  if (!op->Verify()) {
    LOG(ERROR) << "Error when setting up operator " << op_def.name();
    return false;
  }
  if (!op->Run()) {
    LOG(ERROR) << "Error when running operator " << op_def.name();
    return false;
  }
  return true;
}
bool Workspace::RunNetOnce(const NetDef& net_def) {
  std::unique_ptr<NetBase> net(caffe2::CreateNet(net_def, this));
  if (!net->Verify()) {
    LOG(ERROR) << "Error when setting up network " << net_def.name();
    return false;
  }
  if (!net->Run()) {
    LOG(ERROR) << "Error when running network " << net_def.name();
    return false;
  }
  return true;
}

bool Workspace::RunPlan(const PlanDef& plan) {
  LOG(INFO) << "Started executing plan.";
  if (plan.networks_size() == 0 || plan.execution_steps_size() == 0) {
    LOG(WARNING) << "Nothing to run - did you define a correct plan?";
    // We will do nothing, but the plan is still legal so we will return true.
    return true;
  }
  LOG(INFO) << "Initializing networks.";

  for (const NetDef& net_def : plan.networks()) {
    if (!CreateNet(net_def)) {
      LOG(ERROR) << "Failed initializing the networks.";
      return false;
    }
  }
  clock_t start_time = clock();
  for (const ExecutionStep& step : plan.execution_steps()) {
    clock_t step_start_time = clock();
    if (!ExecuteStepRecursive(step)) {
      LOG(ERROR) << "Failed initializing step " << step.name();
      return false;
    }
    LOG(INFO) << "Step " << step.name() << " took "
              << static_cast<float>(clock() - step_start_time) / CLOCKS_PER_SEC
              << " seconds.";
  }
  LOG(INFO) << "Total plan took "
            << static_cast<float>(clock() - start_time) / CLOCKS_PER_SEC
            << " seconds.";
  LOG(INFO) << "Plan executed successfully.";
  return true;
}

bool Workspace::ExecuteStepRecursive(const ExecutionStep& step) {
  LOG(INFO) << "Running execution step " << step.name();
  if (!(step.substeps_size() == 0 || step.networks_size() == 0)) {
    LOG(ERROR) << "An ExecutionStep should either have substeps or networks "
               << "but not both.";
    return false;
  }

  if (step.substeps_size()) {
    int iterations = step.has_iterations() ? step.iterations() : 1;
    for (int i = 0; i < iterations; ++i) {
      for (const ExecutionStep& substep : step.substeps()) {
        if (!ExecuteStepRecursive(substep)) {
          return false;
        }
      }
    }
    return true;
  } else {
    // If this ExecutionStep just contains nets, we can directly run it.
    vector<NetBase*> networks;
    // Collect the networks to run.
    for (const string& network_name : step.networks()) {
      if (!net_map_.count(network_name)) {
        LOG(ERROR) << "Network " << network_name << " not found.";
        return false;
      }
      VLOG(1) << "Going to execute network " << network_name;
      networks.push_back(net_map_[network_name].get());
    }
    int iterations = step.has_iterations() ? step.iterations() : 1;
    VLOG(1) << "Executing networks for " << iterations << " iterations.";
    for (int iter = 0; iter < iterations; ++iter) {
      VLOG(1) << "Executing network iteration " << iter;
      for (NetBase* network : networks) {
        if (!network->Run()) {
          return false;
        }
      }
    }
  }
  return true;
}

}  // namespace caffe2
