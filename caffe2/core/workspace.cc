#include <algorithm>
#include <ctime>

#include "caffe2/core/operator.h"
#include "caffe2/core/net.h"
#include "caffe2/core/workspace.h"
#include "caffe2/proto/caffe2.pb.h"

namespace caffe2 {

Blob* Workspace::CreateBlob(const string& name) {
  if (HasBlob(name)) {
    CAFFE_VLOG(1) << "Blob " << name << " already exists. Skipping.";
  } else {
    CAFFE_VLOG(1) << "Creating blob " << name;
    (*blob_map_)[name] = unique_ptr<Blob>(new Blob());
  }
  return (*blob_map_)[name].get();
}

const Blob* Workspace::GetBlob(const string& name) const {
  if (!HasBlob(name)) {
    CAFFE_LOG_WARNING << "Blob " << name << " not in the workspace.";
    // TODO(Yangqing): do we want to always print out the list of blobs here?
    CAFFE_LOG_WARNING << "Current blobs:";
    for (const auto& entry : *blob_map_) {
      CAFFE_LOG_WARNING << entry.first;
    }
    return nullptr;
  } else {
    return (*blob_map_)[name].get();
  }
}

Blob* Workspace::GetBlob(const string& name) {
  return const_cast<Blob*>(
      static_cast<const Workspace*>(this)->GetBlob(name));
}

NetBase* Workspace::CreateNet(const NetDef& net_def) {
  CAFFE_CHECK(net_def.has_name()) << "Net definition should have a name.";
  if (net_map_.count(net_def.name()) > 0) {
    CAFFE_LOG_WARNING << "Overwriting existing network of the same name.";
    // Note(Yangqing): Why do we explicitly erase it here? Some components of
    // the old network, such as a opened LevelDB, may prevent us from creating a
    // new network before the old one is deleted. Thus we will need to first
    // erase the old one before the new one can be constructed.
    net_map_.erase(net_def.name());
  }
  // Create a new net with its name.
  CAFFE_LOG_INFO << "Initializing network " << net_def.name();
  net_map_[net_def.name()] =
      unique_ptr<NetBase>(caffe2::CreateNet(net_def, this));
  if (net_map_[net_def.name()].get() == nullptr) {
    CAFFE_LOG_ERROR << "Error when creating the network.";
    net_map_.erase(net_def.name());
    return nullptr;
  }
  if (!net_map_[net_def.name()]->Verify()) {
    CAFFE_LOG_ERROR << "Error when setting up network " << net_def.name();
    net_map_.erase(net_def.name());
    return nullptr;
  }
  return net_map_[net_def.name()].get();
}

void Workspace::DeleteNet(const string& name) {
  if (net_map_.count(name)) {
    net_map_.erase(name);
  }
}

bool Workspace::RunNet(const string& name) {
  if (!net_map_.count(name)) {
    CAFFE_LOG_ERROR << "Network " << name << " does not exist yet.";
    return false;
  }
  return net_map_[name]->Run();
}

bool Workspace::RunOperatorOnce(const OperatorDef& op_def) {
  std::unique_ptr<OperatorBase> op(CreateOperator(op_def, this));
  if (!op->Verify()) {
    CAFFE_LOG_ERROR << "Error when setting up operator " << op_def.name();
    return false;
  }
  if (!op->Run()) {
    CAFFE_LOG_ERROR << "Error when running operator " << op_def.name();
    return false;
  }
  return true;
}
bool Workspace::RunNetOnce(const NetDef& net_def) {
  std::unique_ptr<NetBase> net(caffe2::CreateNet(net_def, this));
  if (!net->Verify()) {
    CAFFE_LOG_ERROR << "Error when setting up network " << net_def.name();
    return false;
  }
  if (!net->Run()) {
    CAFFE_LOG_ERROR << "Error when running network " << net_def.name();
    return false;
  }
  return true;
}

bool Workspace::RunPlan(const PlanDef& plan) {
  CAFFE_LOG_INFO << "Started executing plan.";
  if (plan.network_size() == 0 || plan.execution_step_size() == 0) {
    CAFFE_LOG_WARNING << "Nothing to run - did you define a correct plan?";
    // We will do nothing, but the plan is still legal so we will return true.
    return true;
  }
  CAFFE_LOG_INFO << "Initializing networks.";

  for (const NetDef& net_def : plan.network()) {
    if (!CreateNet(net_def)) {
      CAFFE_LOG_ERROR << "Failed initializing the networks.";
      return false;
    }
  }
  clock_t start_time = clock();
  for (const ExecutionStep& step : plan.execution_step()) {
    clock_t step_start_time = clock();
    if (!ExecuteStepRecursive(step)) {
      CAFFE_LOG_ERROR << "Failed initializing step " << step.name();
      return false;
    }
    CAFFE_LOG_INFO << "Step " << step.name() << " took "
              << static_cast<float>(clock() - step_start_time) / CLOCKS_PER_SEC
              << " seconds.";
  }
  CAFFE_LOG_INFO << "Total plan took "
            << static_cast<float>(clock() - start_time) / CLOCKS_PER_SEC
            << " seconds.";
  CAFFE_LOG_INFO << "Plan executed successfully.";
  return true;
}

bool Workspace::ExecuteStepRecursive(const ExecutionStep& step) {
  CAFFE_LOG_INFO << "Running execution step " << step.name();
  if (!(step.substep_size() == 0 || step.network_size() == 0)) {
    CAFFE_LOG_ERROR << "An ExecutionStep should either have substep or networks "
               << "but not both.";
    return false;
  }

  int iterations = step.has_num_iter() ? step.num_iter() : 1;
  CAFFE_VLOG(1) << "Executing step for " << iterations << " iterations.";
  if (step.substep_size()) {
    for (int i = 0; i < iterations; ++i) {
      for (const ExecutionStep& substep : step.substep()) {
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
    for (const string& network_name : step.network()) {
      if (!net_map_.count(network_name)) {
        CAFFE_LOG_ERROR << "Network " << network_name << " not found.";
        return false;
      }
      CAFFE_VLOG(1) << "Going to execute network " << network_name;
      networks.push_back(net_map_[network_name].get());
    }
    for (int iter = 0; iter < iterations; ++iter) {
      CAFFE_VLOG(1) << "Executing network iteration " << iter;
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
