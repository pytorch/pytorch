#include "caffe2/core/jit/net_jit.h"

#include "c10/util/Logging.h"
#include "caffe2/core/jit/net_jit_constants.h"
#include "caffe2/core/operator.h"
#include "caffe2/utils/proto_utils.h"

namespace caffe2 {

JITC2Program::JITC2Program(
    const std::shared_ptr<const NetDef>& net_def,
    Workspace* ws) {
  initGraph(net_def, ws);

  // Generate JIT program from a graph of chains
  //
  // The main task, launching async graph tasks
  traverseTasks([this](int task_id) {
    std::vector<int> parent_future_ids;
    parent_future_ids.reserve(parents(task_id).size());
    for (auto parent_task_id : parents(task_id)) {
      parent_future_ids.push_back(taskToId(parent_task_id));
    }
    // Using task_id as an address, will patch later
    emit(JITOp::ForkOp(taskToId(task_id), task_id, parent_future_ids));
  });

  // Waiting on the final tasks
  std::vector<int> final_future_ids;
  for (auto task_id = 0; task_id < numTasks(); ++task_id) {
    if (children(task_id).empty()) {
      final_future_ids.push_back(taskToId(task_id));
    }
  }
  emit(JITOp::JoinOp(final_future_ids));

  // Return from the main task
  emit(JITOp::ReturnOp());

  // Output chain tasks
  std::unordered_map<int, size_t> task_to_address;
  for (auto task_id = 0; task_id < numTasks(); ++task_id) {
    task_to_address[task_id] = nextAddress();
    auto num_parents = parents(task_id).size();
    if (num_parents > 0) {
      // joining on all input futures (all formal args)
      emit(JITOp::JoinOp(sequence(num_parents)));
    }
    for (auto op_id : taskOps(task_id)) {
      emit(JITOp::C2Op(op_id));
    }
    emit(JITOp::ReturnOp());
  }

  // Patching task addresses
  for (auto op_id = 0; op_id < ops_.size(); ++op_id) {
    auto& op = ops_[op_id];
    if (op.GetOpCode() == JITOpCode::FORK) {
      auto task_id = op.GetTaskAddress();
      op.SetTaskAddress(task_to_address[task_id]);
    }
  }
}

const std::vector<JITOp>& JITC2Program::GetOps() const {
  return ops_;
}

const std::vector<OperatorBase*>& JITC2Program::GetC2Ops() const {
  return c2_operators_;
}

////

size_t JITC2Program::numTasks() const {
  return chain_nodes_.size();
}

const std::vector<int>& JITC2Program::parents(size_t task_id) const {
  const auto& chain_node = chain_nodes_[task_id];
  return chain_node.parents_;
}

const std::vector<int>& JITC2Program::children(size_t task_id) const {
  const auto& chain_node = chain_nodes_[task_id];
  return chain_node.children_;
}

const std::vector<int>& JITC2Program::taskOps(size_t task_id) const {
  return chains_[task_id];
}

/* static */
std::vector<int> JITC2Program::sequence(size_t length) {
  std::vector<int> seq;
  seq.reserve(length);
  for (auto seq_idx = 0; seq_idx < length; ++seq_idx) {
    seq.push_back(seq_idx);
  }
  return seq;
}

void JITC2Program::traverseTasks(std::function<void(int)> visitor) {
  std::vector<bool> visited(numTasks(), false);
  std::queue<int> tasks;
  for (auto task_id = 0; task_id < numTasks(); ++task_id) {
    if (parents(task_id).empty()) {
      tasks.push(task_id);
    }
  }
  while (!tasks.empty()) {
    auto task_id = tasks.front();
    tasks.pop();

    visitor(task_id);
    visited[task_id] = true;

    for (auto child_id : children(task_id)) {
      if (!visited[child_id]) {
        bool all_parents_visited = true;
        for (auto parent_id : parents(child_id)) {
          if (!visited[parent_id]) {
            all_parents_visited = false;
            break;
          }
        }
        if (all_parents_visited) {
          tasks.push(child_id);
        }
      }
    }
  }
}

void JITC2Program::emit(const JITOp& op) {
  ops_.push_back(op);
}

size_t JITC2Program::nextAddress() const {
  return ops_.size();
}

void JITC2Program::initGraph(
    const std::shared_ptr<const NetDef>& net_def,
    Workspace* ws) {
  operator_nodes_ = dag_utils::prepareOperatorNodes(net_def, ws);
  c2_operators_.reserve(operator_nodes_.size());
  for (const auto& node : operator_nodes_) {
    c2_operators_.push_back(node.operator_.get());
  }
  auto execution_chains = dag_utils::computeChains(operator_nodes_);
  chains_.reserve(execution_chains.size());
  for (const auto& kv : execution_chains) {
    chains_.push_back(kv.second);
  }
  chain_nodes_ = dag_utils::prepareChainGraphNodes(operator_nodes_, chains_);

  // Disable events within ops, except for async CPU ops
  // that use them to signal their completion
  for (const auto& chain : chains_) {
    for (const auto& op_id : chain) {
      const auto& op = c2_operators_[op_id];
      if (!IsCPUDeviceType(op->device_option().device_type()) ||
          !op->HasAsyncPart()) {
        op->DisableEvent();
      }
    }
  }
}

size_t JITC2Program::taskToId(size_t task_id) const {
  return jit::MAX_FUTURE_INPUTS + task_id;
}

}; // namespace caffe2
