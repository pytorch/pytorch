#include "caffe2/core/net.h"
#include "caffe2/core/operator.h"
#include "caffe2/proto/caffe2.pb.h"

namespace caffe2 {

NetBase* CreateNet(const NetDef& net_def, Workspace* ws) {
  if (!net_def.has_net_type() || net_def.net_type() == "simple") {
    VLOG(1) << "Creating simple net.";
    return new SimpleNet(net_def, ws);
  } else if (net_def.net_type() == "parallel") {
    VLOG(1) << "Creating parallel net.";
    return new ParallelNet(net_def, ws);
  } else {
    LOG(ERROR) << "Unknown net type: " << net_def.net_type();
    return nullptr;
  }
  // Just to suppress compiler warning
  return nullptr;
}

SimpleNet::SimpleNet(const NetDef& net_def, Workspace* ws)
    : NetBase(net_def, ws) {
  bool net_def_has_device_option = net_def.has_device_option();
  // Initialize the operators
  for (const OperatorDef& operator_def : net_def.op()) {
    VLOG(1) << "Creating operator " << operator_def.name()
            << ":" << operator_def.type();
    if (!operator_def.has_device_option() && net_def_has_device_option) {
      // In the case that the operator def does not specify a device option but
      // the net def has a default option, we copy the device option over to the
      // operator def.
      OperatorDef temp_def(operator_def);
      temp_def.mutable_device_option()->CopyFrom(net_def.device_option());
      operators_.emplace_back(CreateOperator(temp_def, ws));
    } else {
      operators_.emplace_back(CreateOperator(operator_def, ws));
    }
  }
}

bool SimpleNet::Verify() {
  for (auto& op : operators_) {
    VLOG(1) << "Verifying operator " << op->def().name()
            << "(" << op->def().type() << ").";
    if (op.get() == nullptr || !op->Verify()) {
      return false;
    }
  }
  return true;
}

bool SimpleNet::Run() {
  VLOG(1) << "Running net.";
  for (const auto& op : operators_) {
    VLOG(1) << "Running operator " << op->def().name()
            << "(" << op->def().type() << ").";
    // TODO(Yangqing): convert this sequential run to event-based.
    if (!op->Run()) return false;
  }
  return true;
}

ParallelNet::ParallelNet(const NetDef& net_def, Workspace* ws)
    : NetBase(net_def, ws), operator_nodes_(net_def.op_size()) {
  // Blob creator allows us to track which operator created which blob.
  std::map<string, int> blob_creator;
  std::map<string, int> execution_chains;
  bool net_def_has_device_option = net_def.has_device_option();
  // Initialize the operators
  for (int idx = 0; idx < net_def.op_size(); ++idx) {
    const OperatorDef& op_def = net_def.op(idx);
    VLOG(1) << "Creating operator #" << idx << ": "
            << op_def.name() << ":" << op_def.type();
    if (!op_def.has_device_option() && net_def_has_device_option) {
      OperatorDef temp_def(op_def);
      temp_def.mutable_device_option()->CopyFrom(net_def.device_option());
      operator_nodes_[idx].operator_.reset(CreateOperator(temp_def, ws));
    } else {
      operator_nodes_[idx].operator_.reset(CreateOperator(op_def, ws));
    }
    // Check the inputs, and set up parents if necessary.
    for (const string& input : op_def.input()) {
      if (blob_creator.count(input) == 0) {
        VLOG(1) << "Input " << input << " not produced by this net. "
                << "Assuming it is pre-existing.";
      } else {
        int parent = blob_creator[input];
        VLOG(1) << "op dependency: " << parent << "->" << idx;
        operator_nodes_[idx].parents_.push_back(parent);
        operator_nodes_[parent].children_.push_back(idx);
      }
    }
    for (const string& output : op_def.output()) {
      if (blob_creator.count(output) != 0) {
        LOG(WARNING) << "Output " << output << " produced again. "
                     << "Such operation is not strictly tested. "
                     << "Use at your own risk.";
      }
      blob_creator[output] = idx;
    }

    for (const auto& arg : op_def.arg()) {
      if (arg.name() == "execution_chain") {
        for (const string& name : arg.strings()) {
          if (execution_chains.count(name) == 0) {
            // New execution chain. Do nothing but add it.
            execution_chains[name] = idx;
          } else {
            int parent = execution_chains[name];
            VLOG(1) << "op dependency due to execution chain " << name
                    << ": " << parent << "->" << idx;
            operator_nodes_[idx].parents_.push_back(parent);
            operator_nodes_[parent].children_.push_back(idx);
            // update the tail of the current execution chain.
            execution_chains[name] = idx;
          }
        }
      }
    }
  }
  // Figure out the initial frontier - this is the one we will feed into the job
  // queue to start a run.
  for (int idx = 0; idx < operator_nodes_.size(); ++idx) {
    if (operator_nodes_[idx].parents_.size() == 0) {
      initial_frontier_.push_back(idx);
    }
  }
  // Finally, start the workers.
  int num_workers = net_def.has_num_workers() ? net_def.num_workers() : 1;
  CHECK_GT(num_workers, 0) << "Must have a nonnegative number of workers";
  if (num_workers == 1) {
    LOG(WARNING) << "Number of workers is 1: this means that all operators "
                 << "will be executed sequentially. Did you forget to set "
                 << "num_workers in the NetDef?";
  }
  for (int i = 0; i < num_workers; ++i) {
    VLOG(1) << "Start worker #" << i;
    workers_.push_back(std::thread(&ParallelNet::WorkerFunction, this));
  }
}

ParallelNet::~ParallelNet() {
  // Safely join all the workers before exiting.
  job_queue_.NoMoreJobs();
  VLOG(1) << "Joining workers.";
  for (auto& worker : workers_) {
    worker.join();
  }
}

bool ParallelNet::Verify() {
  for (auto& op_node : operator_nodes_) {
    auto& op = op_node.operator_;
    VLOG(1) << "Verifying operator " << op->def().name()
            << "(" << op->def().type() << ").";
    if (op.get() == nullptr || !op->Verify()) {
      return false;
    }
  }
  return true;
}

bool ParallelNet::Run() {
  VLOG(1) << "Running parallel net.";
  // First, set up job queue.
  remaining_ops_ = operator_nodes_.size();
  success_ = true;
  // TODO(jiayq): Start all worker threads.
  // Initialize the runtime parent count.
  for (auto& node : operator_nodes_) {
    node.runtime_parent_count_ = node.parents_.size();
  }
  // Kickstart the job queue.
  for (auto& value : initial_frontier_) {
    job_queue_.Push(value);
  }
  std::unique_lock<std::mutex> mutex_lock(remaining_ops_mutex_);
  while (remaining_ops_ > 0) {
    VLOG(2) << "Remaining ops to run: " << remaining_ops_;
    cv_.wait(mutex_lock);
  }
  VLOG(2) << "All ops finished running.";
  // If the above while loop finished, we know that the current run finished.
  return success_;
}

void ParallelNet::WorkerFunction() {
  // WorkerFunctions() is an infinite loop until there are no more jobs to run.
  while (true) {
    int idx;
    // If there is no more jobs - meaning that the ParallelNet is destructing -
    // we will exit safely.
    if (!job_queue_.Pop(&idx)) {
      return;
    }
    VLOG(1) << "Running operator #" << idx << " "
            << operator_nodes_[idx].operator_->def().name()
            << "(" << operator_nodes_[idx].operator_->def().type() << ").";
    bool this_success = operator_nodes_[idx].operator_->Run();
    for (int child : operator_nodes_[idx].children_) {
      int count = --operator_nodes_[child].runtime_parent_count_;
      // The count should never be smaller than zero.
      DCHECK_GE(count, 0)
          << "Found runtime parent count smaller than zero for "
          << "operator node "
          << operator_nodes_[child].operator_->def().name()
          << "(" << operator_nodes_[child].operator_->def().type() << ").";
      if (count == 0) {
        VLOG(2) << "Pushing operator #" << child << " to queue.";
        job_queue_.Push(child);
      }
    }
    // Notify that the processed op is incremented by one.
    std::unique_lock<std::mutex> mutex_lock(remaining_ops_mutex_);
    --remaining_ops_;
    success_ &= this_success;
    DCHECK_GE(remaining_ops_, 0);
    cv_.notify_one();
    VLOG(2) << "Finished executing operator #" << idx;
  }
}

}  // namespace caffe2
