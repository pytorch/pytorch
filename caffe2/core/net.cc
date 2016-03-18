#include <set>

#include "caffe2/core/net.h"
#include "caffe2/core/operator.h"
#include "caffe2/core/timer.h"
#include "caffe2/proto/caffe2.pb.h"


namespace caffe2 {

DEFINE_REGISTRY(NetRegistry, NetBase, const NetDef&, Workspace*);
REGISTER_NET(simple, SimpleNet);
REGISTER_NET(dag, DAGNet);

NetBase* CreateNet(const NetDef& net_def, Workspace* ws) {
  // In default, we will return a simple network that just runs all operators
  // sequentially.
  if (!net_def.has_net_type()) {
    return new SimpleNet(net_def, ws);
  }
  return NetRegistry()->Create(net_def.net_type(), net_def, ws);
}

SimpleNet::SimpleNet(const NetDef& net_def, Workspace* ws)
    : NetBase(net_def, ws) {
  bool net_def_has_device_option = net_def.has_device_option();
  // Initialize the operators
  for (const OperatorDef& operator_def : net_def.op()) {
    CAFFE_VLOG(1) << "Creating operator " << operator_def.name()
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
    if (op.get() == nullptr) {
      CAFFE_LOG_ERROR << "Found empty operator.";
      return false;
    }
    CAFFE_VLOG(1) << "Verifying operator " << op->def().name()
            << "(" << op->def().type() << ").";
    if (!op->Verify()) {
      CAFFE_LOG_ERROR << "Failed to verify operator.";
      return false;
    }
  }
  return true;
}

bool SimpleNet::Run() {
  CAFFE_VLOG(1) << "Running net.";
  for (auto& op : operators_) {
    CAFFE_VLOG(1) << "Running operator " << op->def().name()
            << "(" << op->def().type() << ").";
    if (!op->Run()) return false;
  }
  return true;
}

void SimpleNet::TEST_Benchmark(const int warmup_runs, const int main_runs,
                               const bool run_individual) {
  CAFFE_LOG_INFO << "Starting benchmark.";
  CAFFE_LOG_INFO << "Running warmup runs.";
  CAFFE_CHECK_GE(warmup_runs, 0);
  for (int i = 0; i < warmup_runs; ++i) {
    CAFFE_CHECK(Run());
  }

  CAFFE_LOG_INFO << "Main runs.";
  CAFFE_CHECK_GE(main_runs, 0);
  Timer timer;
  for (int i = 0; i < main_runs; ++i) {
    CAFFE_CHECK(Run());
  }
  CAFFE_LOG_INFO << "Main run finished. Milliseconds per iter: "
                 << timer.MilliSeconds() / main_runs;

  vector<float> time_per_op(operators_.size(), 0);
  CaffeMap<string, float> time_per_op_type;
  if (run_individual) {
    for (int i = 0; i < main_runs; ++i) {
      int idx = 0;
      for (auto& op : operators_) {
        const string& op_type = op->def().type();
        timer.Start();
        CAFFE_CHECK(op->Run());
        float spent = timer.MilliSeconds();
        time_per_op[idx] += spent;
        time_per_op_type[op_type] += spent;
        ++idx;
      }
    }

    int idx = 0;
    for (auto& op : operators_) {
      const string& op_type = op->def().type();
      CAFFE_LOG_INFO << "Operator #" << idx << " ("
                     << op->def().name() << ", " << op_type << ") "
                     << time_per_op[idx] / main_runs << " ms/iter";
      ++idx;
    }
    CAFFE_LOG_INFO << "Time per operator type:";
    for (const auto& item : time_per_op_type) {
      CAFFE_LOG_INFO << std::setw(15) << std::setfill(' ')
                     << item.second / main_runs
                     << " " << item.first;
    }
  }
}

DAGNet::DAGNet(const NetDef& net_def, Workspace* ws)
    : NetBase(net_def, ws), operator_nodes_(net_def.op_size()) {
  // Blob creator allows us to track which operator created which blob.
  std::map<string, int> blob_creator;
  std::map<string, std::set<int> > blob_readers;
  std::map<string, int> execution_chains;
  bool net_def_has_device_option = net_def.has_device_option();
  // Initialize the operators
  for (int idx = 0; idx < net_def.op_size(); ++idx) {
    const OperatorDef& op_def = net_def.op(idx);
    CAFFE_VLOG(1) << "Creating operator #" << idx << ": "
            << op_def.name() << ":" << op_def.type();
    if (!op_def.has_device_option() && net_def_has_device_option) {
      OperatorDef temp_def(op_def);
      temp_def.mutable_device_option()->CopyFrom(net_def.device_option());
      operator_nodes_[idx].operator_.reset(CreateOperator(temp_def, ws));
    } else {
      operator_nodes_[idx].operator_.reset(CreateOperator(op_def, ws));
    }
    // Check the inputs, and set up parents if necessary. This addressese the
    // read after write case.
    for (const string& input : op_def.input()) {
      if (blob_creator.count(input) == 0) {
        CAFFE_VLOG(1) << "Input " << input << " not produced by this net. "
                << "Assuming it is pre-existing.";
      } else {
        int parent = blob_creator[input];
        CAFFE_VLOG(1) << "op dependency (RaW " << input << "): "
                      << parent << "->" << idx;
        operator_nodes_[idx].parents_.push_back(parent);
        operator_nodes_[parent].children_.push_back(idx);
      }
      // Add the current idx to the readers of this input.
      blob_readers[input].insert(idx);
    }
    // Check the outputs.
    for (const string& output : op_def.output()) {
      if (blob_creator.count(output) != 0) {
        CAFFE_LOG_WARNING << "Output " << output << " produced again. "
                     << "Such operation is not strictly tested. "
                     << "Use at your own risk.";
        // This addresses the write after write case - we will assume that all
        // writes are inherently sequential.
        int waw_parent = blob_creator[output];
        CAFFE_VLOG(1) << "op dependency (WaW " << output << "): "
                      << waw_parent << "->" << idx;
        operator_nodes_[idx].parents_.push_back(waw_parent);
        operator_nodes_[waw_parent].children_.push_back(idx);
      }
      // This addresses the write after read case - we will assume that writes
      // should only occur after all previous reads are finished.
      for (const int war_parent : blob_readers[output]) {
        CAFFE_VLOG(1) << "op dependency (WaR " << output << "): "
                      << war_parent << "->" << idx;
        operator_nodes_[idx].parents_.push_back(war_parent);
        operator_nodes_[war_parent].children_.push_back(idx);
      }
      // Renew the creator of the output name.
      blob_creator[output] = idx;
      // The write would create an implicit barrier that all earlier readers of
      // this output is now parents of the current op, and future writes would
      // not need to depend on these earlier readers. Thus, we can clear up the
      // blob readers.
      blob_readers[output].clear();
    }

    // Explicitly specified dependency with execution_chain.
    for (const auto& arg : op_def.arg()) {
      if (arg.name() == "execution_chain") {
        for (const string& name : arg.strings()) {
          if (execution_chains.count(name) == 0) {
            // New execution chain. Do nothing but add it.
            execution_chains[name] = idx;
          } else {
            int parent = execution_chains[name];
            CAFFE_VLOG(1) << "op dependency due to execution chain " << name
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
  // Now, make sure that the parent list and the children list do not contain
  // duplicated items.
  for (int i = 0; i < operator_nodes_.size(); ++i) {
    auto& node = operator_nodes_[i];
    // Sort, remove duplicates, and delete self dependency.
    auto& p = node.parents_;
    std::sort(p.begin(), p.end());
    p.erase(std::unique(p.begin(), p.end()), p.end());
    p.erase(std::remove(p.begin(), p.end(), i), p.end());
    // Do the same for the children vector.
    auto& c = node.children_;
    std::sort(c.begin(), c.end());
    c.erase(std::unique(c.begin(), c.end()), c.end());
    c.erase(std::remove(c.begin(), c.end(), i), c.end());
  }

  // TODO: do we want to make sure that there are no loops in the dependency
  // graph?

  // Figure out the initial frontier - this is the one we will feed into the job
  // queue to start a run.
  for (int idx = 0; idx < operator_nodes_.size(); ++idx) {
    if (operator_nodes_[idx].parents_.size() == 0) {
      initial_frontier_.push_back(idx);
    }
  }
  // Finally, start the workers.
  int num_workers = net_def.has_num_workers() ? net_def.num_workers() : 1;
  CAFFE_CHECK_GT(num_workers, 0) << "Must have a nonnegative number of workers";
  if (num_workers == 1) {
    CAFFE_LOG_WARNING << "Number of workers is 1: this means that all operators "
                 << "will be executed sequentially. Did you forget to set "
                 << "num_workers in the NetDef?";
  }
  for (int i = 0; i < num_workers; ++i) {
    CAFFE_VLOG(1) << "Start worker #" << i;
    workers_.push_back(std::thread(&DAGNet::WorkerFunction, this));
  }
}

DAGNet::~DAGNet() {
  // Safely join all the workers before exiting.
  job_queue_.NoMoreJobs();
  CAFFE_VLOG(1) << "Joining workers.";
  for (auto& worker : workers_) {
    worker.join();
  }
}

bool DAGNet::Verify() {
  for (auto& op_node : operator_nodes_) {
    auto& op = op_node.operator_;
    CAFFE_VLOG(1) << "Verifying operator " << op->def().name()
            << "(" << op->def().type() << ").";
    if (op.get() == nullptr || !op->Verify()) {
      return false;
    }
  }
  return true;
}

bool DAGNet::Run() {
  // Lock the run_in_progress_ lock so that we do not accidentally call Run()
  // in parallel.
  std::unique_lock<std::mutex> run_lock(run_in_progress_);
  CAFFE_VLOG(1) << "Running parallel net.";
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
    CAFFE_VLOG(2) << "Remaining ops to run: " << remaining_ops_;
    cv_.wait(mutex_lock);
  }
  CAFFE_VLOG(2) << "All ops finished running.";
  // If the above while loop finished, we know that the current run finished.
  return success_;
}

void DAGNet::WorkerFunction() {
  // WorkerFunctions() is an infinite loop until there are no more jobs to run.
  while (true) {
    int idx;
    // If there is no more jobs - meaning that the DAGNet is destructing -
    // we will exit safely.
    if (!job_queue_.Pop(&idx)) {
      return;
    }
    CAFFE_VLOG(1) << "Running operator #" << idx << " "
            << operator_nodes_[idx].operator_->def().name()
            << "(" << operator_nodes_[idx].operator_->def().type() << ").";
    bool this_success = operator_nodes_[idx].operator_->Run();
    for (int child : operator_nodes_[idx].children_) {
      int count = --operator_nodes_[child].runtime_parent_count_;
      // The count should never be smaller than zero.
      CAFFE_DCHECK_GE(count, 0)
          << "Found runtime parent count smaller than zero for "
          << "operator node "
          << operator_nodes_[child].operator_->def().name()
          << "(" << operator_nodes_[child].operator_->def().type() << ").";
      if (count == 0) {
        CAFFE_VLOG(2) << "Pushing operator #" << child << " to queue.";
        job_queue_.Push(child);
      }
    }
    // Notify that the processed op is incremented by one.
    std::unique_lock<std::mutex> mutex_lock(remaining_ops_mutex_);
    --remaining_ops_;
    success_ &= this_success;
    CAFFE_DCHECK_GE(remaining_ops_, 0);
    cv_.notify_one();
    CAFFE_VLOG(2) << "Finished executing operator #" << idx;
  }
}

}  // namespace caffe2
