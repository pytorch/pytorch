#ifndef CAFFE2_OPERATORS_RECURRENT_NETWORK_EXECUTOR_H_
#define CAFFE2_OPERATORS_RECURRENT_NETWORK_EXECUTOR_H_

#include <map>
#include <vector>

#include "caffe2/core/context.h"
#include "caffe2/core/logging.h"
#include "caffe2/core/operator.h"
#include "caffe2/core/timer.h"
#include "caffe2/operators/recurrent_network_executor_incl.h"

namespace caffe2 {

class RecurrentNetworkExecutorBase {
 protected:
  explicit RecurrentNetworkExecutorBase(
      const NetDef& step_net_def,
      std::map<string, string>& recurrent_input_map,
      std::string timestep_blob)
      : step_net_def_(step_net_def),
        recurrent_input_map_(recurrent_input_map),
        timestep_blob_(timestep_blob) {
    for (int i = 0; i < step_net_def_.op_size(); i++) {
      op_deps_.push_back(op_deps(i));
    }
  }

 public:
  virtual ~RecurrentNetworkExecutorBase() {}

  virtual bool Run(int T) = 0;

  virtual bool RunBackwards(int T) = 0;

  /**
   * Callers must call this before running an execution that contains
   * timestep t. On first call, this will initialize the data structures
   * for the given timestep. For subsequent calls, this has no cost.
   */
  void EnsureTimestepInitialized(int t, Workspace* ws) {
    if (timestep_ops_template_.size() == 0) {
      CalculateInternalDependencies();
    }
    if (timestep_ops_.size() <= t ||
        (timestep_ops_.size() > t && timestep_ops_[t].size() == 0)) {
      for (int j = timestep_ops_.size(); j < t + 1; j++) {
        timestep_ops_.push_back(std::vector<RNNNetOperator>());
      }

      // Create a specific timestep blob for this timestep. This is to
      // avoid conflicting timestep blobs when reusing workspaces, as with
      // the forward-only mode.
      std::string this_timestep_blob =
        timestep_blob_ + "_rnnexec_t" + caffe2::to_string(t);
      ws->CreateBlob(this_timestep_blob)->template GetMutable<TensorCPU>()->Resize(1);
      auto b = ws->GetBlob(this_timestep_blob);
      CAFFE_ENFORCE(b);
      b->template GetMutable<TensorCPU>()
          ->template mutable_data<int32_t>()[0] = t;

      // Copy the operators from template
      for (auto& template_rnn_op : timestep_ops_template_) {
        auto& rnn_op = template_rnn_op;
        OperatorDef op_copy = step_net_def_.op(rnn_op.order);

        // Rename timestep references to use the timestep specific timestep blob
        for (int i = 0; i < op_copy.input_size(); i++) {
          if (op_copy.input(i) == timestep_blob_) {
            op_copy.set_input(i, this_timestep_blob);
          }
        }
        CAFFE_ENFORCE(!HasOutput(op_copy, timestep_blob_),
          "Timestep cannot be output of an op: ", timestep_blob_,
          " op=" + ProtoDebugString(op_copy));

        rnn_op.op = CreateOperator(op_copy, ws);
        timestep_ops_[t].emplace_back(rnn_op);
      }
    }
  }

  void SetMaxParallelTimesteps(int p) {
    max_parallel_timesteps_ = p;
  }

 private:
  // Utility method to check if any of the op inputs or control inputs
  // contain given blob 'input'
  bool has_input(std::string x, int opidx) {
    for (auto& inp : step_net_def_.op(opidx).input()) {
      if (inp == x) {
        return true;
      }
    }
    for (auto& inp : step_net_def_.op(opidx).control_input()) {
      if (inp == x) {
        return true;
      }
    }
    return false;
  }

  // Return all outbound dependencies of an op. Special case for
  // rnn dependencies, that are set in recurent_network_op.
  std::vector<string> op_deps(int i) {
    std::vector<string> outs;
    auto& opdef = step_net_def_.op(i);
    for (string o : opdef.output()) {
      outs.push_back(o);
    };
    for (auto& arg : opdef.arg()) {
      if (arg.name().find("rnn_dependency") == 0) {
        outs.push_back(arg.s());
      }
    }
    return outs;
  }

  /**
   * Calculate dependencies of this op, for the ops following it in this
   * timestep and also for the next timestep. Removes redundant dependencies.
   */
  void infer_dependencies(
      int start_i,
      std::set<string> outputs,
      std::vector<RNNNetOperator>& rnn_ops,
      std::set<int>* dep_ops) {
    std::set<string> frontier = outputs;
    std::set<int> already_accounted_deps;
    int num_ops = step_net_def_.op_size();
    for (int j = 0; j < num_ops - 1 && !outputs.empty(); j++) {
      int i = (start_i + j) % num_ops;
      if (rnn_ops[i].link_op && this->ignoreLinkDependencies()) {
        continue;
      }
      for (auto& outp : frontier) {
        if (has_input(outp, i)) {
          if (outputs.find(outp) != outputs.end()) {
            if (already_accounted_deps.find(i) ==
                already_accounted_deps.end()) {
              dep_ops->insert(i);
            }

            // Now we can take the deps of this ops and not
            // add them anymore
            for (int odep : rnn_ops[i].dependencies) {
              already_accounted_deps.insert(odep);
            }
          }
          for (string& dep_out : op_deps_[i]) {
            auto oit = outputs.find(dep_out);
            if (oit != outputs.end()) {
              // This op produces output of the orignal op, so the dependency
              // passed through that op
              outputs.erase(oit);
            }
            frontier.insert(dep_out);
          }
        }
      }
    }
  }

  /**
   * Add dependencies to ops in the next timestep that would write an op
   * that this op has as an input or output. This is special for RNNs,
   * since we can have ops running in different timesteps concurrently.
   */
  void add_race_conflict_dependencies(
      int opidx,
      std::vector<RNNNetOperator>& rnn_ops,
      std::set<int>* dep_ops) {
    for (int i = 0; i < opidx; i++) {
      if (rnn_ops[i].link_op && this->ignoreLinkDependencies()) {
        continue;
      }
      for (auto& dep_blob : op_deps_[i]) {
        for (auto& inp : step_net_def_.op(opidx).input()) {
          if (inp == dep_blob) {
            dep_ops->insert(i);
            break;
          }
        }
        for (auto& outp : step_net_def_.op(opidx).output()) {
          if (outp == dep_blob) {
            dep_ops->insert(i);
            break;
          }
        }
      }
    }
  }



  void CalculateInternalDependencies() {
    /**
     * Calculate the dependencies between ops inside timestep and across
     * timestep. These are store in timestep_ops_ vector that is copied
     * for each timestep.
     */
    for (int i = 0; i < step_net_def_.op_size(); i++) {
      timestep_ops_template_.push_back(RNNNetOperator(step_net_def_.op(i), i));
    }

    // Then see which outputs appear as inputs, and those are
    // the internal blobs.
    for (auto& rnn_op : timestep_ops_template_) {
      set<string> dep_outputs;
      for (auto& outp : op_deps_[rnn_op.order]) {
        dep_outputs.insert(outp);
      }

      // Add recurrent dependencies as 'outputs' for this op
      for (auto& outp : dep_outputs) {
        auto rit = recurrent_input_map_.find(outp);
        if (rit != recurrent_input_map_.end()) {
          dep_outputs.insert(rit->second);
        } else {
          dep_outputs.insert(outp);
        }
      }

      // Compute dependencies of this op.
      if (!rnn_op.link_op || !this->ignoreLinkDependencies()) {
        std::set<int> dependent_ops;
        infer_dependencies(
            rnn_op.order + 1,
            dep_outputs,
            timestep_ops_template_,
            &dependent_ops);

        if (!this->ignoreLinkDependencies()) {
          // Race conditions arise when link ops can be run out-of-order.
          add_race_conflict_dependencies(
              rnn_op.order, timestep_ops_template_, &dependent_ops);
        }
        for (int i : dependent_ops) {
          rnn_op.dependencies.push_back(i);
        }

        // Sort in ascending order of dependency distance. If op
        // j > i, then distance is j - i. But if j < i, then distance
        // from i to j passes the timestep boundary and is j + num ops - i.
        std::sort(
            rnn_op.dependencies.begin(),
            rnn_op.dependencies.end(),
            [&](const int& a, const int& b) {
              if (a < rnn_op.order && b < rnn_op.order) {
                return a < b;
              }
              if (a >= rnn_op.order && b >= rnn_op.order) {
                return a < b;
              }
              if (a >= rnn_op.order && b < rnn_op.order) {
                return true;
              }
              return false;
            });
      }
    }

    // Update dependency counts
    for (auto& rnn_op : timestep_ops_template_) {
      for (int i : rnn_op.dependencies) {
        timestep_ops_template_[i].num_dynamic_inputs++;

        if (i > rnn_op.order) {
          timestep_ops_template_[i].frontier = false;
        } else {
          timestep_ops_template_[i].num_recurrent_inputs++;
        }
      }
    }

    // Find ops that have no recurrent inputs, and bind them
    // to the last op of the timestep. If there is only one op
    // in the step net, then it will depend on itself. Note that
    // we do not increase the dynamic input counter.
    for (auto& rnn_op : timestep_ops_template_) {
      if (rnn_op.num_dynamic_inputs == 0 && rnn_op.num_recurrent_inputs == 0) {
        if (rnn_op.link_op && this->ignoreLinkDependencies()) {
          continue;
        }
        timestep_ops_template_.back().dependencies.push_back(rnn_op.order);
      }
    }

    // compute parents
    for (auto& rnn_op : timestep_ops_template_) {
      for (int dep : rnn_op.dependencies) {
        timestep_ops_template_[dep].parents.push_back(rnn_op.order);
      }
    }

    AnalyzeOps();
  }


protected:
  /**
   * For debug purposes
   */
  void PrintInfo(int t) {
    auto& rnn_ops = timestep_ops_[t];

    LOG(INFO) << "Timestep: " << t;
    for (auto& rnn_op : rnn_ops) {
      auto& op = rnn_op.op;
      LOG(INFO) << "Operator " << rnn_op.order << ": " << op->type()
                << " dep inputs:" << rnn_op.num_dynamic_inputs
                << " rec inputs:" << rnn_op.num_recurrent_inputs
                << " frontier: " << rnn_op.frontier;
      for (auto& inp : rnn_op.op->debug_def().input()) {
        LOG(INFO) << " ---- input: " << inp;
      }
      for (auto& outp : rnn_op.op->debug_def().output()) {
        LOG(INFO) << " ---- output: " << outp;
      }
      for (auto j : rnn_op.dependencies) {
        LOG(INFO) << " dep: " << j << ": " << rnn_ops[j].op->type();
      }
      for (auto j : rnn_op.parents) {
        LOG(INFO) << " parent: " << j << ": " << rnn_ops[j].op->type();
      }
    }

    LOG(INFO) << "recurrent_inputs:" << recurrent_input_map_;

    for (auto& rnn_op : rnn_ops) {
      LOG(INFO) << "Operator " << rnn_op.order;
      LOG(INFO) << ProtoDebugString(rnn_op.op->debug_def());
    }
  }

 protected:

  virtual void AnalyzeOps() {}

  virtual bool ignoreLinkDependencies() = 0;

  std::vector<std::vector<RNNNetOperator>> timestep_ops_;
  std::vector<OperatorBase*> op_ptrs_;

  std::vector<RNNNetOperator> timestep_ops_template_;

  NetDef step_net_def_;
  std::vector<std::vector<string>> op_deps_;
  std::map<string, string> recurrent_input_map_;
  std::string timestep_blob_;

  int num_threads_ = 4; // TODO: make configurable
  int max_parallel_timesteps_ = -1;
};

template <class Context>
std::unique_ptr<RecurrentNetworkExecutorBase> createRNNExecutor(
    const NetDef& step_net_def,
    std::map<string, string>& recurrent_input_map,
    std::string timestep_blob);

class ThreadedRecurrentNetworkExecutor : public RecurrentNetworkExecutorBase {
 public:
  ThreadedRecurrentNetworkExecutor(
      const NetDef& step_net_def,
      std::map<string, string>& recurrent_input_map,
      std::string timestep_blob)
      : RecurrentNetworkExecutorBase(step_net_def, recurrent_input_map, timestep_blob),
        failed_(false) {}

  ~ThreadedRecurrentNetworkExecutor() {
    job_queue_.NoMoreJobs();
    VLOG(1) << "Joining workers.";
    for (auto& worker : workers_) {
      worker.join();
    }
  }

  bool Run(int T) override;

  bool RunBackwards(int T) override;

  bool ignoreLinkDependencies() override {
    return false;
  }

 private:
  void _ExecRange(int from, int to);

  void _Exec();

  void WorkerFunction();

  void RunOp(OpJob job, int thread_id);

  SimpleQueue<OpJob> job_queue_;
  std::atomic<int> countdown_;
  std::atomic<bool> failed_;
  std::atomic<int> finished_timesteps_;
  int num_ops_;
  std::mutex countdown_mtx_;
  std::condition_variable cv_;
  std::vector<std::thread> workers_;
};

} // namespace caffe2

#endif // CAFFE2_OPERATORS_RECURRENT_NETWORK_EXECUTOR_H_
