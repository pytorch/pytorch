#include "caffe2/operators/rnn/recurrent_network_executor.h"

#include "caffe2/core/timer.h"

namespace caffe2 {

/**
 * Implementation of RecurrentNetworkExecutor that uses thread pool for
 * multithreaded execution of RNNs. Used with CPU.
 */

template <>
std::unique_ptr<RecurrentNetworkExecutorBase> createRNNExecutor<CPUContext>(
    const NetDef& step_net_def,
    std::map<string, string>& recurrent_input_map,
    std::string timestep_blob,
    ArgumentHelper rnn_args) {
  auto* exec = new ThreadedRecurrentNetworkExecutor(
      step_net_def, recurrent_input_map, timestep_blob);
  int num_threads =
      rnn_args.GetSingleArgument<int>("rnn_executor.num_threads", 0);
  if (num_threads > 0) {
    exec->setNumThreads(num_threads);
    LOG(INFO) << "Set num threads: " << num_threads;
  }
  exec->debug_ = rnn_args.GetSingleArgument<int>("rnn_executor_debug", 0);
  return std::unique_ptr<RecurrentNetworkExecutorBase>(exec);
}

/**
 * Run forwardpass with T timesteps.
 */
bool ThreadedRecurrentNetworkExecutor::Run(int T) {
  CAFFE_ENFORCE_GE(T, 0, "Negative number of steps");
  if (T == 0) {
    return true;
  }

  CAFFE_ENFORCE(timestep_ops_.size() >= T);
  countdown_ = T * timestep_ops_[0].size();
  finished_timesteps_ = 0;

  CHECK(task_queue_.size() == 0);

  for (auto& rnn_op : timestep_ops_[0]) {
    // Launch "frontier"-ops first.
    if (rnn_op.frontier) {
      task_queue_.Push(OpTask(0, rnn_op.order, T, 1));
    }
  }

  _Exec();
  return true;
}

/**
 * Run backward pass with T timesteps.
 */
bool ThreadedRecurrentNetworkExecutor::RunBackwards(int T) {
  CAFFE_ENFORCE_GE(T, 0, "Negative number of steps");
  if (T == 0) {
    return true;
  }

  CAFFE_ENFORCE(timestep_ops_.size() >= T);
  countdown_ = T * timestep_ops_[0].size();
  finished_timesteps_ = 0;

  // Frontier
  CHECK(task_queue_.size() == 0);

  for (auto& rnn_op : timestep_ops_[T - 1]) {
    if (rnn_op.frontier) {
      task_queue_.Push(OpTask(T - 1, rnn_op.order, T, -1));
    }
  }

  _Exec();
  return true;
}

/**
 * Runs a single op and updates its dependencies when finished. If
 * dependent ops are ready to run, adds them to the task_queue.
 */
void ThreadedRecurrentNetworkExecutor::RunOp(OpTask job, int /*thread_id*/) {
  bool first_timestep =
      ((job.forward() && job.timestep == 0) ||
       (job.backward() && job.timestep == job.T - 1));
  bool last_timestep =
      ((job.backward() && job.timestep == 0) ||
       (job.forward() && job.timestep == job.T - 1));
  auto& rnn_op = timestep_ops_[job.timestep][job.op_idx];
  if (rnn_op.num_dynamic_inputs > 0 && !rnn_op.frontier) {
    CAFFE_ENFORCE_EQ(
        rnn_op.proc_inputs,
        rnn_op.num_dynamic_inputs -
            first_timestep * rnn_op.num_recurrent_inputs,
        "Error at operator ",
        job.op_idx,
        " on timestep ",
        job.timestep,
        " T=",
        job.T,
        " first =",
        first_timestep);
  }

  // Reset input dependency counter
  rnn_op.proc_inputs = 0;

  // Run the operator
  rnn_op.op->Run();

  // Knock down dependencies and start next ops, if this
  // was last dependency fulfilled.
  for (int depidx : rnn_op.dependencies) {
    int t = job.timestep;
    bool for_next_timestep = depidx <= rnn_op.order;
    if (!last_timestep && for_next_timestep) {
      t += job.direction;
    } else if (for_next_timestep) {
      continue;
    }

    auto& dep_op = timestep_ops_[t][depidx];
    int proc_inputs = dep_op.proc_inputs.fetch_add(1) + 1;

    // Schedule next op, if this was the last dependency. Note that on
    // first timestep we don't have recurrent inputs.
    int num_req_inputs = dep_op.num_dynamic_inputs;
    if (first_timestep && !for_next_timestep) {
      num_req_inputs -= dep_op.num_recurrent_inputs;
    }

    if (proc_inputs == num_req_inputs || num_req_inputs == 0) {
      task_queue_.Push(OpTask(t, depidx, job.T, job.direction));
    }
  }

  // Decrement countdown: when at zero, we have run all ops and can
  // notify the caller thread.
  if (countdown_.fetch_sub(1) == 1) {
    CAFFE_ENFORCE_EQ(0, task_queue_.size());
    std::unique_lock<std::mutex> lk(countdown_mtx_);
    cv_.notify_one();
  }
}

/**
 * Run-loop for executor threads: pop tasks from task_queue and execute
 * them with RunOp().
 */
void ThreadedRecurrentNetworkExecutor::WorkerFunction() {
  size_t num_jobs = 0;
  static std::atomic<int> seq(0);
  int id = seq.fetch_add(1);

  while (!failed_) {
    OpTask job;
    if (!task_queue_.Pop(&job)) {
      break;
    }

    // Check for limited timestep parallelism, and if too many timesteps would
    // be started concurrently, return the task to task queue.
    if (max_parallel_timesteps_ > 0) {
      int t = (job.direction == 1 ? job.timestep : job.T - job.timestep + 1);
      if (t - finished_timesteps_ >= max_parallel_timesteps_) {
        // Return to queue
        task_queue_.Push(job);
        continue;
      }
    }

    try {
      RunOp(job, id);
      if (job.op_idx == timestep_ops_template_.size() - 1) {
        finished_timesteps_.fetch_add(1);
      }
      num_jobs++;
    } catch (::caffe2::EnforceNotMet& enf) {
      std::unique_lock<std::mutex> lk(countdown_mtx_);
      LOG(ERROR) << "Crash at thread " << id << " timestep " << job.timestep
                 << " op:" << ProtoDebugString(step_net_def_.op(job.op_idx))
                 << enf.what();
      task_queue_.NoMoreJobs();
      failed_ = true;
      cv_.notify_one();
      return;
    }
  }
  VLOG(1) << "Worker exiting, did run: " << num_jobs << " jobs";
}

/**
 * Start worker threads if not started yet, wait until all tasks
 * finished, or a failure. Called by Run() and RunBackwards().
 */
void ThreadedRecurrentNetworkExecutor::_Exec() {
  CAFFE_ENFORCE_EQ(
      false, failed_, "Tried to execute a previously failed RNN executor");

  // Start threads if not started
  std::unique_lock<std::mutex> lk(countdown_mtx_);
  while (workers_.size() < num_threads_) {
    VLOG(1) << "Start RNN worker " << workers_.size() << " / " << num_threads_;
    workers_.push_back(
        std::thread(&ThreadedRecurrentNetworkExecutor::WorkerFunction, this));
  }

  // Wait until threads finish.
  Timer t;
  while (!failed_ && countdown_ > 0) {
    cv_.wait_for(lk, std::chrono::seconds(30), [&] {
      // Log if we are still running, so that we catch deadlocks.. there
      // should not be any deadlocks, but...
      if (t.Seconds() > 10) {
        LOG(INFO) << "RNN Executor still running, remaining ops: "
                  << countdown_;
      }
      return failed_ || countdown_ == 0;
    });
  }

  CAFFE_ENFORCE_EQ(
      false,
      failed_,
      "RNN executor encountered failure. See prior error logs for details.");
}

} // namespace caffe2
