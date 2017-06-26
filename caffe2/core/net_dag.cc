#include "caffe2/core/net.h"

#include <set>
#include <stack>
#include <unordered_map>
#include <unordered_set>

#include "caffe2/core/operator.h"
#include "caffe2/core/static_tracepoint.h"
#include "caffe2/core/timer.h"
#include "caffe2/proto/caffe2.pb.h"
#include "caffe2/utils/proto_utils.h"

CAFFE2_DEFINE_bool(
    caffe2_disable_chaining,
    false,
    "Disable chaining logic (some latent multi-device issues).");

namespace caffe2 {

namespace {

bool sameDevice(const OperatorDef& lhs, const OperatorDef& rhs) {
  return lhs.device_option().device_type() ==
      rhs.device_option().device_type() &&
      lhs.device_option().cuda_gpu_id() == rhs.device_option().cuda_gpu_id();
}

using OpIndex = int;
DAGNetBase::ExecutionChains singleChains(
    const std::vector<internal::OperatorNode>& nodes) {
  DAGNetBase::ExecutionChains chains;
  for (auto i = 0; i < nodes.size(); ++i) {
    chains[i] = {i};
  }
  return chains;
}

static void prune(int node_idx, std::vector<internal::OpGraphNode>& nodes) {
  // Ancestor table for tracking the visited nodes
  std::vector<bool> ancestors(nodes.size(), false);
  // stack element is pair of <curr_node, previous_node>
  std::stack<std::pair<int, int>> nodes_stack;
  // initialize the prev_node to be -1
  nodes_stack.push(std::make_pair(node_idx, -1));

  while (!nodes_stack.empty()) {
    const auto& node_pair = nodes_stack.top();
    int curr = node_pair.first;
    int prev = node_pair.second;

    // If the node has already been visited, pop curr out of
    // stack and clean up the ancestor table
    CAFFE_ENFORCE(curr < ancestors.size(), "Out of bound access");
    if (ancestors[curr]) {
      ancestors[curr] = false;
      nodes_stack.pop();
      continue;
    }

    // Check if this has a parent that can be pruned:
    //  if parent is not the previous node visited and is
    //  an ancestor of the current traversar, it can be
    //  pruned.
    if (prev >= 0) {
      std::vector<int> new_parents;
      for (auto parent : nodes[curr].parents_) {
        if (parent != prev && ancestors[parent]) {
          // We can prune this one
          nodes[parent].children_.erase(
              std::remove(
                  nodes[parent].children_.begin(),
                  nodes[parent].children_.end(),
                  curr),
              nodes[parent].children_.end());
        } else {
          new_parents.push_back(parent);
        }
      }
      nodes[curr].parents_ = new_parents;
    }

    ancestors[curr] = true;

    // Descend -- but only once from each node
    if (nodes[curr].visited_inputs == nodes[curr].num_orig_parents) {
      const auto& children = nodes[curr].children_;
      for (auto child : children) {
        nodes[child].visited_inputs++;
        nodes_stack.push(std::make_pair(child, curr));
      }
    }
  }
}

/**
  * Prune redundant dependencies to improve chaining.
  * TODO: t15868555 This algorithm is fast but can miss dependencies.
  */
std::vector<internal::OpGraphNode> pruneOpNodeGraph(
    const std::vector<internal::OperatorNode>& orig_nodes) {
  Timer t;
  std::vector<internal::OpGraphNode> pruned;

  // Create a separate list of pruned operatornodes used
  // for the chaining computation. Because of the unique_ptr
  // in the OperatorNode, we cannot do a copy but have to
  // copy just the fields we need.
  for (auto& node : orig_nodes) {
    internal::OpGraphNode nd;
    nd.children_ = node.children_;
    nd.parents_ = node.parents_;
    nd.num_orig_parents = nd.parents_.size();
    pruned.push_back(nd);
  }

  for (int i = 0; i < pruned.size(); ++i) {
    if (pruned[i].parents_.size() == 0) {
      prune(i, pruned);
    }
  }

  LOG(INFO) << "Operator graph pruning prior to chain compute took: "
            << t.Seconds() << " secs";
  return pruned;
}

DAGNetBase::ExecutionChains computeChains(
    const std::vector<internal::OperatorNode>& orig_nodes) {
  const std::vector<internal::OpGraphNode> nodes = pruneOpNodeGraph(orig_nodes);
  vector<int> initial_frontier;
  for (int idx = 0; idx < nodes.size(); ++idx) {
    if (nodes[idx].parents_.size() == 0) {
      initial_frontier.push_back(idx);
    }
  }

  // We need to construct the node_seen_count to know how many inner edges each
  // node has.
  std::unordered_map<OpIndex, int> node_seen_count;

  for (int root_index : initial_frontier) {
    const auto& root = nodes[root_index];
    std::stack<std::pair<OpIndex, std::vector<int>::const_iterator>>
        depth_stack;
    depth_stack.push(make_pair(root_index, root.children_.begin()));
    node_seen_count[root_index]++;
    CAFFE_ENFORCE(
        node_seen_count[root_index] == 1,
        "root node ",
        root_index,
        " visit count must be == 1");

    while (depth_stack.size() > 0) {
      auto cur = depth_stack.top();
      depth_stack.pop();
      if (cur.second != nodes[cur.first].children_.end()) {
        OpIndex node_index = *cur.second;
        node_seen_count[node_index]++;
        cur.second++;
        depth_stack.push(cur);
        if (node_seen_count[node_index] == 1) {
          // Visit each child only once.
          depth_stack.push(
              make_pair(node_index, nodes[node_index].children_.begin()));
        }
      }
    }
  }
  // Now, we compute the set of execution chains An execution chain is
  // a linear set of nodes that can be executed on a single stream
  // (e.g. a chain of single input, single output operators)
  DAGNetBase::ExecutionChains chains;
  std::unordered_set<OpIndex> seen_nodes;
  std::vector<OpIndex> chain;
  std::pair<OpIndex, std::vector<int>::const_iterator> cur;
  std::stack<std::pair<OpIndex, std::vector<int>::const_iterator>> depth_stack;
  auto check_current_for_chaining = [&]() -> bool {
    return (
        node_seen_count[cur.first] == 1 &&
        (chain.size() == 0 || sameDevice(
                                  orig_nodes[cur.first].operator_->def(),
                                  orig_nodes[chain.back()].operator_->def())));
  };
  auto commit_chain = [&]() {
    if (chain.size() > 0) {
      CAFFE_ENFORCE(
          chains.insert({chain.front(), chain}).second,
          "Chain ",
          chain.front(),
          " was already added.");
      VLOG(2) << "Added chain: " << chain.front() << "with elements";
      for (auto ch : chain) {
        VLOG(2) << ch << ", ";
      }
      chain.clear();
    }
  };
  auto depth_traverse = [&]() {
    while (cur.second != nodes[cur.first].children_.end() &&
           seen_nodes.find(*cur.second) != seen_nodes.end()) {
      cur.second++;
    }

    if (cur.second != nodes[cur.first].children_.end()) {
      auto next = make_pair(*cur.second, nodes[*cur.second].children_.begin());
      depth_stack.push(cur);
      depth_stack.push(next);
    }
  };
  for (int root_index : initial_frontier) {
    depth_stack.push(
        make_pair(root_index, nodes[root_index].children_.begin()));
    while (depth_stack.size() > 0) {
      cur = depth_stack.top();
      depth_stack.pop();
      if (seen_nodes.find(cur.first) == seen_nodes.end()) {
        seen_nodes.insert(cur.first);
        // Has one child, can be candidate for chain or can be added to the
        // previous chain.
        if (nodes[cur.first].children_.size() == 1) {
          if (check_current_for_chaining()) {
            // Add oneself to the current chain.
            VLOG(1) << "Adding to existing chain" << cur.first;
            chain.push_back(cur.first);
            int index = *nodes[cur.first].children_.begin();
            depth_stack.push(make_pair(index, nodes[index].children_.begin()));
          } else {
            // Can't belong to the previous chain, commit previous chain and
            // start a new one.
            commit_chain();
            chain.push_back(cur.first);
            int index = *nodes[cur.first].children_.begin();
            depth_stack.push(make_pair(index, nodes[index].children_.begin()));
          }
        } else if (
            nodes[cur.first].children_.size() == 0 &&
            check_current_for_chaining()) {
          // Add current node to the current chain and commit.
          chain.push_back(cur.first);
          commit_chain();
        } else {
          // Node has more than one child.
          commit_chain();
          // Add current node as an independent chain since it won't be a part
          // of a bigger chain.
          chain.push_back(cur.first);
          commit_chain();
          depth_traverse();
        }
      } else {
        // This node has been seen before, we will only traverse its children.
        // Commit any pending chains and continue traversing.
        commit_chain();
        depth_traverse();
      }
    } // End while

    // Check if this if is even needed.
    commit_chain();
  }
  CAFFE_ENFORCE(
      seen_nodes.size() == nodes.size(),
      "Haven't seen all the nodes, expected number of nodes ",
      nodes.size(),
      ", but seen only ",
      seen_nodes.size(),
      ".");
  return chains;
}
}

DAGNetBase::DAGNetBase(const NetDef& net_def, Workspace* ws)
    : NetBase(net_def, ws), operator_nodes_(net_def.op_size()), iter_(0) {
  // Blob creator allows us to track which operator created which blob.
  VLOG(1) << "Constructing DAGNet " << net_def.name();
  std::map<string, int> blob_creator;
  std::map<string, std::set<int>> blob_readers;
  bool net_def_has_device_option = net_def.has_device_option();
  // Initialize the operators
  for (int idx = 0; idx < net_def.op_size(); ++idx) {
    const OperatorDef& op_def = net_def.op(idx);
    VLOG(1) << "Creating operator #" << idx << ": " << op_def.name() << ":"
            << op_def.type();
    if (!op_def.has_device_option() && net_def_has_device_option) {
      OperatorDef temp_def(op_def);
      temp_def.mutable_device_option()->CopyFrom(net_def.device_option());
      operator_nodes_[idx].operator_ = CreateOperator(temp_def, ws, idx);
    } else {
      operator_nodes_[idx].operator_ = CreateOperator(op_def, ws, idx);
    }
    // Check the inputs, and set up parents if necessary. This addressese the
    // read after write case.
    auto checkInputs =
        [&](const google::protobuf::RepeatedPtrField<std::string>& inputs) {
          for (const string& input : inputs) {
            if (blob_creator.count(input) == 0) {
              VLOG(1) << "Input " << input << " not produced by this net. "
                      << "Assuming it is pre-existing.";
            } else {
              int parent = blob_creator[input];
              VLOG(1) << "op dependency (RaW " << input << "): " << parent
                      << "->" << idx;
              operator_nodes_[idx].parents_.push_back(parent);
              operator_nodes_[parent].children_.push_back(idx);
            }
            // Add the current idx to the readers of this input.
            blob_readers[input].insert(idx);
          }
        };
    checkInputs(op_def.input());
    checkInputs(op_def.control_input());

    // Check the outputs.
    for (const string& output : op_def.output()) {
      if (blob_creator.count(output) != 0) {
        // This addresses the write after write case - we will assume that all
        // writes are inherently sequential.
        int waw_parent = blob_creator[output];
        VLOG(1) << "op dependency (WaW " << output << "): " << waw_parent
                << "->" << idx;
        operator_nodes_[idx].parents_.push_back(waw_parent);
        operator_nodes_[waw_parent].children_.push_back(idx);
      }
      // This addresses the write after read case - we will assume that writes
      // should only occur after all previous reads are finished.
      for (const int war_parent : blob_readers[output]) {
        VLOG(1) << "op dependency (WaR " << output << "): " << war_parent
                << "->" << idx;
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

  execution_chains_ =
      (FLAGS_caffe2_disable_chaining ? singleChains(operator_nodes_)
                                     : computeChains(operator_nodes_));

  // Tag operator nodes that start chains
  for (int i = 0; i < operator_nodes_.size(); ++i) {
    auto& node = operator_nodes_[i];
    if (execution_chains_.find(i) != execution_chains_.end()) {
      node.is_chain_start_ = true;
    } else {
      node.is_chain_start_ = false;
    }
    node.runtime_parent_count_ = 0;
  }

  LOG(INFO) << "Number of parallel execution chains "
            << execution_chains_.size()
            << " Number of operators = " << net_def.op_size();
  // TODO: do we want to make sure that there are no loops in the
  // dependency graph?

  // Figure out the initial frontier - this is the one we will feed into the job
  // queue to start a run.
  for (int idx = 0; idx < operator_nodes_.size(); ++idx) {
    if (operator_nodes_[idx].parents_.size() == 0) {
      initial_frontier_.push_back(idx);
    }
  }
  // Finally, start the workers.
  int num_workers = net_def.has_num_workers() ? net_def.num_workers() : 1;
  CAFFE_ENFORCE(num_workers > 0, "Must have a positive number of workers.");
  if (num_workers == 1) {
    LOG(WARNING) << "Number of workers is 1: this means that all operators "
                 << "will be executed sequentially. Did you forget to set "
                 << "num_workers in the NetDef?";
  }
  num_workers_ = num_workers;
  num_workers_first_iteration_ = num_workers_;

  // Option to start only one thread for first iteration.
  // This hack is needed to prevent deadlocks happening with CUDA and
  // concurrent allocations that operators do when run the first time.
  ArgumentHelper arg_helper(net_def);
  if (arg_helper.HasArgument("first_iter_only_one_worker")) {
    if (arg_helper.GetSingleArgument<int64_t>(
            "first_iter_only_one_worker", 0)) {
      num_workers_first_iteration_ = 1;
    }
  }
}

DAGNetBase::~DAGNetBase() {
  if (job_queue_) {
    job_queue_->NoMoreJobs();
    VLOG(1) << "Joining workers.";
    for (auto& worker : workers_) {
      worker.join();
    }
  }
}

bool DAGNetBase::Run() {
  if (observer_) {
    observer_->Start();
  }
  // Lock run_in_progress_ to prevent concurrent Run()s.
  std::unique_lock<std::mutex> run_lock(run_in_progress_);
  VLOG(1) << "Running parallel net.";
  // First, set up job queue.
  remaining_ops_ = operator_nodes_.size();
  success_ = true;
  iter_++;
  if (!job_queue_) {
    job_queue_ = caffe2::make_unique<SimpleQueue<int>>();
  }
  // Figure out number of workers to start.
  auto num_workers_to_start = num_workers_ - workers_.size();
  if (iter_ == 1) {
    num_workers_to_start = num_workers_first_iteration_;
  }
  // Ensure the number of workers matches the defined in case
  // any of the previously started threads terminated.
  for (auto i = 0; i < num_workers_to_start; i++) {
    VLOG(1) << "Start worker #" << workers_.size();
    workers_.push_back(std::thread(&DAGNetBase::WorkerFunction, this));
  }
  // Initialize the runtime parent count.
  for (auto& node : operator_nodes_) {
    node.runtime_parent_count_ = node.parents_.size();
  }
  // Kickstart the job queue.
  for (auto& value : initial_frontier_) {
    job_queue_->Push(value);
  }
  // Wait for failure or completed execution.
  {
    std::unique_lock<std::mutex> mutex_lock(remaining_ops_mutex_);
    for (;;) {
      if (remaining_ops_ == 0 || !success_) {
        break;
      }
      cv_.wait(mutex_lock);
    }
  }
  // Wait for all workers to terminate after failure.
  // If there is a failure, it is unlikely that the net is executed
  // again without modifications. Therefore it's easier to let the
  // workers terminate here, versus adding a drain state to make the
  // sure the job queue is cleared.
  if (!success_) {
    for (auto& worker : workers_) {
      worker.join();
    }
    workers_.clear();
    job_queue_.reset(nullptr);
    return success_;
  }
  VLOG(2) << "All ops finished running.";
  for (const auto& op : operator_nodes_) {
    CAFFE_ENFORCE(
        op.runtime_parent_count_ == 0,
        "Operator ",
        op.operator_->def().name(),
        "(",
        op.operator_->def().type(),
        ") has some runtime parents left.");
  }
  if (observer_) {
    observer_->Stop();
  }
  // If the above while loop finished, we know that the current run finished.
  return success_;
}

void DAGNetBase::WorkerFunction() {
  // WorkerFunctions() is an infinite loop until there are no more jobs to run.
  while (true) {
    int idx = 0;

    // Return if there are no more operators to run (e.g. the
    // DAGNetBase is destructing, or there was an error on another
    // worker and we're cleaning up).
    if (!job_queue_->Pop(&idx)) {
      return;
    }

    VLOG(1) << "Running operator #" << idx << " "
            << operator_nodes_[idx].operator_->def().name() << "("
            << operator_nodes_[idx].operator_->def().type() << ").";
    CAFFE_ENFORCE(
        execution_chains_.find(idx) != execution_chains_.end(),
        "Can't find chain ",
        idx,
        ".");
    const auto& chain = execution_chains_[idx];
    bool this_success = RunAt(execution_chains_[idx]);
    if (!this_success) {
      LOG(ERROR) << "Operator chain failed: "
                 << ProtoDebugString(operator_nodes_[idx].operator_->def());
    }

    // Do book-keeping
    std::vector<int> chains_to_queue;
    for (const auto idx : chain) {
      for (const auto child : operator_nodes_[idx].children_) {
        const int count = --operator_nodes_[child].runtime_parent_count_;
        CAFFE_ENFORCE(
            count >= 0,
            "Found runtime parent count smaller than zero for ",
            "operator node ",
            operator_nodes_[child].operator_->def().name(),
            "(",
            operator_nodes_[child].operator_->def().type(),
            ").");

        if (count != 0) {
          continue;
        }

        if (operator_nodes_[child].is_chain_start_) {
          VLOG(2) << "Pushing chain #" << child << " to queue.";
          chains_to_queue.push_back(child);
        }
      }
    }

    // Notify the caller of Run
    {
      std::unique_lock<std::mutex> mutex_lock(remaining_ops_mutex_);
      remaining_ops_ -= chain.size();
      CAFFE_ENFORCE(remaining_ops_ >= 0);
      success_ &= this_success;
      cv_.notify_one();

      // Terminate thread if this or any other operator chain failed.
      if (!success_) {
        job_queue_->NoMoreJobs();
        return;
      }

      // Queue follow up operator chains.
      // Can't do this inline because it can race with another thread
      // calling NoMoreJobs(). So the lock needs to be held on push.
      for (const auto idx : chains_to_queue) {
        job_queue_->Push(idx);
      }
    }

    VLOG(2) << "Finished executing operator #" << idx;
  }
}

vector<float> DAGNetBase::TEST_Benchmark(
    const int warmup_runs,
    const int main_runs,
    const bool run_individual) {
  LOG(INFO) << "Starting benchmark.";
  LOG(INFO) << "Running warmup runs.";
  CAFFE_ENFORCE(
      warmup_runs >= 0,
      "Number of warm up runs should be non negative, provided ",
      warmup_runs,
      ".");
  for (int i = 0; i < warmup_runs; ++i) {
    CAFFE_ENFORCE(Run(), "Warmup run ", i, " has failed.");
  }

  LOG(INFO) << "Main runs.";
  CAFFE_ENFORCE(
      main_runs >= 0,
      "Number of main runs should be non negative, provided ",
      main_runs,
      ".");
  Timer timer;
  for (int i = 0; i < main_runs; ++i) {
    CAFFE_ENFORCE(Run(), "Main run ", i, " has failed.");
  }
  auto millis = timer.MilliSeconds();
  LOG(INFO) << "Main run finished. Milliseconds per iter: "
            << millis / main_runs
            << ". Iters per second: " << 1000.0 * main_runs / millis;

  if (run_individual) {
    LOG(INFO) << "DAGNet does not do per-op benchmark. To do so, "
                 "switch to a simple net type.";
  }
  return vector<float>{millis / main_runs};
}

class DAGNet : public DAGNetBase {
 public:
  using DAGNetBase::DAGNetBase;

 protected:
  bool RunAt(const std::vector<int>& chain) override {
    const auto& net_name = name_.c_str();
    for (const auto i : chain) {
      const auto& op = operator_nodes_[i].operator_.get();
      const auto& op_name = op->def().name().c_str();
      const auto& op_type = op->def().type().c_str();
      CAFFE_SDT(operator_start, net_name, op_name, op_type, op);
      const auto success = operator_nodes_[i].operator_->Run();
      CAFFE_SDT(operator_done, net_name, op_name, op_type, op);
      if (!success) {
        return false;
      }
    }
    return true;
  }
};

namespace {

REGISTER_NET(dag, DAGNet);
}

} // namespace caffe2
