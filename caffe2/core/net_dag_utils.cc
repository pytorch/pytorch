#include "caffe2/core/net_dag_utils.h"

#include <set>
#include <stack>
#include <unordered_map>
#include <unordered_set>

#include "caffe2/core/operator.h"
#include "caffe2/core/static_tracepoint.h"
#include "caffe2/core/timer.h"
#include "caffe2/opt/converter.h"
#include "caffe2/proto/caffe2_pb.h"
#include "caffe2/utils/proto_utils.h"

#include "nomnigraph/Graph/Algorithms.h"

namespace caffe2 {
namespace dag_utils {

namespace {
void prune(int node_idx, std::vector<OpGraphNode>& nodes) {
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
    CAFFE_ENFORCE(curr < (int)ancestors.size(), "Out of bound access");
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
std::vector<OpGraphNode> pruneOpNodeGraph(
    const std::vector<OperatorNode>& nodes) {
  Timer t;
  std::vector<OpGraphNode> pruned;

  // Create a separate list of pruned operatornodes used
  // for the chaining computation. Because of the unique_ptr
  // in the OperatorNode, we cannot do a copy but have to
  // copy just the fields we need.
  for (auto& node : nodes) {
    OpGraphNode nd;
    nd.children_ = node.children_;
    nd.parents_ = node.parents_;
    nd.num_orig_parents = nd.parents_.size();
    pruned.push_back(nd);
  }

  for (int i = 0; i < (int)pruned.size(); ++i) {
    if (pruned[i].parents_.size() == 0) {
      prune(i, pruned);
    }
  }

  LOG(INFO) << "Operator graph pruning prior to chain compute took: "
            << t.Seconds() << " secs";
  return pruned;
}

void updateOperatorNodes(
    std::vector<OperatorNode>& nodes,
    const ExecutionChains& chains) {
  for (int i = 0; i < (int)nodes.size(); ++i) {
    auto& node = nodes[i];
    if (chains.find(i) != chains.end()) {
      node.is_chain_start_ = true;
    } else {
      node.is_chain_start_ = false;
    }
    node.runtime_parent_count_ = 0;
    node.scheduled_.clear();
  }
}
} // namespace

using namespace nom::repr;
using DepGraph = nom::Graph<NNGraph::NodeRef>;

// \brief This function prunes edges in the dependency
// graph to increase the chaining opportunity.
// It does not eliminate parallelism opportunity.
void optimizeDependencyGraph(DepGraph* deps) {
  auto edges = deps->getMutableEdges();
  for (const auto& edge : edges) {
    auto tail = edge->tail();
    auto head = edge->head();
    deps->deleteEdge(edge);
    std::unordered_set<DepGraph::NodeRef> seen;
    nom::algorithm::reachable<DepGraph>(tail, nullptr, &seen);
    // Removing that edge removes a dominator, which is invalid
    if (!seen.count(head)) {
      deps->createEdge(tail, head);
    }
  }
}

ExecutionChains computeChains(
    const caffe2::NetDef& predict_net,
    std::vector<OperatorNode>& orig_nodes) {
  // These serve as the map into predict_net.op()
  std::vector<NNGraph::NodeRef> nom_ops;
  auto nn = convertToNNModule(predict_net, false, &nom_ops);
  CAFFE_ENFORCE_EQ(nom_ops.size(), predict_net.op().size());

  // Create a map from NodeRef to index into predict_net.op()
  // Now we can use pure nomnigraph functions and map back later
  std::unordered_map<NNGraph::NodeRef, int> nom_op_to_pos;
  for (auto idx = 0; idx < nom_ops.size(); ++idx) {
    nom_op_to_pos[nom_ops[idx]] = idx;
  }

  // The algorithm:
  // 1) create dependency graph of ops
  // 2) for all nodes thats have multiple in edges, remove all in edges
  // 3) for all nodes thats have multiple out edges, remove all out edges
  // 4) return the components as chains

  // Caveats that can easily be handled
  // 1) Cannot have a chain that crosses device options
  //    insert extra edge at each boundary
  // 2) All CPU async ops have to be the last op in a chain
  //    insert extra out edge
  DepGraph deps;

  // Map NodeRef to the node in the dependency graph
  std::unordered_map<NNGraph::NodeRef, DepGraph::NodeRef> dep_map;
  for (const auto& node : nn::filter<NeuralNetOperator>(nn)) {
    dep_map[node] = deps.createNode(node);
  }

  // 1) Create dependency graph
  for (const auto& node : nn::filter<NeuralNetOperator>(nn)) {
    for (const auto& output : nn::getOutputs(node)) {
      for (const auto& consumer : nn::getConsumers(output)) {
        // Record single dependencies first
        if (!deps.hasEdge(dep_map[node], dep_map[consumer])) {
          deps.createEdge(dep_map[node], dep_map[consumer]);
        }
      }
    }
  }

  optimizeDependencyGraph(&deps);

  // Fixup device boundary and async op issues
  for (const auto& dep : deps.getMutableNodes()) {
    int op_idx = nom_op_to_pos[dep->data()];
    auto d1 = orig_nodes.at(op_idx).operator_->device_option();
    auto outEdges = dep->getOutEdges();
    for (const auto& outEdge : outEdges) {
      int op2_idx = nom_op_to_pos[outEdge->head()->data()];
      auto d2 = orig_nodes.at(op2_idx).operator_->device_option();
      if (!IsSameDevice(d1, d2)) {
        deps.createEdge(dep, outEdge->head());
      }
    }
    if (d1.device_type() == PROTO_CUDA) {
      continue;
    }
    if (orig_nodes.at(op_idx).operator_->HasAsyncPart()) {
      outEdges = dep->getOutEdges();
      for (const auto& outEdge : outEdges) {
        // Clone out edges
        deps.createEdge(outEdge->tail(), outEdge->head());
      }
    }
  }

  // 2) Prune in edges if multiplicity > 1
  // 3) Prune out edges if multiplicity > 1
  for (const auto& dep : deps.getMutableNodes()) {
    auto inEdges = dep->getInEdges();
    if (inEdges.size() > 1) {
      for (const auto& inEdge : inEdges) {
        NOM_REQUIRE_OR_CONT(inEdge);
        deps.deleteEdge(inEdge);
      }
    }
    auto outEdges = dep->getOutEdges();
    if (outEdges.size() > 1) {
      for (const auto& outEdge : outEdges) {
        NOM_REQUIRE_OR_CONT(outEdge);
        deps.deleteEdge(outEdge);
      }
    }
  }

  // 4) Return components as chains
  std::vector<DepGraph::NodeRef> chain_starts;
  for (const auto& dep : deps.getMutableNodes()) {
    if (dep->getInEdges().size() == 0) {
      chain_starts.emplace_back(dep);
    }
  }

  ExecutionChains chains;
  for (const auto& dep : chain_starts) {
    DepGraph::NodeRef front = dep;
    std::vector<int> ops;
    do {
      ops.emplace_back(nom_op_to_pos[front->data()]);
      auto outEdges = front->getOutEdges();
      if (outEdges.size()) {
        front = outEdges.at(0)->head();
      } else {
        front = nullptr;
      }
    } while (front);
    chains[nom_op_to_pos[dep->data()]] = ops;
  }

  updateOperatorNodes(orig_nodes, chains);
  return chains;
}

ExecutionChains computeChains(std::vector<OperatorNode>& orig_nodes) {
  const std::vector<OpGraphNode> nodes = pruneOpNodeGraph(orig_nodes);
  vector<int> initial_frontier;
  for (int idx = 0; idx < (int)nodes.size(); ++idx) {
    if (nodes[idx].parents_.size() == 0) {
      initial_frontier.push_back(idx);
    }
  }

  // We need to construct the node_seen_count to know how many inner edges each
  // node has.
  std::unordered_map<int, int> node_seen_count;

  for (int root_index : initial_frontier) {
    const auto& root = nodes[root_index];
    std::stack<std::pair<int, std::vector<int>::const_iterator>> depth_stack;
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
        int node_index = *cur.second;
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
  ExecutionChains chains;
  std::unordered_set<int> seen_nodes;
  std::vector<int> chain;
  std::pair<int, std::vector<int>::const_iterator> cur;
  std::stack<std::pair<int, std::vector<int>::const_iterator>> depth_stack;
  auto check_current_for_chaining = [&]() -> bool {
    return (
        node_seen_count[cur.first] == 1 &&
        (chain.size() == 0 ||
         (
             // A chain of operators is executed without additional
             // synchronization by calling RunAsync sequentially on each
             // operator and passing the same stream id on each call.
             // RunAsync may schedule an async computation on device.
             // In order to be scheduled on the same chain two operators
             // (parent and dependent) need to satisfy:
             //  1. Both ops are on the same device _and_
             //  2. Parent op does not have an async part or
             //     dependent op can be executed as an async dependency

             IsSameDevice(
                 orig_nodes[cur.first].operator_->device_option(),
                 orig_nodes[chain.back()].operator_->device_option()) &&
             (!orig_nodes[chain.back()].operator_->HasAsyncPart() ||
              orig_nodes[cur.first].operator_->SupportsAsyncScheduling()))));
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

  updateOperatorNodes(orig_nodes, chains);
  return chains;
}

// Here chains are essentially groups, we used chain/group interchangeably
ExecutionChains computeGroups(std::vector<OperatorNode>& orig_nodes) {
  const std::vector<OpGraphNode> nodes = pruneOpNodeGraph(orig_nodes);
  ExecutionChains chains;
  std::vector<int> sync_frontier;
  std::vector<int> async_frontier;

  std::vector<int> in_degrees;
  in_degrees.reserve(nodes.size());
  std::transform(
      nodes.begin(),
      nodes.end(),
      std::back_inserter(in_degrees),
      [](const OpGraphNode& n) { return n.parents_.size(); });

  // Screen out the primary root nodes
  for (int idx = 0; idx < (int)nodes.size(); ++idx) {
    if (in_degrees[idx] == 0) {
      if (orig_nodes[idx].operator_->HasAsyncPart()) {
        async_frontier.push_back(idx);
      } else {
        sync_frontier.push_back(idx);
      }
    }
  }

  // We check sync ops on the froniter first and then async ops. This gives us a
  // head start to execute sync ops locally while waiting for async ops to
  // finish.
  std::queue<int> q;
  while (!(async_frontier.empty() && sync_frontier.empty())) {
    // Sync ops
    for (const auto i : sync_frontier) {
      q.push(i);
    }
    sync_frontier.clear();
    std::vector<int> chain;
    while (!q.empty()) {
      int idx = q.front();
      q.pop();
      chain.push_back(idx);
      for (int child : nodes[idx].children_) {
        if (--in_degrees[child] == 0) {
          if (orig_nodes[child].operator_->HasAsyncPart()) {
            async_frontier.push_back(child);
          } else {
            q.push(child);
          }
        }
      }
    }
    // add the whole group of continuous sync ops into one chain
    if (!chain.empty()) {
      chains.emplace(chain.front(), chain);
    }

    // Async ops
    for (const auto i : async_frontier) {
      q.push(i);
    }
    async_frontier.clear();
    while (!q.empty()) {
      int idx = q.front();
      q.pop();
      // Put each individual node as a new chain
      chains[idx] = {idx};
      for (int child : nodes[idx].children_) {
        if (--in_degrees[child] == 0) {
          if (orig_nodes[child].operator_->HasAsyncPart()) {
            q.push(child);
          } else {
            sync_frontier.push_back(child);
          }
        }
      }
    }
  }

  updateOperatorNodes(orig_nodes, chains);
  return chains;
}

ExecutionChains singleChains(std::vector<OperatorNode>& nodes) {
  ExecutionChains chains;
  for (int i = 0; i < (int)nodes.size(); ++i) {
    chains[i] = {i};
  }
  updateOperatorNodes(nodes, chains);
  return chains;
}

std::vector<OperatorNode> prepareOperatorNodes(
    const std::shared_ptr<const NetDef>& net_def,
    Workspace* ws) {
  std::vector<OperatorNode> operator_nodes(net_def->op_size());
  std::map<string, int> blob_creator;
  std::map<string, std::set<int>> blob_readers;
  bool net_def_has_device_option = net_def->has_device_option();
  // Initialize the operators
  for (int idx = 0; idx < net_def->op_size(); ++idx) {
    const OperatorDef& op_def = net_def->op(idx);
    VLOG(1) << "Creating operator #" << idx << ": " << op_def.name() << ": "
            << op_def.type();
    if (!op_def.has_device_option() && net_def_has_device_option) {
      OperatorDef temp_def(op_def);
      temp_def.mutable_device_option()->CopyFrom(net_def->device_option());
      operator_nodes[idx].operator_ = CreateOperator(temp_def, ws, idx);
    } else {
      auto op = CreateOperator(op_def, ws, idx);
      op->set_debug_def(
          std::shared_ptr<const OperatorDef>{net_def, &(net_def->op(idx))});
      operator_nodes[idx].operator_ = std::move(op);
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
              operator_nodes[idx].parents_.push_back(parent);
              operator_nodes[parent].children_.push_back(idx);
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
        operator_nodes[idx].parents_.push_back(waw_parent);
        operator_nodes[waw_parent].children_.push_back(idx);
      }
      // This addresses the write after read case - we will assume that writes
      // should only occur after all previous reads are finished.
      for (const int war_parent : blob_readers[output]) {
        VLOG(1) << "op dependency (WaR " << output << "): " << war_parent
                << "->" << idx;
        operator_nodes[idx].parents_.push_back(war_parent);
        operator_nodes[war_parent].children_.push_back(idx);
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
  for (int i = 0; i < (int)operator_nodes.size(); ++i) {
    auto& node = operator_nodes[i];
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

  return operator_nodes;
}

std::vector<OpGraphNode> prepareChainGraphNodes(
    const std::vector<dag_utils::OperatorNode>& operator_nodes,
    const std::vector<std::vector<int>>& execution_chains) {
  std::unordered_map<int, int> op_to_chain_idx;
  for (int chain_idx = 0; chain_idx < (int)execution_chains.size(); ++chain_idx) {
    const auto& chain_indices = execution_chains[chain_idx];
    for (const auto& chain_op_idx : chain_indices) {
      CAFFE_ENFORCE(!op_to_chain_idx.count(chain_op_idx));
      op_to_chain_idx[chain_op_idx] = chain_idx;
    }
  }

  std::vector<OpGraphNode> chain_nodes(execution_chains.size());
  for (int op_idx = 0; op_idx < (int)operator_nodes.size(); ++op_idx) {
    CAFFE_ENFORCE(op_to_chain_idx.count(op_idx));
    auto chain_idx = op_to_chain_idx[op_idx];
    auto& chain = chain_nodes[chain_idx];
    auto& op_node = operator_nodes[op_idx];

    for (const auto& child_idx : op_node.children_) {
      CAFFE_ENFORCE(op_to_chain_idx.count(child_idx));
      auto child_chain_idx = op_to_chain_idx[child_idx];
      if (child_chain_idx != chain_idx) {
        auto it = std::find(
            chain.children_.begin(), chain.children_.end(), child_chain_idx);
        if (it == chain.children_.end()) {
          chain.children_.push_back(child_chain_idx);
        }
      }
    }

    for (const auto& parent_idx : op_node.parents_) {
      CAFFE_ENFORCE(op_to_chain_idx.count(parent_idx));
      auto parent_chain_idx = op_to_chain_idx[parent_idx];
      if (parent_chain_idx != chain_idx) {
        auto it = std::find(
            chain.parents_.begin(), chain.parents_.end(), parent_chain_idx);
        if (it == chain.parents_.end()) {
          chain.parents_.push_back(parent_chain_idx);
        }
      }
    }
  }

  return chain_nodes;
}

} // namespace dag_utils
} // namespace caffe2
