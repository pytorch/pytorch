#pragma once

#include "caffe2/core/common.h"
#include "caffe2/proto/caffe2_pb.h"
#include "caffe2/utils/proto_utils.h"
#include "caffe2/utils/string_utils.h"

#include <algorithm>
#include <unordered_map>
#include <unordered_set>

namespace caffe2 {

namespace transform {

/**
 *  Graph representation of an operator.
 */
struct CAFFE2_API Node {
 public:
  // Empty constructor for resize
  Node() {}

  // Alternate constructor
  Node(
      const OperatorDef& op,
      bool active,
      std::map<int, std::vector<string>> parents,
      std::map<int, std::vector<string>> children)
      : op(op), active(active), parents(parents), children(children) {}

  // The OperatorDef which this node represents.
  OperatorDef op;

  // Keeps track of if an operator has been deleted through a transformation.
  bool active = true;

  // Stores a pair (idx, blob_list),
  //  idx = index of the child
  //  blob_list = a list of strings, containing the blobs that connect the nodes
  std::map<int, std::vector<string>> parents;
  std::map<int, std::vector<string>> children;
};

/**
 *  Graph representation of a Netdef.
 */
struct CAFFE2_API Graph {
 public:
  /**
   * Given a subgraph, gets all of the parents of the subgraph, as well as
   * their associated blob names. Sorted by blob names.
   *
   * <string, int> := (name of blob writing into subgraph,
   *                  index of node that writes into subgraph using that blob)
   */
  const std::vector<std::pair<string, int>> GetSubgraphInput(
      const std::vector<int>& subgraph);

  /**
   * Given a subgraph, gets all of the children of the subgraph, as well as
   * their associated blob names. Sorted by blob names.
   *
   * <string, int> := (name of blob reading from subgraph,
   *                  index of node that reads from subgraph using that blob)
   */
  const std::vector<std::pair<string, int>> GetSubgraphOutput(
      const std::vector<int>& subgraph);

  /**
   * Graph generation.
   * Given a netdef, returns a Graph.
   *
   * Each node represents an operator.
   * An edge exists between two nodes if the parent op writes to a blob, which
   * is the input of the child blob, with no other op writing to the blob in
   * between the execution order.
   *
   * Time Complexity: O(E), where E is the number of blobs
   */
  explicit Graph(const NetDef& net_def);

  /**
   * Generates a NetDef Representation for the current graph.
   * Nodes are visited in topological order, which is proper Opdef ordering.
   * TODO(benz):
   * There exists conflicts with repeated blob names, where topological sorting
   * is not sufficient for correct netdef representation, unless blobs are
   * renamed.
   * For example, if after a transformation, We have operator ancestry:
   * A --> B --> C, and also A --> D --> E, where B -> C and D -> E uses the
   * same blob name, then A, B, D, E, C is a correct topological ordering,
   * but D will write to the blob that C reads from, instead of B.
   * Currently believe that there will always be ambiguity unless blobs are
   * renamed.
   * This is solved by performing SSA on all transformed blob names.
   */
  NetDef GetNetDef();

  /**
   * Deactivate a subgraph, and get rid of all edges into this subgraph.
   */
  void DeactivateSubgraph(std::vector<int> subgraph);

  size_t size() const {
    return nodes_.size();
  }

  void push_node(const Node& new_node) {
    return nodes_.push_back(new_node);
  }

  void resize_nodes(size_t new_size) {
    nodes_.resize(new_size);
  }

  // Index safe, less verbose way to access nodes
  inline const Node& node(size_t idx) const {
    return nodes_.at(idx);
  }

  inline Node& node(size_t idx) {
    return nodes_.at(idx);
  }

  inline bool is_node_active(size_t idx) {
    return node(idx).active;
  }

  inline const std::set<string>& external_input() const {
    return external_input_;
  }

  inline const std::set<string>& external_output() const {
    return external_output_;
  }

 private:
  const std::vector<std::pair<string, int>> GetSubgraphPerimeterHelper(
      bool from_children,
      const std::vector<int>& match);

  // Stores the netdef representation. Is updated upon calls to GetNetDef.
  NetDef netdef_;

  // Stores which blobs the graph reads from, and writes to.
  std::set<string> external_input_;
  std::set<string> external_output_;

  // Keeps track of all the Operators currently within graph, even if inactive.
  std::vector<Node> nodes_;
};

} // namespace transform

// Adds an operator def to a netdef.
// Returns the ptr, if you want to add anything extra (such as device_option)
CAFFE2_API OperatorDef* AddOp(
    NetDef* netdef_ptr,
    string op_type,
    std::vector<string> inputs,
    std::vector<string> outputs);

/**
 * This allows for the use of * and | to match operator types,
 * engines, or any other property that is represented by strings.
 *
 * For example, if we wanted to match an operator to Conv or FC, we can give:
 * "Conv|FC" as the type() of that op.
 */
CAFFE2_API bool MatchStrings(string p, string s);

/**
 * This ensures that each named arg that exists in the pattern exists in g_op,
 * is equal in value.
 */
CAFFE2_API bool MatchArguments(const OperatorDef& p_op, const OperatorDef& g_op);

} // namespace caffe2
