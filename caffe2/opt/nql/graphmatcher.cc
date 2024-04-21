#include "graphmatcher.h"
#include "ast.h"
#include "nomnigraph/Graph/Algorithms.h"

#include <mutex>

static std::mutex mtx_;

namespace nom {
namespace nql {
using namespace nom::repr;

NNGraph::NodeRef MatchedSubgraph::operator[](const std::string& key) const {
  auto search = matchMap.find(key);
  CAFFE_ENFORCE(
      search != matchMap.end(), "Could not find key in map of matches:", key);
  return search->second;
}

TestMatchGraph::NodeRef GraphMatcher::genMatcherFromASTExpr(
    ASTExpr* expr,
    bool insertTemp = false) {
  if (!expr->isCall()) {
    if (expr->starInputs()) {
      return matchGraph_.createNode(std::move(
          testMatchPredicate(Criteria("*")).starCount().nonTerminal()));
    }
    if (!varMap_.count(expr->name)) {
      varMap_[expr->name] = matchGraph_.createNode(
          std::move(testMatchPredicate(Criteria("*")).nonTerminal()));
    }
    return varMap_[expr->name];
  }

  std::vector<TestMatchGraph::NodeRef> children;
  for (auto child : expr->children) {
    children.push_back(genMatcherFromASTExpr(child, true));
  }

  auto res = matchGraph_.createNode(testMatchPredicate(Criteria(expr->name)));
  callMap_[expr->name] = res;
  for (auto child : children) {
    matchGraph_.createEdge(child, res);
  }

  if (insertTemp) {
    auto temp = matchGraph_.createNode(testMatchPredicate(Criteria("*")));
    matchGraph_.createEdge(res, temp);
    res = temp;
  }

  return res;
}

TestMatchGraph::NodeRef GraphMatcher::genMatcherFromASTStmt(ASTStmt* stmt) {
  auto right = genMatcherFromASTExpr(stmt->rhs);
  auto res = right;
  /* For cases like
   %x, %y = Foo(%z)
   for now we just say that both %x and %y are defined by node Foo, we don't
   distinguish them (i.e. we don't keep any information about their order. */
  for (auto v : stmt->lhs) {
    res = matchGraph_.createNode(testMatchPredicate(Criteria("*")));
    matchGraph_.createEdge(right, res);
    varMap_[v] = res;
  }
  return res;
}

void deallocTokenStrings() {
  for (auto p : tokens) {
    delete (std::string*)p;
  }
  tokens.clear();

  for (auto p : tokenVectors) {
    delete (std::vector<void*>*)p;
  }
  tokenVectors.clear();
}

TestMatchGraph::NodeRef GraphMatcher::genMatcherFromASTGraph(ASTGraph* ast) {
  matchGraph_ = TestMatchGraph();
  // TODO: Cleanup this.
  TestMatchGraph::NodeRef last = nullptr;
  if (ast->stmts.empty()) {
    syntaxIsValid_ = false; // Temporary solution, which works because we don't
                            // allow empty graphs.
  }

  for (auto stmt : ast->stmts) {
    auto r = genMatcherFromASTStmt(stmt);
    if (r) {
      last = r;
    }
  }

  return last;
}

TestMatchGraph::NodeRef GraphMatcher::genMatcherFromIRFile(const char* fname) {
  std::lock_guard<std::mutex> lock(mtx_);
  ASTGraph g;
  parseFile(fname, &g);
  matchGraphRootNode_ = genMatcherFromASTGraph(&g);
  deallocTokenStrings();
  return matchGraphRootNode_;
}

TestMatchGraph::NodeRef GraphMatcher::genMatcherFromIRStr(const char* str) {
  std::lock_guard<std::mutex> lock(mtx_);
  ASTGraph g;
  parseString(str, &g);
  matchGraphRootNode_ = genMatcherFromASTGraph(&g);
  deallocTokenStrings();
  return matchGraphRootNode_;
}

TestMatchPredicate testMatchPredicate(const Criteria& criteria) {
  auto predicate =
      TestMatchPredicate([criteria](nom::repr::NNGraph::NodeRef nodeRef) {
        std::string nodeLabel = getNodeName(nodeRef);
        return (criteria == "*" || criteria == nodeLabel);
      });
  predicate.setDebugString(criteria);
  return predicate;
}

// Helper function for convertToNQLString function.
// Given a node and a renameMap return the unique name for this node.
static std::string getNameForBlob(
    NNGraph::NodeRef node,
    const std::unordered_map<NNGraph::NodeRef, std::string>& renameMap) {
  if (renameMap.count(node)) {
    return renameMap.at(node);
  }
  return getNodeName(node);
}

// Helper function for convertToNQLString function.
// Given a node and a renameMap return a string representing the node, which
// looks something like:
//   %a = Op(%b, %c, %d)
static const std::string getNQLStringForBlob(
    NNGraph::NodeRef node,
    const std::unordered_map<NNGraph::NodeRef, std::string>& renameMap) {
  if (!nn::is<Data>(node) || !nn::hasProducer(node)) {
    return "";
  }
  NNGraph::NodeRef defOp = nn::getProducer(node);

  std::string result =
      getNameForBlob(node, renameMap) + " = " + getNodeName(defOp) + "(";
  int i = 0;
  for (auto inputTensor : nn::getInputs(defOp)) {
    if (i) {
      result += ", ";
    }
    result += getNameForBlob(inputTensor, renameMap);
    i++;
  }
  result += ")";
  return result;
}

// Helper function for convertToNQLString function.
// It takes a list of nodes and returns a map node->unique_name. The new names
// are based on the existing ones, but are also unique.
static std::unordered_map<NNGraph::NodeRef, std::string> computeDedupRenameMap(
    const std::vector<NNGraph::NodeRef>& nodes) {
  std::unordered_map<NNGraph::NodeRef, std::string> renameMap;
  std::unordered_set<std::string> takenNames;
  takenNames.clear();
  for (auto node : nodes) {
    std::string name = getNodeName(node);
    if (!isa<Data>(node->data())) {
      continue;
    }
    std::string newName = name;
    int dedupCounter = 0;
    while (takenNames.count(newName)) {
      newName = name + "_" + caffe2::to_string(dedupCounter);
      dedupCounter++;
    }
    renameMap[node] = newName;
    takenNames.insert(newName);
  }
  return renameMap;
}

std::vector<MatchedSubgraph> GraphMatcher::getMatches(
    nom::repr::NNGraph& df) const {
  std::vector<MatchedSubgraph> matches;
  if (!syntaxIsValid_) {
    return matches;
  }
  // Attempt to match at each node
  for (const auto& node : df.getMutableNodes()) {
    auto match = matchGraph_.isSubgraphMatch(node, matchGraphRootNode_, true);
    if (match.isMatch()) {
      MatchedSubgraph ms;
      ms.subgraph = *match.getMatchedSubgraph();
      // This is a map from the internal TestMatchGraph to the nodes in the
      // NNGraph
      auto match_graph_map = match.getMatchNodeMap();
      // We iterate through the "varMap_" map (string ->
      // TestMatchGraph::NodeRef) to generate string -> NNGraph::NodeRef
      for (auto p : varMap_) {
        auto iter = match_graph_map->find(p.second);
        if (iter != match_graph_map->end()) {
          ms.matchMap[p.first] = iter->second;
        }
      }
      for (auto p : callMap_) {
        auto iter = match_graph_map->find(p.second);
        if (iter != match_graph_map->end()) {
          ms.matchMap[p.first] = iter->second;
        }
      }
      matches.emplace_back(ms);
    }
  }
  return matches;
}

// \brief Return a short string name for the given \param node.
// The function works with both tensors and operators.
std::string getNodeName(const NNGraph::NodeRef node) {
  if (!node) {
    return "";
  }
  if (nn::is<NeuralNetOperator>(node)) {
    if (auto* op = nn::get<NeuralNetOperator>(node)) {
      return op->getName();
    }
  }
  if (nn::is<NeuralNetData>(node)) {
    if (auto tensor = nn::get<NeuralNetData>(node)) {
      return "%" + tensor->getName();
    }
  }
  return "";
}

// \brief Return a string representing the given graph \param g.
// The returned string is a valid NQL query.
std::string convertToNQLString(NNGraph& g) {
  // Order nodes in a topological order.
  // TODO: Currently tarjans mutates the graph, and that's the only reason we
  // are not using const reference for `g`. We need to fix tarjans so that it
  // doesn't mutate the graph and use const reference in this function too.
  auto topoMatch = nom::algorithm::tarjans(&g);
  std::vector<NNGraph::NodeRef> nodes;
  for (auto scc : topoMatch) {
    for (auto node : scc.getNodes()) {
      nodes.emplace_back(node);
    }
  }
  std::reverse(nodes.begin(), nodes.end());

  // Different nodes might have the same name. We want to change that so that
  // they are distinguishable by the name. NQL assumes that names are unique.
  std::unordered_map<NNGraph::NodeRef, std::string> renameMap =
      computeDedupRenameMap(nodes);

  // Going from top to bottom (nodes are in topological order), print all
  // nodes.
  std::string result = "def nn {\n";
  for (auto node : nodes) {
    std::string r = getNQLStringForBlob(node, renameMap);
    if (!r.empty()) {
      result += "  " + r + "\n";
    }
  }
  result += "}\n";
  return result;
}
}; // namespace nql
}; // namespace nom
