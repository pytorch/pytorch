#include "ast.h"
#include "caffe2/opt/converter.h"
#include "nomnigraph/Transformations/SubgraphMatcher.h"

namespace nom {
namespace nql {

using Criteria = std::string;

using TestMatchGraph = nom::matcher::MatchGraph<nom::repr::NNGraph>;
using TestMatchPredicate = nom::matcher::MatchPredicate<nom::repr::NNGraph>;

// Each match is a struct of
// subgraph and map from the string used in the query to a NodeRef in the
// subgraph note: the maps are injective but not necessarily bijective -- if
// you use the same name in the query twice only one will be mapped.
//
// See `getMatches` to generate these structs.
struct MatchedSubgraph {
  // A subgraph that contains at least all the nodes in matchMap
  // This is the canonical match -- the matchMap is only a useful utility
  nom::repr::NNGraph::SubgraphType subgraph;

  // Provides safer access to matchMap with nicer semantics
  nom::repr::NNGraph::NodeRef operator[](const std::string& key) const;

  // Maps a variable name to a Node in a dataflow graph
  std::map<std::string, nom::repr::NNGraph::NodeRef> matchMap;
};

/// \brief Main graph matcher interface.
///
/// This class solves a problem of finding a matching subgraph, which is
/// specified in a text form.
class GraphMatcher {
 public:
  /// \brief Initialize subgraph pattern from \p STR.
  void initFromString(const char* str) {
    genMatcherFromIRStr(str);
  }
  /// \brief Initialize subgraph patter from IR stored in file \p fname.
  void initFromFile(const char* fname) {
    genMatcherFromIRFile(fname);
  }
  /// \brief Try to find the pattern in the given graph \p DF and return true
  /// if it was found.
  bool findSubgraph(nom::repr::NNGraph& df) {
    return doesMatch(df);
  }
  /// \brief Replace the found subgraph with another one.
  void replaceSubgraphWith() {
    CAFFE_THROW("Subgraph replacement is not implemented yet.");
  }
  /// \brief Return the matcher graph.
  TestMatchGraph* getMatcherGraph() {
    return &matchGraph_;
  }
  // TODO: Do we need this, or can we get it from getMatcherGraph?
  TestMatchGraph::NodeRef getMatcher() {
    return matchGraphRootNode_;
  }
  // \brief Return a mapping from IR variable name (std::string) to Node in the
  // matched graph.
  std::unordered_map<std::string, nom::repr::NNGraph::NodeRef> getMatchMap()
      const {
    return matchMap_;
  }

  // \brief Returns a vector of matches.
  std::vector<MatchedSubgraph> getMatches(nom::repr::NNGraph& df) const;

 private:
  std::unordered_map<std::string, nom::repr::NNGraph::NodeRef> matchMap_;
  std::unordered_map<std::string, TestMatchGraph::NodeRef> varMap_;
  std::unordered_map<std::string, TestMatchGraph::NodeRef> callMap_;
  TestMatchGraph matchGraph_;
  TestMatchGraph::NodeRef matchGraphRootNode_;
  bool syntaxIsValid_ = true;

  bool doesMatch(nom::repr::NNGraph& df) {
    if (!syntaxIsValid_) {
      return false;
    }
    matchMap_.clear();
    std::vector<nom::repr::NNGraph::NodeRef> Nodes = df.getMutableNodes();
    for (auto& Node : Nodes) {
      auto match =
          matchGraph_.isSubgraphMatch(Node, matchGraphRootNode_, true, true);
      if (match.isMatch()) {
        // Fill the match map
        auto subgraphMatcherMap = match.getMatchNodeMap();
        for (auto p : varMap_) {
          auto iter = subgraphMatcherMap->find(p.second);
          if (iter != subgraphMatcherMap->end()) {
            matchMap_[p.first] = iter->second;
          }
        }
        for (auto p : callMap_) {
          auto iter = subgraphMatcherMap->find(p.second);
          if (iter != subgraphMatcherMap->end()) {
            matchMap_[p.first] = iter->second;
          }
        }

        return true;
      }
    }
    return false;
  }
  TestMatchGraph::NodeRef genMatcherFromIRFile(const char* fname);
  TestMatchGraph::NodeRef genMatcherFromIRStr(const char* str);
  TestMatchGraph::NodeRef genMatcherFromASTGraph(ASTGraph* ast);
  TestMatchGraph::NodeRef genMatcherFromASTStmt(ASTStmt* stmt);
  TestMatchGraph::NodeRef genMatcherFromASTExpr(ASTExpr* expr, bool insertTemp);
};

// Node matches a criteria (string) if the data string is the same as the
// criteria. Special case: "*" will match any thing.
TestMatchPredicate testMatchPredicate(const Criteria& criteria);

// \brief Return a short string name for the given \param node.
// The function works with both tensors and operators.
std::string getNodeName(const nom::repr::NNGraph::NodeRef);

// \brief Return a string representing the given graph \param g.
// The returned string is a valid NQL query.
std::string convertToNQLString(nom::repr::NNGraph&);

void deallocTokenStrings();

} // namespace nql
} // namespace nom
