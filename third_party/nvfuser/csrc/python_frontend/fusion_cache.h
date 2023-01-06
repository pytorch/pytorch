#pragma once
#include <c10/macros/Export.h>

#include <kernel_cache.h>
#include <python_frontend/fusion_record.h>

#include <memory>

//! nvFuser Fusion IR namespace abbreviation
namespace Nvf = torch::jit::fuser::cuda;

namespace nvfuser {

//! \struct TrieNode
//! \brief Is the container for a Node in a prefix tree or trie
//! where each node represents a statement in a fusion definition and
//! the leaf Nodes represent a complete Fusion that is cached.

struct TORCH_CUDA_CU_API TrieNode {
  TrieNode(RecordFunctor* rec, size_t _fusion_id = 0);

  // Queries whether the entry denotes a leaf node which also represents
  // a the end of Fusion entry in the cache.
  bool isTerminal() const;

  //! An entry's primary data is the record it holds
  std::unique_ptr<RecordFunctor> record;
  //! A hash map of the children for the current node.
  //! The hash map hashs a pointer to a RecordFunctor because
  //! the hash function is virtual.
  std::unordered_map<RecordFunctor*, std::unique_ptr<TrieNode>> children;
  //! An index into FusionCache's vector of nvFuser object that holds an
  //! unscheduled Fusion.  The id is only valid if the entry is terminal.
  size_t fusion_id;
  //! Count of times the Entry is traversed
  size_t visits;
};

//! \class FusionCache
//! \brief A singleton class used in the nvFuser python interface
//! to manage the caching of fusions.
//!
//! The fusion cache implements a prefix tree (trie) of records in order to
//! cache fusions.  A leaf of the tree with a terminal node contains a
//! container for caching the kernels generated for specific fusions.
//!
//! \todo Add the ability to evict a fusion.  There is currently a max number
//! of fusions that is checked to prevent a runaway case.

class TORCH_CUDA_CU_API FusionCache {
  //! The constructor is private given the FusionCache is only constructed
  //! as a singleton.
  FusionCache(size_t max_fusions);

  //! Copy and Assignment of the FusionCache is not supported
  FusionCache(const FusionCache&) = delete;
  FusionCache& operator=(const FusionCache&) = delete;

 public:
  //! The next 4 pubic methods are the python interface methods

  //! Gets a pointer to the singleton and creates a new one if necessary
  static FusionCache* get(size_t max_fusions = 8192);
  //! Number of fusions cached
  size_t numFusions() const;
  //! print cache stats
  void print(std::ostream& os);
  //! Reset Cache to an empty state
  static void reset();

  //! The rest of the public methods are only used in C++

  //! Queries the current trie node to see if a record matches one of its
  //! children
  c10::optional<TrieNode*> queryChildren(RecordFunctor* rec) const;
  //! Creates a child node for the current cache entry and an optional
  //! fusion_id is returned if the new entry is terminal
  c10::optional<size_t> createChild(RecordFunctor* rec);
  //! Resets the current cache pointer to the top of the tree
  void resetTriePtr();
  //! Traverses the trie from the current node to the child associated
  //! with the record given.
  void traverseTrie(RecordFunctor* rec);

  friend class FusionInterface;

 private:
  //! Returns the pointer to the current trie node
  TrieNode* triePtr() const;

  //! The static pointer to the FusionCache
  static FusionCache* singleton_;

  //! The max allowed number of fusions in the cache
  size_t max_fusions_;
  //! The root (start) of the prefix tree to start a cache look up of a given
  //! fusion definition.
  std::unique_ptr<TrieNode> root_;
  //! A pointer to the trie node in a cache lookup of a fusion definition.
  TrieNode* trie_ptr_;
  //! A vector of nvFuser Fusion IR fusions.
  std::vector<std::unique_ptr<Nvf::FusionExecutorCache>> fusions_;
  //! A vector of Terminal trie nodes for Stats collection
  std::vector<TrieNode*> terminal_nodes_;
};

} // namespace nvfuser
