#include <python_frontend/fusion_cache.h>
#include <mutex>

namespace nvfuser {

static std::mutex fusion_cache_lock;
FusionCache* FusionCache::singleton_ = nullptr;

TrieNode::TrieNode(RecordFunctor* rec, size_t _fusion_id)
    : record(rec), children(), fusion_id(_fusion_id), visits(0) {}

bool TrieNode::isTerminal() const {
  return (record.get()->recordType() == RecordType::End);
}

FusionCache* FusionCache::get(size_t max_fusions) {
  std::lock_guard<std::mutex> guard(fusion_cache_lock);
  if (singleton_ == nullptr) {
    singleton_ = new FusionCache(max_fusions);
  }
  TORCH_CHECK(
      max_fusions >= singleton_->fusions_.size(),
      "The max fusions is set less than the number of fusions in the cache.");
  singleton_->max_fusions_ = max_fusions;
  return singleton_;
}

size_t FusionCache::numFusions() const {
  return fusions_.size();
}

void FusionCache::print(std::ostream& os) {
  os << "Total Fusions: " << fusions_.size() << "\n";

  // Does not make sense to print stats if the cache is disabled.
  if (fusions_.size() > 0) {
    os << "Cache Hits by Fusion Id:\n";
    auto total_cache_hits = 0;
    for (size_t i = 0; i < terminal_nodes_.size(); ++i) {
      // The first visit is a miss!
      auto visits = terminal_nodes_[i]->visits - 1;
      total_cache_hits += visits;
      os << "\t" << i << " -> " << visits << " hits\n";
    }

    auto hit_rate = static_cast<float>(total_cache_hits) /
        static_cast<float>(root_->visits) * 100.0;
    os << "Cache Lookups: " << root_->visits;
    os << " Cache Hits: " << total_cache_hits;
    os << " Hit Rate: " << hit_rate << "%\n";
  }
}

void FusionCache::reset() {
  std::lock_guard<std::mutex> guard(fusion_cache_lock);
  if (singleton_ != nullptr) {
    auto max_fusions = singleton_->max_fusions_;
    delete singleton_;
    singleton_ = new FusionCache(max_fusions);
  }
}

FusionCache::FusionCache(size_t max_fusions)
    : max_fusions_(max_fusions),
      root_(nullptr),
      trie_ptr_(nullptr),
      fusions_() {
  RecordFunctor* start = new StartRecord();
  root_ = std::make_unique<TrieNode>(start);
  trie_ptr_ = root_.get();
}

c10::optional<TrieNode*> FusionCache::queryChildren(RecordFunctor* rec) const {
  TORCH_CHECK(
      !triePtr()->isTerminal(),
      "There should be no children from a Terminal Node!");
  TORCH_CHECK(rec, "Record is null!");
  auto trie_node = triePtr()->children.find(rec);
  if (trie_node == std::end(triePtr()->children)) {
    return c10::nullopt;
  } else {
    return c10::optional<TrieNode*>(trie_node->second.get());
  }
}

c10::optional<size_t> FusionCache::createChild(RecordFunctor* rec) {
  c10::optional<size_t> result = c10::nullopt;
  TORCH_CHECK(
      !triePtr()->isTerminal(),
      "Cannot create a trie node from a terminal node!");
  TORCH_CHECK(rec, "Record is null!");

  size_t fusion_id = 0;
  if (rec->recordType() == RecordType::End) {
    TORCH_CHECK(
        (fusions_.size() + 1) <= max_fusions_,
        "The number of fusions in nvfuser has exceeded ",
        max_fusions_,
        "fusions.  The max_fusions for the FusionCache might need to be ",
        "increased if the max number is not being exceeded due to an error.");
    fusions_.push_back(std::make_unique<Nvf::FusionExecutorCache>(
        std::make_unique<Nvf::Fusion>()));
    fusion_id = fusions_.size() - 1;
    result = c10::optional<size_t>(fusion_id);
  }

  // Copying the record owned by the FusionDefinition that calls this function
  // so the trie owns a copy when the FusionDefinition gets destroyed rather
  // than managing a shared pointer that would only share with
  // FusionDefinition that creates a trie node but not cache lookups
  RecordFunctor* new_rec = rec->clone();
  triePtr()->children[new_rec] = std::make_unique<TrieNode>(new_rec, fusion_id);
  if (rec->recordType() == RecordType::End) {
    terminal_nodes_.push_back(triePtr()->children[new_rec].get());
  }
  if (Nvf::isDebugDumpEnabled(Nvf::DebugDumpOption::PythonFrontendDebug)) {
    std::stringstream ss;
    new_rec->print(ss);
    std::cout << "\nFusionDefinition: Create new trie node for: " << ss.str()
              << "\n";
  }
  return result;
}

void FusionCache::resetTriePtr() {
  trie_ptr_ = root_.get();
  TORCH_CHECK(triePtr()->record->recordType() == RecordType::Start);
  ++(triePtr()->visits);
}

void FusionCache::traverseTrie(RecordFunctor* rec) {
  TORCH_CHECK(
      !triePtr()->isTerminal(), "Cannot traverse trie from a terminal entry!");
  auto trie_node = triePtr()->children.find(rec);
  TORCH_CHECK(
      trie_node != std::end(triePtr()->children),
      "Trie Node for Trie Traverse is not found!");
  TORCH_CHECK(trie_node->second, "Record in Trie Node is null!");
  trie_ptr_ = trie_node->second.get();
  ++(triePtr()->visits);
}

TrieNode* FusionCache::triePtr() const {
  TORCH_INTERNAL_ASSERT(
      trie_ptr_ != nullptr, "The trie node is unexpectedly null.");
  return trie_ptr_;
}

} // namespace nvfuser
