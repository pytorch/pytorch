#include <torch/csrc/jit/codegen/cuda/python_frontend/fusion_cache.h>
#include <torch/csrc/jit/codegen/cuda/python_frontend/fusion_record.h>
#include <mutex>

namespace nvfuser {

static std::mutex fusion_cache_lock;
FusionCache* FusionCache::singleton_ = nullptr;

FusionCacheEntry::FusionCacheEntry(RecordFunctor* rec, size_t _fusion_id)
    : record(rec), record_hash_map(), fusion_id(_fusion_id), visits(0) {}

bool FusionCacheEntry::isTerminal() const {
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
    for (size_t i = 0; i < terminal_cache_entries_.size(); ++i) {
      // The first visit is a miss!
      auto visits = terminal_cache_entries_[i]->visits - 1;
      total_cache_hits += visits;
      os << "\t" << i << " -> " << visits << " hits\n";
    }

    auto hit_rate = static_cast<float>(total_cache_hits) /
        static_cast<float>(fusion_cache_start_->visits) * 100.0;
    os << "Cache Lookups: " << fusion_cache_start_->visits;
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
      fusion_cache_start_(nullptr),
      fusion_cache_ptr_(nullptr),
      fusions_() {
  RecordFunctor* start = new StartRecord();
  fusion_cache_start_ = std::make_unique<FusionCacheEntry>(start);
  fusion_cache_ptr_ = fusion_cache_start_.get();
}

c10::optional<FusionCacheEntry*> FusionCache::lookupFusionCacheEntry(
    RecordFunctor* rec) const {
  TORCH_CHECK(
      !fusionCachePtr()->isTerminal(),
      "There should be no children from a Terminal Cache Entry!");
  TORCH_CHECK(rec, "Record is null!");
  auto cache_entry = fusionCachePtr()->record_hash_map.find(rec);
  if (cache_entry == std::end(fusionCachePtr()->record_hash_map)) {
    return c10::nullopt;
  } else {
    return c10::optional<FusionCacheEntry*>(cache_entry->second.get());
  }
}

c10::optional<size_t> FusionCache::createFusionCacheEntry(RecordFunctor* rec) {
  c10::optional<size_t> result = c10::nullopt;
  TORCH_CHECK(
      !fusionCachePtr()->isTerminal(),
      "Cannot create a cache entry from a terminal entry!");
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
  // so the cache owns a copy when the FusionDefinition gets destroyed rather
  // than managing a shared pointer that would  only share with
  // FusionDefinition that creates a cache entry but not cache lookups
  RecordFunctor* new_rec = rec->clone();
  fusionCachePtr()->record_hash_map[new_rec] =
      std::make_unique<FusionCacheEntry>(new_rec, fusion_id);
  if (rec->recordType() == RecordType::End) {
    terminal_cache_entries_.push_back(
        fusionCachePtr()->record_hash_map[new_rec].get());
  }
  if (Nvf::isDebugDumpEnabled(Nvf::DebugDumpOption::PythonFrontendDebug)) {
    std::stringstream ss;
    new_rec->print(ss);
    std::cout << "\nFusionDefinition: Create new cache entry for: " << ss.str()
              << "\n";
  }
  return result;
}

void FusionCache::resetFusionCachePtr() {
  fusion_cache_ptr_ = fusion_cache_start_.get();
  TORCH_CHECK(fusionCachePtr()->record->recordType() == RecordType::Start);
  ++(fusionCachePtr()->visits);
}

void FusionCache::traverseFusionCache(RecordFunctor* rec) {
  TORCH_CHECK(
      !fusionCachePtr()->isTerminal(),
      "Cannot traverse cache from a terminal entry!");
  auto cache_entry = fusionCachePtr()->record_hash_map.find(rec);
  TORCH_CHECK(
      cache_entry != std::end(fusionCachePtr()->record_hash_map),
      "Cache Entry for Cache Traverse is not found!");
  TORCH_CHECK(cache_entry->second, "Record in Cache Entry is null!");
  fusion_cache_ptr_ = cache_entry->second.get();
  ++(fusionCachePtr()->visits);
}

FusionCacheEntry* FusionCache::fusionCachePtr() const {
  TORCH_INTERNAL_ASSERT(
      fusion_cache_ptr_ != nullptr,
      "The fusion cache entry is unexpectedly null.");
  return fusion_cache_ptr_;
}

} // namespace nvfuser
