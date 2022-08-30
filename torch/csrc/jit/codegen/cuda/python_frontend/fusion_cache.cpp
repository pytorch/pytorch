#include <torch/csrc/jit/codegen/cuda/python_frontend/fusion_cache.h>
#include <torch/csrc/jit/codegen/cuda/python_frontend/fusion_record.h>
#include <mutex>

namespace nvfuser {

static std::mutex fusion_cache_lock;
FusionCache* FusionCache::singleton_ = nullptr;

FusionCacheEntry::FusionCacheEntry(
    RecordFunctor* rec,
    bool _is_terminal,
    size_t _fusion_id)
    : record(rec),
      record_hash_map(),
      is_terminal(_is_terminal),
      fusion_id(_fusion_id) {}

FusionCache* FusionCache::get(size_t max_fusions) {
  std::lock_guard<std::mutex> guard(fusion_cache_lock);
  if (singleton_ == nullptr) {
    singleton_ = new FusionCache(max_fusions);
  }
  return singleton_;
}

size_t FusionCache::numFusions() const {
  return fusions_.size();
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
      !fusionCachePtr()->is_terminal,
      "There should be no children from a Terminal Cache Entry!");
  TORCH_CHECK(rec, "Record is null!");
  auto cache_entry = fusionCachePtr()->record_hash_map.find(rec);
  if (cache_entry == std::end(fusionCachePtr()->record_hash_map)) {
    return c10::nullopt;
  } else {
    return c10::optional<FusionCacheEntry*>(cache_entry->second.get());
  }
}
void FusionCache::createFusionCacheEntry(RecordFunctor* rec) {
  TORCH_CHECK(
      !fusionCachePtr()->is_terminal,
      "Cannot create a cache entryfrom a terminal entry!");
  TORCH_CHECK(rec, "Record is null!");

  // Copying the record owned by the FusionDefinition that calls this function
  // so the cache owns a copy when the FusionDefinition gets destroyed rather
  // than managing a shared pointer that would  only share with
  // FusionDefinition that creates a cache entry but not cache lookups
  RecordFunctor* new_rec = rec->clone();
  fusionCachePtr()->record_hash_map[new_rec] =
      std::make_unique<FusionCacheEntry>(new_rec);
  if (Nvf::isDebugDumpEnabled(Nvf::DebugDumpOption::PythonFrontendDebug)) {
    std::stringstream ss;
    new_rec->print(ss);
    std::cout << "\nFusionDefinition: Create new cache entry for: " << ss.str()
              << "\n";
  }
}
size_t FusionCache::createTerminalFusionCacheEntry(RecordFunctor* rec) {
  TORCH_CHECK(
      !fusionCachePtr()->is_terminal,
      "Cannot create a cache entry from a terminal entry!");
  TORCH_CHECK(rec, "Record is null!");
  TORCH_CHECK(
      rec->recordType() == RecordType::End,
      "A Terminal Cache Entry can only be created with an EndRecord!");
  TORCH_CHECK(
      (fusions_.size() + 1) <= max_fusions_,
      "The number of fusions in nvfuser has exceeded ",
      max_fusions_,
      "fusions.  The max_fusions for the FusionCache might need to be ",
      "increased if the max number is not being exceeded due to an error.");

  // Copying the record owned by the FusionDefinition that calls this function
  // so the cache owns a copy when the FusionDefinition gets destroyed rather
  // than managing a shared pointer that would  only share with
  // FusionDefinition that creates a cache entry but not cache lookups
  RecordFunctor* new_rec = rec->clone();
  fusions_.push_back(std::make_unique<Nvf::FusionExecutorCache>(
      std::make_unique<Nvf::Fusion>()));
  auto fusion_id = fusions_.size() - 1;
  fusionCachePtr()->record_hash_map[new_rec] =
      std::make_unique<FusionCacheEntry>(new_rec, true, fusion_id);
  if (Nvf::isDebugDumpEnabled(Nvf::DebugDumpOption::PythonFrontendDebug)) {
    std::stringstream ss;
    new_rec->print(ss);
    std::cout << "\nFusionDefinition: Create new terminal cache entry.\n";
  }
  return fusion_id;
}
void FusionCache::resetFusionCachePtr() {
  fusion_cache_ptr_ = fusion_cache_start_.get();
  TORCH_CHECK(fusionCachePtr()->record->recordType() == RecordType::Start);
}
void FusionCache::traverseFusionCache(RecordFunctor* rec) {
  TORCH_CHECK(
      !fusionCachePtr()->is_terminal,
      "Cannot traverse cache from a terminal entry!");
  auto cache_entry = fusionCachePtr()->record_hash_map.find(rec);
  TORCH_CHECK(
      cache_entry != std::end(fusionCachePtr()->record_hash_map),
      "Cache Entry for Cache Traverse is not found!");
  TORCH_CHECK(cache_entry->second, "Record in Cache Entry is null!");
  fusion_cache_ptr_ = cache_entry->second.get();
}

FusionCacheEntry* FusionCache::fusionCachePtr() const {
  TORCH_INTERNAL_ASSERT(
      fusion_cache_ptr_ != nullptr,
      "The fusion cache entry is unexpectedly null.");
  return fusion_cache_ptr_;
}

} // namespace nvfuser
