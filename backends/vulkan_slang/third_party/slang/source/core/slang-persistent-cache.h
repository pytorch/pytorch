#pragma once
#include "../core/slang-crypto.h"
#include "../core/slang-io.h"
#include "../core/slang-string.h"
#include "slang.h"

#include <mutex>

namespace Slang
{

/// Implements a simple persistent cache on the filesystem for storing key/value pairs.
/// Keys are SHA1 hashes and values are arbitrary blobs of data.
/// The cache is save for concurrent access from multiple threads/processes by using
/// a lock file within the cache directory. Furthermore, the cache implements a LRU
/// eviction policy.
class PersistentCache : public RefObject
{
public:
    struct Desc
    {
        // The root directory for the cache.
        const char* directory = nullptr;
        // The maximum number of entries stored in the cache. By default, there is no limit.
        Count maxEntryCount = 0;
    };

    struct Stats
    {
        // Number of cache hits since last resetting the stats.
        Count hitCount;
        // Number of cache misses since last resetting the stats.
        Count missCount;
        // Current number of entries in the cache.
        Count entryCount;
    };

    using Key = SHA1::Digest;

    PersistentCache(const Desc& desc);
    ~PersistentCache();

    /// Clear the contents of the cache by removing the cache index and all entry files.
    SlangResult clear();

    const Stats& getStats() const { return m_stats; }
    void resetStats();

    /// Read an entry from the cache.
    /// Returns SLANG_OK if successful, SLANG_E_NOT_FOUND if the entry is not in the cache.
    SlangResult readEntry(const Key& key, ISlangBlob** outData);

    /// Write an entry to the cache.
    /// Returns SLANG_OK if successful.
    SlangResult writeEntry(const Key& key, ISlangBlob* data);

private:
    struct CacheEntry
    {
        Key key;
        uint32_t age;
    };

    using CacheIndex = List<CacheEntry>;

    SlangResult initialize();

    String getEntryFileName(const Key& key);

    SlangResult readIndex(const String& fileName, CacheIndex& outIndex);
    SlangResult writeIndex(const String& fileName, const CacheIndex& index);

    String m_cacheDirectory;
    String m_lockFileName;
    String m_indexFileName;

    // For exclusive locking we need both a mutex (acquired first)
    // followed by a a file lock. The mutex is needed because on Linux
    // the file lock is only locking between processes, not threads.
    std::mutex m_mutex;
    Slang::LockFile m_lockFile;

    Count m_maxEntryCount;

    Stats m_stats;

    // Used for unit tests.
    friend struct PersistentCacheTest;
};

} // namespace Slang
