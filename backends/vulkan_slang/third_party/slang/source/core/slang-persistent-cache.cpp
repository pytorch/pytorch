#include "slang-persistent-cache.h"

#include "../core/slang-blob.h"
#include "../core/slang-io.h"
#include "../core/slang-stream.h"
#include "../core/slang-string-util.h"

namespace Slang
{

PersistentCache::PersistentCache(const Desc& desc)
{
    m_cacheDirectory = Path::simplify(desc.directory);
    Path::createDirectory(m_cacheDirectory);

    m_lockFileName = Path::simplify(m_cacheDirectory + "/lock");
    m_indexFileName = Path::simplify(m_cacheDirectory + "/index");

    m_lockFile.open(m_lockFileName);

    m_maxEntryCount = desc.maxEntryCount;

    resetStats();

    initialize();
}

PersistentCache::~PersistentCache() {}

SlangResult PersistentCache::clear()
{
    if (!m_lockFile.isOpen())
    {
        return SLANG_E_CANNOT_OPEN;
    }

    // Acquire the exclusive lock.
    std::lock_guard<std::mutex> mutexLock(m_mutex);
    LockFileGuard fileLock(m_lockFile);

    struct Visitor : Path::Visitor
    {
        const String& directory;
        const String& lockFileName;

        Visitor(const String& directory, const String& lockFileName)
            : directory(directory), lockFileName(lockFileName)
        {
        }

        void accept(Path::Type type, const UnownedStringSlice& fileName) SLANG_OVERRIDE
        {
            String fullPath = Path::simplify(directory + "/" + fileName);
            ;
            if (type == Path::Type::File && lockFileName != fullPath)
            {
                Path::remove(fullPath);
            }
        }
    };

    Visitor visitor(m_cacheDirectory, m_lockFileName);
    Path::find(m_cacheDirectory, nullptr, &visitor);

    m_stats.entryCount = 0;

    return SLANG_OK;
}

void PersistentCache::resetStats()
{
    m_stats.entryCount = 0;
    m_stats.hitCount = 0;
    m_stats.missCount = 0;
}

SlangResult PersistentCache::readEntry(const Key& key, ISlangBlob** outData)
{
    // Be pessimistic and assume we have a cache miss.
    ++m_stats.missCount;

    if (!m_lockFile.isOpen())
    {
        return SLANG_E_CANNOT_OPEN;
    }

    // Acquire the exclusive lock.
    std::lock_guard<std::mutex> mutexLock(m_mutex);
    LockFileGuard fileLock(m_lockFile);

    // Return if index does not exist.
    if (!File::exists(m_indexFileName))
    {
        return SLANG_E_NOT_FOUND;
    }

    // Read the cache index.
    CacheIndex cacheIndex;
    SLANG_RETURN_ON_FAIL(readIndex(m_indexFileName, cacheIndex));

    // Increase the age of all entries in the cache.
    for (auto& entry : cacheIndex)
    {
        ++entry.age;
    }

    // Find the entry.
    Index entryIndex =
        cacheIndex.findFirstIndex([&key](const CacheEntry& entry) { return entry.key == key; });
    if (entryIndex == -1)
    {
        return SLANG_E_NOT_FOUND;
    }

    // Read the entry.
    String entryFileName = getEntryFileName(key);
    ScopedAllocation data;
    SlangResult result = File::readAllBytes(entryFileName, data);
    if (result == SLANG_OK)
    {
        --m_stats.missCount;
        ++m_stats.hitCount;
        cacheIndex[entryIndex].age = 0;
        auto blob = RawBlob::moveCreate(data);
        *outData = blob.detach();
    }
    else
    {
        cacheIndex.removeAt(entryIndex);
    }

    // Write the cache index.
    SLANG_RETURN_ON_FAIL(writeIndex(m_indexFileName, cacheIndex));
    m_stats.entryCount = (Count)cacheIndex.getCount();

    return result;
}

SlangResult PersistentCache::writeEntry(const Key& key, ISlangBlob* data)
{
    SLANG_ASSERT(data);

    if (!m_lockFile.isOpen())
    {
        return SLANG_E_CANNOT_OPEN;
    }

    // Acquire the exclusive lock.
    std::lock_guard<std::mutex> mutexLock(m_mutex);
    LockFileGuard fileLock(m_lockFile);

    // Read the cache index.
    // We ignore any errors when reading the index and just write a new one.
    CacheIndex cacheIndex;
    readIndex(m_indexFileName, cacheIndex);

    // Increase the age of all entries in the cache and get the index of
    // the oldest entry.
    Index oldestEntryIndex = -1;
    uint32_t oldestEntryAge = 0;
    for (Index entryIndex = 0; entryIndex < cacheIndex.getCount(); ++entryIndex)
    {
        auto& entry = cacheIndex[entryIndex];
        ++entry.age;
        if (entry.age > oldestEntryAge)
        {
            oldestEntryIndex = entryIndex;
            oldestEntryAge = entry.age;
        }
    }

    // Write the cache entry.
    String entryFileName = getEntryFileName(key);
    SLANG_RETURN_ON_FAIL(
        File::writeAllBytes(entryFileName, data->getBufferPointer(), data->getBufferSize()));

    // Update the index.
    if (m_maxEntryCount > 0 && cacheIndex.getCount() >= m_maxEntryCount)
    {
        // Replace oldest entry.
        SLANG_ASSERT(oldestEntryIndex >= 0);
        File::remove(getEntryFileName(cacheIndex[oldestEntryIndex].key));
        cacheIndex[oldestEntryIndex] = CacheEntry{key, 0};
    }
    else
    {
        // Add new entry.
        cacheIndex.add(CacheEntry{key, 0});
    }

    // Write the cache index.
    SlangResult result = writeIndex(m_indexFileName, cacheIndex);
    if (result == SLANG_OK)
    {
        m_stats.entryCount = (Count)cacheIndex.getCount();
    }
    else
    {
        // If writing the index failed, remove the entry file to avoid growing the cache.
        Path::remove(entryFileName);
    }

    return result;
}

SlangResult PersistentCache::initialize()
{
    if (!m_lockFile.isOpen())
    {
        return SLANG_E_CANNOT_OPEN;
    }

    // Acquire the exclusive lock.
    std::lock_guard<std::mutex> mutexLock(m_mutex);
    LockFileGuard fileLock(m_lockFile);

    CacheIndex cacheIndex;
    if (SLANG_SUCCEEDED(readIndex(m_indexFileName, cacheIndex)))
    {
        m_stats.entryCount = (Count)cacheIndex.getCount();
    }

    return SLANG_OK;
}

String PersistentCache::getEntryFileName(const Key& key)
{
    StringBuilder str;
    str << m_cacheDirectory << "/" << key.toString();
    return str;
}

struct CacheIndexHeader
{
    char magic[4];
    uint32_t version;
    uint32_t count;
    uint32_t reserved;
};

static const char* kMagic = "SLS$";
static const uint32_t kVersion = 1;

SlangResult PersistentCache::readIndex(const String& fileName, CacheIndex& outIndex)
{
    FileStream fs;
    SLANG_RETURN_ON_FAIL(fs.init(fileName, FileMode::Open));

    // Get file size.
    SLANG_RETURN_ON_FAIL(fs.seek(SeekOrigin::End, 0));
    size_t fileSize = (size_t)fs.getPosition();
    SLANG_RETURN_ON_FAIL(fs.seek(SeekOrigin::Start, 0));

    CacheIndexHeader header;
    SLANG_RETURN_ON_FAIL(fs.readExactly(&header, sizeof(header)));
    if (::memcmp(header.magic, kMagic, 4) != 0 || header.version != kVersion)
    {
        return SLANG_E_INTERNAL_FAIL;
    }

    // Return if payload does not have the right size.
    if (header.count * sizeof(CacheEntry) != fileSize - sizeof(header))
    {
        return SLANG_E_INTERNAL_FAIL;
    }

    outIndex.setCount(header.count);
    SLANG_RETURN_ON_FAIL(fs.readExactly(outIndex.getBuffer(), header.count * sizeof(CacheEntry)));

    return SLANG_OK;
}

SlangResult PersistentCache::writeIndex(const String& fileName, const CacheIndex& index)
{
    FileStream fs;
    SLANG_RETURN_ON_FAIL(fs.init(fileName, FileMode::Create));

    CacheIndexHeader header;
    ::memcpy(header.magic, kMagic, 4);
    header.version = kVersion;
    header.count = (uint32_t)index.getCount();
    header.reserved = 0;
    SLANG_RETURN_ON_FAIL(fs.write(&header, sizeof(header)));

    SLANG_RETURN_ON_FAIL(fs.write(index.getBuffer(), index.getCount() * sizeof(CacheEntry)));

    return SLANG_OK;
}

} // namespace Slang
