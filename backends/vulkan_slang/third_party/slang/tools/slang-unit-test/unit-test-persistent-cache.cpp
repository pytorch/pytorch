// unit-test-persistent-cache.cpp
#include "../../source/core/slang-file-system.h"
#include "../../source/core/slang-io.h"
#include "../../source/core/slang-persistent-cache.h"
#include "../../source/core/slang-process.h"
#include "../../source/core/slang-random-generator.h"
#include "unit-test/slang-unit-test.h"

#include <atomic>
#include <chrono>
#include <condition_variable>
#include <functional>
#include <mutex>
#include <thread>

using namespace Slang;

static DefaultRandomGenerator rng(0xdeadbeef);

inline ComPtr<ISlangBlob> createRandomBlob(size_t size)
{
    ScopedAllocation alloc;
    alloc.allocate(size);
    rng.nextData(alloc.getData(), size);
    return RawBlob::moveCreate(alloc);
}

inline bool isBlobEqual(ISlangBlob* a, ISlangBlob* b)
{
    return a->getBufferSize() == b->getBufferSize() &&
           ::memcmp(a->getBufferPointer(), b->getBufferPointer(), a->getBufferSize()) == 0;
}

class Barrier
{
public:
    Barrier(size_t threadCount, std::function<void()> completionFunc = nullptr)
        : m_threadCount(threadCount), m_waitCount(threadCount), m_completionFunc(completionFunc)
    {
    }

    Barrier(const Barrier& barrier) = delete;
    Barrier& operator=(const Barrier& barrier) = delete;

    void wait()
    {
        std::unique_lock<std::mutex> lock(m_mutex);

        auto generation = m_generation;

        if (--m_waitCount == 0)
        {
            if (m_completionFunc)
                m_completionFunc();
            ++m_generation;
            m_waitCount = m_threadCount;
            m_condition.notify_all();
        }
        else
        {
            m_condition.wait(lock, [this, generation]() { return generation != m_generation; });
        }
    }

private:
    size_t m_threadCount;
    size_t m_waitCount;
    size_t m_generation = 0;
    std::function<void()> m_completionFunc;
    std::mutex m_mutex;
    std::condition_variable m_condition;
};

namespace Slang
{

/// Helper class for performing tests on the persistent cache.
/// This class is a friend class of PersistentCache and can access its internals.
struct PersistentCacheTest
{
    ISlangMutableFileSystem* osFileSystem;
    String cacheDirectory;
    RefPtr<PersistentCache> cache;

    PersistentCacheTest(Count maxEntryCount = 0)
    {
        osFileSystem = OSFileSystem::getMutableSingleton();
        cacheDirectory = Path::simplify(
            Path::getParentDirectory(Path::getExecutablePath()) + "/persistent-cache-test" +
            String(Process::getId()));

        removeCacheFiles();

        PersistentCache::Desc desc;
        desc.directory = cacheDirectory.getBuffer();
        desc.maxEntryCount = maxEntryCount;
        cache = new PersistentCache(desc);
    }

    virtual ~PersistentCacheTest()
    {
        cache = nullptr;

        removeCacheFiles();
    }

    void removeCacheFiles()
    {
        // Remove all files the cache created.
        osFileSystem->enumeratePathContents(
            cacheDirectory.getBuffer(),
            [](SlangPathType pathType, const char* fileName, void* userData)
            {
                PersistentCacheTest* self = static_cast<PersistentCacheTest*>(userData);
                String path = self->cacheDirectory + "/" + fileName;
                self->osFileSystem->remove(path.getBuffer());
            },
            this);

        // Also remove the cache directory.
        osFileSystem->remove(cacheDirectory.getBuffer());
    }

    // Entry (key, data) for testing.
    struct Entry
    {
        PersistentCache::Key key;
        ComPtr<ISlangBlob> data;
    };

    // Helper to write an entry to the cache.
    void writeEntry(const Entry& entry)
    {
        SLANG_CHECK(cache->writeEntry(entry.key, entry.data) == SLANG_OK);
    }

    // Helper to read an entry from the cache and discard the data.
    // Returns true if the entry was found, false otherwise.
    bool readEntry(const Entry& entry)
    {
        ComPtr<ISlangBlob> data;
        SlangResult result = cache->readEntry(entry.key, data.writeRef());
        SLANG_CHECK(result == SLANG_OK || result == SLANG_E_NOT_FOUND);
        if (result == SLANG_OK)
        {
            SLANG_CHECK(isBlobEqual(data, entry.data));
        }
        if (result == SLANG_E_NOT_FOUND)
        {
            SLANG_CHECK(data == nullptr);
        }
        return result == SLANG_OK;
    }

    // Get the absolute filename for a cache entry file.
    String getEntryFileName(const Entry& entry) { return cache->getEntryFileName(entry.key); }

    // Get the absolute filename of the cache index file.
    String getIndexFilename() { return cache->m_indexFileName; }
};

} // namespace Slang

// Performs basic tests on the cache.
// - write/read entries
// - check for correct cache stats
// - clearing the cache
// - resetting stats
struct BasicTest : public PersistentCacheTest
{
    BasicTest()
        : PersistentCacheTest()
    {
    }

    void run()
    {
        // Check that cache is empty.
        SLANG_CHECK(cache->getStats().entryCount == 0);
        SLANG_CHECK(cache->getStats().hitCount == 0);
        SLANG_CHECK(cache->getStats().missCount == 0);

        // Setup a list of entries to store in the cache.
        List<Entry> entries;
        for (size_t i = 0; i < 10; ++i)
        {
            auto data = createRandomBlob(i * 1024);
            auto key = SHA1::compute(data->getBufferPointer(), data->getBufferSize());
            entries.add(Entry{key, data});
        }

        for (size_t i = 0; i < 10; ++i)
        {
            const auto& entry = entries[i];
            ComPtr<ISlangBlob> data;

            // Try to read an entry. Check that its not found and counts as a miss.
            SLANG_CHECK(cache->readEntry(entry.key, data.writeRef()) == SLANG_E_NOT_FOUND);
            SLANG_CHECK(cache->getStats().missCount == i + 1);

            // Write the entry. Check that it gets added.
            SLANG_CHECK(cache->writeEntry(entry.key, entry.data) == SLANG_OK);
            SLANG_CHECK(cache->getStats().entryCount == i + 1);
        }

        SLANG_CHECK(cache->getStats().entryCount == 10);
        SLANG_CHECK(cache->getStats().hitCount == 0);
        SLANG_CHECK(cache->getStats().missCount == 10);

        for (size_t i = 0; i < 10; ++i)
        {
            const auto& entry = entries[i];
            ComPtr<ISlangBlob> data;

            // Read entries. Check that these are cache hits and return the correct data.
            SLANG_CHECK(cache->readEntry(entry.key, data.writeRef()) == SLANG_OK);
            SLANG_CHECK(cache->getStats().hitCount == i + 1);
            SLANG_CHECK(isBlobEqual(data, entry.data));
        }

        SLANG_CHECK(cache->getStats().entryCount == 10);
        SLANG_CHECK(cache->getStats().hitCount == 10);
        SLANG_CHECK(cache->getStats().missCount == 10);

        // Clear the cache. Check that entry count is reset.
        SLANG_CHECK(cache->clear() == SLANG_OK);
        SLANG_CHECK(cache->getStats().entryCount == 0);
        SLANG_CHECK(cache->getStats().hitCount == 10);
        SLANG_CHECK(cache->getStats().missCount == 10);

        // Reset stats.
        cache->resetStats();
        SLANG_CHECK(cache->getStats().entryCount == 0);
        SLANG_CHECK(cache->getStats().hitCount == 0);
        SLANG_CHECK(cache->getStats().missCount == 0);

        // Check that cache is empty.
        for (size_t i = 0; i < 10; ++i)
        {
            const auto& entry = entries[i];
            ComPtr<ISlangBlob> data;
            SLANG_CHECK(cache->readEntry(entry.key, data.writeRef()) == SLANG_E_NOT_FOUND);
        }
        SLANG_CHECK(cache->getStats().missCount == 10);
    }
};

// Tests the least-recently-used cache eviction policy.
struct EvictionTest : public PersistentCacheTest
{
    EvictionTest()
        : PersistentCacheTest(3)
    {
    }

    void run()
    {
        // Setup a list of entries to store in the cache.
        List<Entry> entries;
        for (size_t i = 0; i < 10; ++i)
        {
            auto data = createRandomBlob(4096);
            auto key = SHA1::compute(data->getBufferPointer(), data->getBufferSize());
            entries.add(Entry{key, data});
        }

        writeEntry(entries[0]);
        writeEntry(entries[1]);
        writeEntry(entries[2]);

        SLANG_CHECK(readEntry(entries[0]) == true);
        SLANG_CHECK(readEntry(entries[1]) == true);
        SLANG_CHECK(readEntry(entries[2]) == true);

        // Evict LRU entry 0.
        writeEntry(entries[3]);
        SLANG_CHECK(readEntry(entries[0]) == false);
        SLANG_CHECK(readEntry(entries[1]) == true);
        SLANG_CHECK(readEntry(entries[2]) == true);
        SLANG_CHECK(readEntry(entries[3]) == true);

        // Evict LRU entry 1.
        writeEntry(entries[4]);
        SLANG_CHECK(readEntry(entries[1]) == false);
        SLANG_CHECK(readEntry(entries[2]) == true);
        SLANG_CHECK(readEntry(entries[3]) == true);
        SLANG_CHECK(readEntry(entries[4]) == true);

        // Evict LRU entry 2.
        writeEntry(entries[5]);
        SLANG_CHECK(readEntry(entries[2]) == false);
        SLANG_CHECK(readEntry(entries[3]) == true);
        SLANG_CHECK(readEntry(entries[4]) == true);
        SLANG_CHECK(readEntry(entries[5]) == true);

        // Evict LRU entry 4.
        SLANG_CHECK(readEntry(entries[3]) == true);
        writeEntry(entries[6]);
        SLANG_CHECK(readEntry(entries[3]) == true);
        SLANG_CHECK(readEntry(entries[4]) == false);
        SLANG_CHECK(readEntry(entries[5]) == true);
        SLANG_CHECK(readEntry(entries[6]) == true);
    }
};


// Tests the cache to be robust against various corruptions.
// These can happen if the cache files are manipulated externally.
// The cache might also be corrupted if the application is terminated while writing.
struct CorruptionTest : public PersistentCacheTest
{
    List<Entry> entries;

    template<typename Func>
    void testIndexCorruption(Func func, SlangResult expectedReadResult)
    {
        writeEntry(entries[0]);
        SLANG_CHECK(readEntry(entries[0]) == true);
        func();
        // We expect a SLANG_E_NOT_FOUND because the cache has an empty index now.
        ComPtr<ISlangBlob> data;
        SLANG_CHECK(cache->readEntry(entries[0].key, data.writeRef()) == expectedReadResult);

        writeEntry(entries[0]);
        SLANG_CHECK(readEntry(entries[0]) == true);
        func();
        writeEntry(entries[0]);
        SLANG_CHECK(readEntry(entries[0]) == true);
    }

    void run()
    {
        // Setup a list of entries to store in the cache.
        for (size_t i = 0; i < 10; ++i)
        {
            auto data = createRandomBlob(4096);
            auto key = SHA1::compute(data->getBufferPointer(), data->getBufferSize());
            entries.add(Entry{key, data});
        }

        // Test behavior when a cached entry file is removed externally before reading.
        writeEntry(entries[0]);
        SLANG_CHECK(readEntry(entries[0]) == true);
        osFileSystem->remove(getEntryFileName(entries[0]).getBuffer());
        ComPtr<ISlangBlob> data;
        // First time we read the entry, we expect a SLANG_E_CANNOT_OPEN because the file is gone.
        SLANG_CHECK(cache->readEntry(entries[0].key, data.writeRef()) == SLANG_E_CANNOT_OPEN);
        // The next time we read the entry, we expect a SLANG_E_NOT_FOUND because the entry has
        // been removed from the cache index.
        SLANG_CHECK(cache->readEntry(entries[0].key, data.writeRef()) == SLANG_E_NOT_FOUND);

        // Test behavior when a cached entry file is removed externally before writing.
        writeEntry(entries[0]);
        SLANG_CHECK(readEntry(entries[0]) == true);
        osFileSystem->remove(getEntryFileName(entries[0]).getBuffer());
        writeEntry(entries[0]);
        SLANG_CHECK(readEntry(entries[0]) == true);

        // Test behavior when the index file is removed before reading.
        writeEntry(entries[0]);
        SLANG_CHECK(readEntry(entries[0]) == true);
        osFileSystem->remove(getIndexFilename().getBuffer());
        // We expect a SLANG_E_NOT_FOUND because the cache has an empty index now.
        SLANG_CHECK(cache->readEntry(entries[0].key, data.writeRef()) == SLANG_E_NOT_FOUND);

        // Test behavior when the index file is removed before writing.
        writeEntry(entries[0]);
        SLANG_CHECK(readEntry(entries[0]) == true);
        osFileSystem->remove(getIndexFilename().getBuffer());
        writeEntry(entries[1]);
        SLANG_CHECK(readEntry(entries[1]) == true);

        // Test different corruptions of the index file.
        testIndexCorruption(
            [this]() { osFileSystem->remove(getIndexFilename().getBuffer()); },
            SLANG_E_NOT_FOUND);

        testIndexCorruption(
            [this]()
            {
                FileStream fs;
                fs.init(
                    getIndexFilename(),
                    FileMode::Open,
                    FileAccess::ReadWrite,
                    FileShare::ReadWrite);
                fs.write("x", 1);
            },
            SLANG_E_INTERNAL_FAIL);

        testIndexCorruption(
            [this]()
            {
                FileStream fs;
                fs.init(
                    getIndexFilename(),
                    FileMode::Open,
                    FileAccess::ReadWrite,
                    FileShare::ReadWrite);
                fs.seek(SeekOrigin::Start, 4);
                uint32_t version = 0xffffffff;
                fs.write(&version, sizeof(version));
            },
            SLANG_E_INTERNAL_FAIL);

        testIndexCorruption(
            [this]()
            {
                FileStream fs;
                fs.init(
                    getIndexFilename(),
                    FileMode::Open,
                    FileAccess::ReadWrite,
                    FileShare::ReadWrite);
                fs.seek(SeekOrigin::Start, 8);
                uint32_t count = 0x7fffffff;
                fs.write(&count, sizeof(count));
            },
            SLANG_E_INTERNAL_FAIL);

        testIndexCorruption(
            [this]()
            {
                FileStream fs;
                fs.init(
                    getIndexFilename(),
                    FileMode::Open,
                    FileAccess::ReadWrite,
                    FileShare::ReadWrite);
                fs.seek(SeekOrigin::Start, 8);
                uint32_t count = 0;
                fs.write(&count, sizeof(count));
            },
            SLANG_E_INTERNAL_FAIL);

        testIndexCorruption(
            [this]()
            {
                FileStream fs;
                fs.init(
                    getIndexFilename(),
                    FileMode::Open,
                    FileAccess::ReadWrite,
                    FileShare::ReadWrite);
                fs.seek(SeekOrigin::End, 0);
                fs.write("x", 1);
            },
            SLANG_E_INTERNAL_FAIL);
    }
};

#undef ENABLE_LOGGING
#undef ENABLE_WRITE_TEST

#ifdef ENABLE_LOGGING
#define LOG(fmt, ...)           \
    printf(fmt, ##__VA_ARGS__); \
    fflush(stdout);
#else
#define LOG(fmt, ...)
#endif

// Stress testing.
// This test spawns a number of threads to do concurrent access to the cache.
// For now this is fairly simple:
// - spawn a number of threads
// - write random entries to the cache concurrenctly (slightly oversubscribe)
// - synchronize
// - read entries from the cache concurretly (test that we get the expected number of hits/misses)
// - synchronize
// - repeat for a number of iterations
struct StressTest : public PersistentCacheTest
{
    // Number of entries to write/read per iteration.
    static const uint32_t kEntryCount = 100;
    // Number of entries the cache is short for storing one iteration.
    static const uint32_t kEntryShortageCount = 10;
    // Number of parallel threads to write/read.
    static const uint32_t kThreadCount = 4;
    // Number of entries to write/read per thread per iteration.
    static const uint32_t kBatchCount = kEntryCount / kThreadCount;
    // Total number of iterations.
    static const uint32_t kIterationCount = 4;

    static_assert(kEntryCount % kThreadCount == 0, "kEntryCount must be divisible by kThreadCount");

    List<Entry> entries;

    std::atomic<uint32_t> iteration{0};
    std::atomic<uint32_t> entriesWritten{0};
    std::atomic<uint32_t> bytesWritten{0};
    std::atomic<uint32_t> entriesRead{0};
    std::atomic<uint32_t> bytesRead{0};
    std::atomic<uint32_t> readSuccess{0};
    std::thread threads[kThreadCount];

    Barrier* read_barrier;
    Barrier* write_barrier;

    StressTest()
        : PersistentCacheTest(kEntryCount - kEntryShortageCount)
    {
    }

    void run()
    {
        // Setup a list of entries to store in the cache.
        for (size_t i = 0; i < kEntryCount * 2; ++i)
        {
            size_t size = rng.nextInt32InRange(256, 64 * 1024);
            auto data = createRandomBlob(size);
            auto key = SHA1::compute(data->getBufferPointer(), data->getBufferSize());
            entries.add(Entry{key, data});
        }

        auto startTime = std::chrono::high_resolution_clock::now();

        Barrier read_barrier_(kThreadCount, []() { LOG("Read synchronized\n"); });
        Barrier write_barrier_(
            kThreadCount,
            [this]()
            {
                LOG("Write synchronized\n");
#ifndef ENABLE_WRITE_TEST
                SLANG_CHECK(readSuccess == kEntryCount - kEntryShortageCount);
                readSuccess.store(0);
#endif
                iteration += 1;
            });

        read_barrier = &read_barrier_;
        write_barrier = &write_barrier_;

        for (uint32_t threadIndex = 0; threadIndex < kThreadCount; ++threadIndex)
        {
            threads[threadIndex] = std::thread(
                [this, threadIndex]()
                {
                    LOG("Thread %u: starting\n", threadIndex);

                    while (true)
                    {
                        // Write to cache.
                        size_t startIndex =
                            (iteration * kEntryCount + (threadIndex * kBatchCount)) %
                            (kEntryCount * 2);
                        for (size_t i = 0; i < kBatchCount; ++i)
                        {
                            const Entry& entry = entries[startIndex + i];
#ifdef ENABLE_WRITE_TEST
                            osFileSystem->saveFileBlob(
                                getEntryFileName(entry).getBuffer(),
                                entry.data);
#else
                            writeEntry(entry);
#endif
                            entriesWritten.fetch_add(1);
                            bytesWritten.fetch_add((uint32_t)entry.data->getBufferSize());
                        }

                        LOG("Thread %u: ended writing (iteration=%u)\n",
                            threadIndex,
                            iteration.load());

                        // Synchronize.
                        read_barrier->wait();

                        // Read from cache.
                        for (size_t i = 0; i < kBatchCount; ++i)
                        {
                            const Entry& entry = entries[startIndex + i];
#ifndef ENABLE_WRITE_TEST
                            if (readEntry(entry))
                            {
                                readSuccess.fetch_add(1);
                                bytesRead.fetch_add((uint32_t)entry.data->getBufferSize());
                            }
#endif
                            entriesRead.fetch_add(1);
                        }

                        LOG("Thread %u: ended reading (iteration=%u)\n",
                            threadIndex,
                            iteration.load());

                        // Synchronize.
                        write_barrier->wait();

                        // Terminate.
                        if (iteration >= kIterationCount)
                        {
                            LOG("Thread %u: terminates\n", threadIndex);
                            return;
                        }
                    }
                });
        }

        for (auto& thread : threads)
        {
            thread.join();
        }

        auto endTime = std::chrono::high_resolution_clock::now();
        auto duration = endTime - startTime;
        auto seconds =
            std::chrono::duration_cast<std::chrono::milliseconds>(duration).count() / 1000.0;

        LOG("Total time: %.3fs\n", seconds);
        LOG("Total bytes written: %d\n", bytesWritten.load());
        LOG("Write througput: %.3fMB/s\n", (bytesWritten.load() / (1024.0 * 1024.0)) / seconds);
        LOG("Total bytes read: %d\n", bytesRead.load());
    }
};

SLANG_UNIT_TEST(persistentCacheBasic)
{
    BasicTest test;
    test.run();
}

SLANG_UNIT_TEST(persistentCacheEviction)
{
    EvictionTest test;
    test.run();
}

SLANG_UNIT_TEST(persistentCacheCorruption)
{
    CorruptionTest test;
    test.run();
}

SLANG_UNIT_TEST(persistentCacheStress)
{
    // aarch64 builds currently fail to run multi-threaded tests within the test-server.
    // Tests work fine without the test-server, which is puzzling. For now we disable them.
#if SLANG_PROCESSOR_ARM_64 || SLANG_LINUX_FAMILY
    SLANG_IGNORE_TEST
#endif
    StressTest test;
    test.run();
}
