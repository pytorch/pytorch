// unit-test-lock-file.cpp
#include "../../source/core/slang-io.h"
#include "../../source/core/slang-process.h"
#include "unit-test/slang-unit-test.h"

#include <atomic>
#include <future>
#include <thread>
#include <vector>

using namespace Slang;

static const String fileName = Path::simplify(
    Path::getParentDirectory(Path::getExecutablePath()) + "/test_lock_file" +
    String(Process::getId()));

SLANG_UNIT_TEST(lockFileOpenClose)
{
    LockFile file;
    SLANG_CHECK(file.isOpen() == false);
    SLANG_CHECK_ABORT(file.open(fileName) == SLANG_OK);
    SLANG_CHECK(file.isOpen() == true);
    SLANG_CHECK(File::exists(fileName) == true);
    file.close();
    SLANG_CHECK(file.isOpen() == false);

    // Cleanup.
    File::remove(fileName);
    SLANG_CHECK(File::exists(fileName) == false);
}

SLANG_UNIT_TEST(lockFileSync)
{
    // aarch64/linux builds currently fail to run multi-threaded tests within the test-server.
    // Tests work fine without the test-server, which is puzzling. For now we disable them.
#if SLANG_PROCESSOR_ARM_64 || SLANG_LINUX
    SLANG_IGNORE_TEST
#endif

    // Test using multiple threads.
    {
        static std::atomic<uint32_t> lockCounter;
        static std::atomic<uint32_t> unlockCounter;

        struct LockTask
        {
            std::thread thread;
            std::promise<void> startPromise;
            std::future<void> startFuture;
            LockFile lockFile;
            SlangResult openResult = false;
            SlangResult tryLockSharedResult = false;
            SlangResult tryLockExclusiveResult = false;
            SlangResult lockResult = false;
            SlangResult unlockResult = false;
            uint32_t lockIteration = 0;
            uint32_t unlockIteration = 0;

            LockTask()
                : startFuture(startPromise.get_future())
            {
                openResult = lockFile.open(fileName);
            }

            void run()
            {
                tryLockSharedResult = lockFile.tryLock(LockFile::LockType::Shared);
                tryLockExclusiveResult = lockFile.tryLock(LockFile::LockType::Exclusive);
                startPromise.set_value();
                lockResult = lockFile.lock(LockFile::LockType::Exclusive);
                lockIteration = lockCounter.fetch_add(1);
                std::this_thread::sleep_for(std::chrono::milliseconds(1));
                unlockIteration = unlockCounter.fetch_add(1);
                unlockResult = lockFile.unlock();
            }
        };

        // Acquire lock from main thread.
        LockFile lockFile;
        SLANG_CHECK(lockFile.open(fileName) == SLANG_OK);
        SLANG_CHECK(lockFile.lock(LockFile::LockType::Exclusive) == SLANG_OK);

        // Make sure we cannot acquire the lock in non-blocking mode from a second instance.
        LockFile lockFile2;
        SLANG_CHECK(lockFile2.open(fileName) == SLANG_OK);
        SLANG_CHECK(lockFile2.tryLock(LockFile::LockType::Shared) == SLANG_E_TIME_OUT);
        SLANG_CHECK(lockFile2.tryLock(LockFile::LockType::Exclusive) == SLANG_E_TIME_OUT);

        // Start a number of threads and wait for them to start up.
        // Each thread immediately tries to acquire the lock in non-blocking mode (expected to
        // fail). Next each thread acquires the lock in blocking mode.
        std::vector<LockTask> tasks(32);
        for (auto& task : tasks)
        {
            task.thread = std::thread(&LockTask::run, &task);
            task.startFuture.wait();
        }

        // Make sure none of the threads were able to acquire the lock yet.
        std::this_thread::sleep_for(std::chrono::milliseconds(5));
        SLANG_CHECK(lockCounter == 0);

        // Release the lock from the main thread. This will allow all the other
        // threads to acquire the lock, one after the other.
        SLANG_CHECK(lockFile.unlock() == SLANG_OK);

        // Wait for all threads to finish and make sure they behaved as expected.
        std::vector<bool> lockIterationUsed(tasks.size(), false);
        std::vector<bool> unlockIterationUsed(tasks.size(), false);
        for (auto& task : tasks)
        {
            task.thread.join();

            SLANG_CHECK(task.openResult == SLANG_OK);
            SLANG_CHECK(task.tryLockSharedResult == SLANG_E_TIME_OUT);
            SLANG_CHECK(task.tryLockExclusiveResult == SLANG_E_TIME_OUT);
            SLANG_CHECK(task.lockResult == SLANG_OK);
            SLANG_CHECK(task.unlockResult == SLANG_OK);
            SLANG_CHECK(task.lockIteration < lockIterationUsed.size());
            SLANG_CHECK(task.unlockIteration < unlockIterationUsed.size());
            SLANG_CHECK(task.unlockIteration == task.lockIteration);
            SLANG_CHECK(lockIterationUsed[task.lockIteration] == false);
            SLANG_CHECK(unlockIterationUsed[task.unlockIteration] == false);
            lockIterationUsed[task.lockIteration] = true;
            unlockIterationUsed[task.unlockIteration] = true;
        }

        // Ensure all threads did manage to acquire the lock.
        SLANG_CHECK(lockCounter == tasks.size());
        SLANG_CHECK(unlockCounter == tasks.size());

        // Check that we can now acquire the lock in non-blocking mode.
        SLANG_CHECK(lockFile2.tryLock(LockFile::LockType::Exclusive) == SLANG_OK);
        SLANG_CHECK(lockFile2.unlock() == SLANG_OK);
    }

    // Cleanup.
    File::remove(fileName);
    SLANG_CHECK(File::exists(fileName) == false);
}
