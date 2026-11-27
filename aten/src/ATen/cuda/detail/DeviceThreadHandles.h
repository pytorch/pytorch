// Some stateful GPU libraries, such as cuDNN, cuBLAS, use handles to store states.
// These handles are tied to device, and these libraries requires/recommends not to
// share handles across host threads.
//
// These libraries recommend using one handle per host thread. We may not want to do
// this because threads are relatively light-weight, but creating and destroying
// handles is expensive (destroying the handle causes synchronizations). DataParallel,
// for example, creates new threads for each forward pass.
//
// This file implements a handle pool mechanism. The handle pool returns handles on
// demand as threads request them. If all existing handles in the pool are in use,
// it creates a new one. As threads terminate, they release handles back into the pool.
// In this way, the handle pool never creates more handles than the high-water mark of
// active threads, so it's efficient with DataParallel. On ROCm, handles reserved per
// (device, stream) are reused only on their stream, so a thread creates one for a
// stream even while handles returned for other streams sit idle, and the bound
// applies per stream (see released_stream_handles).

#pragma once

#include <map>
#include <unordered_map>
#include <vector>
#include <utility>
#include <mutex>
#include <memory>

#include <c10/util/Exception.h>

#ifdef USE_ROCM
#include <hip/hip_runtime_api.h>
#endif

#ifdef USE_ROCM
namespace at::cuda::detail {
// Outside the template and its anonymous namespace, so that one warning covers
// every pool that shares this definition: every pool in a binary or shared
// library. A library built with hidden symbols, such as torch_hip, keeps its own
// copy, so a binary that links it dynamically warns separately for pools of its
// own.
inline void warnPerThreadStreamHandleKept() {
    TORCH_WARN_ONCE(
        "hipStreamPerThread is a different stream in each thread, so a BLAS handle "
        "reserved for it is never reused by another thread: each thread that uses it "
        "keeps the handle and the device memory the handle holds until the process "
        "exits, and workspaces cached for the handle until the workspace cache is "
        "cleared. Use a pool stream or a torch.cuda.Stream() instead.");
}
} // namespace at::cuda::detail
#endif

namespace at::cuda { namespace {

template <typename Handle_t, void Create(Handle_t *), void Destroy(Handle_t)>
struct DeviceThreadHandlePool : public std::enable_shared_from_this<DeviceThreadHandlePool<Handle_t, Create, Destroy>> {

    struct Handle {
    Handle_t handle;
    Handle(bool create = false) : handle(nullptr)
    {
        if(create) Create(&handle);
    }
    // std::vector.emplace() and push_back() may route through temporaries and call
    // copy/move constructors along the way.  If this is the case, we don't want
    // the destructors of temporaries to call cudnnDestroy on the handle.
    // We can achieve safety (for the narrow case of stashing within std::vectors)
    // by making Handle moveable but not copyable, and transferring handle ownership
    // to the latest constructed object.  This is not a substitute for full-blown
    // reference counting, but reference counting may be overkill here.
    // Another alternative is to wrap the saved Handles in unique_ptrs, i.e.,
    // unordered_map<int, vector<unique_ptr<Handle>>> created_handles;
    Handle(const Handle& rhs) = delete;
    // Following https://stackoverflow.com/questions/3279543/what-is-the-copy-and-swap-idiom
    Handle(Handle&& rhs) noexcept : Handle() { std::swap(handle, rhs.handle); }
    // operator= takes argument by value
    Handle& operator=(Handle rhs) { std::swap(handle, rhs.handle); return *this; }
    ~Handle() {
        if(handle) Destroy(handle);
    }
    };

    std::mutex mutex;

    // Handles are lazily created as different threads request them,
    // but are never destroyed until the end of the process.
    // The maximum number of handles this process will create for each device is equal
    // to the high-water mark of the number of concurrently active threads that request
    // handles for that device (per stream for ROCm's stream-keyed handles, plus one for
    // each thread that used hipStreamPerThread, which also keeps any workspaces cached
    // for it; see released_stream_handles).
    // When threads terminate, they release their handles back into the pool for reuse.
    // Otherwise, new handles would be created every time new threads were spawned,
    // resulting in poor performance for Python modules that repeatedly or frequently
    // spawned new sets of threads (like DataParallel, which creates a new set of threads
    // for each forward pass).
    //
    // To prevent potential deadlocks, we explicitly choose not to cap the number
    // of handles that are created per device.
    // Example of danger: If we cap the max handles at 4, and 5 threads are sharing a device,
    // only 4 can make forward progress at any time. The other 4 will not release their
    // handles until they exit, so the fifth cannot make progress until then.  This is
    // not a problem...UNLESS all 5 threads attempt some sort of synchronization at an
    // intermediate point (ie, before any of them have exited).  We have no way to anticipate
    // or enforce that user threads will not attempt such intermediate synchronization.
    // The only way to ensure safety is to avoid imposing a cap on the number of handles.
    std::unordered_map<int, std::vector<Handle>> created_handles;
    std::unordered_map<int, std::vector<Handle_t>> available_handles;

#ifdef USE_ROCM
    // A stream-keyed handle owns device state that its kernels use while they
    // run, so two launches on one handle must not overlap. An exiting thread
    // returns its handles while work it queued may still be running, so each
    // one goes back to the list of the stream it ran on, and only a thread
    // reserving a handle for that stream takes it, behind that work. So in a
    // pool used only through reserve(device, stream), available_handles holds
    // only handles that no stream has used, such as the spares that
    // ensureAvailable creates.
    //
    // Graph replay is not covered: a replayed graph keeps using the handles it
    // was captured with, so a replay can overlap work that a later holder of
    // one of them runs (see Note [hipblaslt handles] in CublasHandlePool.cpp
    // and https://github.com/pytorch/pytorch/issues/203474).
    //
    // So a stream-keyed pool keeps, for each stream, as many handles as the most
    // live threads that had used it at once, and a list is never freed. A handle
    // reserved for hipStreamPerThread, which is a different stream in each
    // thread, is never returned, so those grow with every thread that used it.
    // A hipblaslt handle holds about 25 MiB of device memory.
    //
    // A list outlives its stream and is reused by a new stream created at the
    // same address. That is safe only because hipStreamDestroy waits for the
    // stream's queued work, which HIP does not promise;
    // StreamDestroyWaitsForQueuedWork in cuda_cublas_handle_pool_test.cpp
    // checks it.
    std::map<std::pair<int, void*>, std::vector<Handle_t>> released_stream_handles;

    // Creates a handle without holding `mutex`, which threads on every device
    // share, because creating a handle can synchronize the device.
    Handle_t createHandle(int device)
    {
        Handle created(true /*create*/);
        const Handle_t handle = created.handle;
        std::lock_guard<std::mutex> guard(mutex);
        created_handles[device].push_back(std::move(created));
        return handle;
    }

    // Tops up the spares for `device`, handles no stream has used, to `n`.
    void ensureAvailable(int device, size_t n)
    {
        while(true)
        {
            {
            std::lock_guard<std::mutex> guard(mutex);
            if(available_handles[device].size() >= n)
                return;
            }
            const Handle_t handle = createHandle(device);
            std::lock_guard<std::mutex> guard(mutex);
            available_handles[device].push_back(handle);
        }
    }
#endif

    // PoolWindow lazily creates and caches the handles that a particular thread is using,
    // so in the common case handle access doesn't incur either handle creation or a mutex lock.
    class PoolWindow
    {
    public:
    PoolWindow(std::shared_ptr<DeviceThreadHandlePool> parent): weak_parent(std::move(parent)) {}
    ~PoolWindow(){ release(); }

    Handle_t reserve(int device)
    {
        // If this thread already has a handle for this device, return it
        auto mine = my_handles.find(device);
        if(mine != my_handles.end())
        return mine->second;

        // otherwise, either grab a handle from the pool if one is available,
        // or if not, create a new one.
        auto parent = weak_parent.lock();
        TORCH_CHECK(parent, "Cannot create handle during program termination");
        std::lock_guard<std::mutex> guard(parent->mutex);

        if(parent->available_handles[device].size() > 0)
        {
        my_handles[device] = parent->available_handles[device].back();
        parent->available_handles[device].pop_back();
        }
        else
        {
        // In local testing, I do observe that emplace_back sometimes routes through temporaries
        // that incur move-constructor and destructor calls.  See comments in Handle above.
        parent->created_handles[device].emplace_back(true /*create*/);
        my_handles[device] = parent->created_handles[device].back().handle;
        }

        return my_handles[device];
    }

#ifdef USE_ROCM
    // hipblaslt cannot share a single handle across multiple streams, so this
    // overload returns a handle unique to each (device, stream) pair. With
    // create=false it returns nullptr instead of creating a handle when this
    // thread has none for the pair, no exited thread returned one for it, and
    // the pool has no spare.
    Handle_t reserve(int device, void* stream, bool create = true)
    {
        auto key = std::make_pair(device, stream);
        // If this thread already has a handle for this (device, stream), return it
        auto mine = my_stream_handles.find(key);
        if(mine != my_stream_handles.end())
        return mine->second;

        // otherwise, take a handle this stream used before, or one that no
        // stream has used, or create a new one.
        auto parent = weak_parent.lock();
        TORCH_CHECK(parent, "Cannot create handle during program termination");
        Handle_t handle = nullptr;
        {
        std::lock_guard<std::mutex> guard(parent->mutex);
        auto released = parent->released_stream_handles.find(key);
        if(released != parent->released_stream_handles.end() && !released->second.empty())
        {
        handle = my_stream_handles[key] = released->second.back();
        released->second.pop_back();
        }
        else if(parent->available_handles[device].size() > 0)
        {
        handle = my_stream_handles[key] = parent->available_handles[device].back();
        parent->available_handles[device].pop_back();
        }
        }
        if(handle == nullptr)
        {
        if(!create)
            return nullptr;
        handle = my_stream_handles[key] = parent->createHandle(device);
        }

        if(stream == static_cast<void*>(hipStreamPerThread))
            detail::warnPerThreadStreamHandleKept();
        return handle;
    }

    // Returns this thread's handle for (device, stream) without creating one,
    // or nullptr if it has none.
    Handle_t find(int device, void* stream) const
    {
        auto it = my_stream_handles.find(std::make_pair(device, stream));
        return it == my_stream_handles.end() ? nullptr : it->second;
    }
#endif

    private:
    // Stores the per-device handles currently owned by this thread
    std::unordered_map<int, Handle_t> my_handles;
#ifdef USE_ROCM
    // Stores per-(device, stream) handles for ROCm, where hipblaslt
    // requires a unique handle per stream.
    std::map<std::pair<int, void*>, Handle_t> my_stream_handles;
#endif

    std::weak_ptr<DeviceThreadHandlePool> weak_parent;

    // Called by the destructor.  Releases this thread's handles back into the pool.
    void release() {
        if(!my_handles.empty()) {
            auto parent = weak_parent.lock();
            if (!parent) {
                // If this thread exits after atexit handlers have completed, the
                // cuda context itself may be invalid, so we must leak the handles.
                return;
            }

            std::lock_guard<std::mutex> guard(parent->mutex);
            for(auto d_h : my_handles)
                parent->available_handles[d_h.first].push_back(d_h.second);
        }
#ifdef USE_ROCM
        if(!my_stream_handles.empty()) {
            auto parent = weak_parent.lock();
            if (!parent) {
                return;
            }

            std::lock_guard<std::mutex> guard(parent->mutex);
            for(auto& [key, handle] : my_stream_handles) {
                if(key.second != static_cast<void*>(hipStreamPerThread))
                    parent->released_stream_handles[key].push_back(handle);
            }
        }
#endif
    }
    };

    // Warning:
    // If you want to change this function, be aware that this function will be called
    // by multiple threads and there is no mutex guarding the call of this function, so
    // make sure your implementation is thread-safe.
    PoolWindow *newPoolWindow() {
        // The returned pointer will be owned by a thread local variable
        // so that different threads does not share the same PoolWindow.
        return new PoolWindow(this->shared_from_this());
    }
};

}}  // namespace at::cuda::detail::<anonymous>
