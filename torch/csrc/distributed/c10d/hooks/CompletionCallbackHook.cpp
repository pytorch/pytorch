// Copyright (c) Meta Platforms, Inc. and affiliates.

#include <torch/csrc/distributed/c10d/hooks/CompletionCallbackHook.hpp>

#include <c10/util/thread_name.h>

#include <utility>
#include <vector>

#if defined(__GLIBCXX__)
#include <cxxabi.h>
#endif

#ifndef _WIN32
#include <pthread.h>
#include <unistd.h>
#endif

namespace c10d {

namespace {

// Hook ids must not collide with user-registered hooks; carve out a range
// far above small hand-picked ids.
std::atomic<int64_t> next_hook_id{0x434f4d50 /* 'COMP' */};

constexpr size_t kMaxPending = 1 << 16;
constexpr size_t kMaxEarly = 1024;

// Set on dispatch threads, where callbacks run.
thread_local bool on_dispatch_thread = false;

// Live dispatchers, for stopAll(). Type-erased: Dispatcher is private.
struct Registry {
  std::mutex mutex;
  std::vector<std::weak_ptr<void>> dispatchers;
  bool stopped = false;
};

// Leaked: used at interpreter exit, after static destructors may have run.
Registry*& registryPtr() {
  static Registry* r = [] {
#ifndef _WIN32
    // A forked child has none of the parent's dispatch threads, so it must
    // not wait for them at exit. The old registry's mutex may be held by a
    // thread that no longer exists: leak it.
    pthread_atfork(nullptr, nullptr, [] { registryPtr() = new Registry(); });
#endif
    return new Registry();
  }();
  return r;
}

// Registers the fork handler at load time rather than racing a fork on first
// use.
[[maybe_unused]] const bool registry_initialized = (registryPtr(), true);

Registry& registry() {
  return *registryPtr();
}

} // namespace

// Shared between the hook and its thread, so the thread can outlive the hook:
// the hook may be destroyed on the dispatch thread itself (a callback dropping
// the last reference to the group), where it cannot be joined.
struct CompletionCallbackHook::Dispatcher {
  explicit Dispatcher(CompletionCallback cb) : callback(std::move(cb)) {}

  void run() {
    c10::setThreadName("pt_completion_cb");
    on_dispatch_thread = true;
    std::unique_lock<std::mutex> lock(mutex);
    thread_id = std::this_thread::get_id();
    while (true) {
      cv.wait(lock, [this] { return stopped || !queue.empty(); });
      if (queue.empty()) {
        break;
      }
      auto info = std::move(queue.front());
      queue.pop_front();
      delivering = true;
      lock.unlock();
      try {
        callback(info);
      } catch (const std::exception& e) {
        LOG(ERROR) << "CompletionCallbackHook: callback threw: " << e.what();
#if defined(__GLIBCXX__)
      } catch (abi::__forced_unwind&) {
        // Thread cancellation, e.g. by an interpreter that is finalizing.
        // Swallowing it aborts the process.
        lock.lock();
        delivering = false;
        exited = true;
        cv.notify_all();
        lock.unlock();
        throw;
#endif
      } catch (...) {
        LOG(ERROR) << "CompletionCallbackHook: callback threw";
      }
      lock.lock();
      delivering = false;
      cv.notify_all();
    }
    // Release the callback (and whatever it captures) on this thread, not on
    // whichever thread happens to drop the last reference to the dispatcher.
    auto cb = std::move(callback);
    lock.unlock();
    cb = nullptr;
    lock.lock();
    exited = true;
    cv.notify_all();
  }

  void push(CompletedOpInfo info) {
    {
      std::lock_guard<std::mutex> lock(mutex);
      if (stopped) {
        return;
      }
      queue.push_back(std::move(info));
    }
    cv.notify_one();
  }

  // discard: drop queued callbacks instead of delivering them.
  // wait: wait for a running callback to return.
  void stop(bool discard, bool wait) {
    std::unique_lock<std::mutex> lock(mutex);
    stopped = true;
    if (discard) {
      queue.clear();
    }
    cv.notify_all();
    if (!wait || thread_id == std::this_thread::get_id()) {
      return;
    }
#ifndef _WIN32
    if (getpid() != owner_pid) {
      return;
    }
#endif
    cv.wait(lock, [this] { return !delivering; });
  }

  void waitExited() {
    std::unique_lock<std::mutex> lock(mutex);
    if (thread_id == std::this_thread::get_id()) {
      return;
    }
    cv.wait(lock, [this] { return exited; });
  }

  CompletionCallback callback;
  std::mutex mutex;
  std::condition_variable cv;
  std::deque<CompletedOpInfo> queue;
  bool stopped = false;
  bool delivering = false;
  bool exited = false;
  std::thread::id thread_id;
#ifndef _WIN32
  // A forked child inherits the state but not the thread.
  pid_t owner_pid = getpid();
#endif
};

std::shared_ptr<CompletionCallbackHook> CompletionCallbackHook::attach(
    const c10::intrusive_ptr<ProcessGroup>& pg,
    CompletionCallback callback) {
  TORCH_CHECK(pg, "CompletionCallbackHook: null process group");
  TORCH_CHECK(
      pg->supportsCompletionHooks(),
      "CompletionCallbackHook: backend ",
      pg->getDefaultBackend()->getBackendName(),
      " does not support completion hooks");
  TORCH_CHECK(
      !on_dispatch_thread,
      "CompletionCallbackHook: attach() must not be called from a completion callback");
  auto hook = std::shared_ptr<CompletionCallbackHook>(
      new CompletionCallbackHook(pg, std::move(callback)));
  {
    auto& r = registry();
    std::lock_guard<std::mutex> lock(r.mutex);
    TORCH_CHECK(
        !r.stopped, "CompletionCallbackHook: the interpreter is exiting");
    std::erase_if(r.dispatchers, [](const auto& d) { return d.expired(); });
    r.dispatchers.push_back(hook->dispatcher_);
  }
  try {
    std::thread([dispatcher = hook->dispatcher_] {
      dispatcher->run();
    }).detach();
  } catch (...) {
    // Already published: a concurrent stopAll() must not wait for it.
    std::lock_guard<std::mutex> lock(hook->dispatcher_->mutex);
    hook->dispatcher_->exited = true;
    throw;
  }
  // The lambdas own the hook, so it lives as long as the group's hook maps:
  // the caller does not have to keep the returned handle alive. Completion
  // first, so no op whose key the post-hook registers can complete unseen.
  pg->registerCompletionHook(
      hook->hook_id_,
      [hook](const CompletionHookArgs& args) { hook->onCompletion(args); });
  pg->registerPostHook(
      hook->hook_id_, [hook](const PostHookArgs& args) { hook->onPost(args); });
  return hook;
}

CompletionCallbackHook::CompletionCallbackHook(
    const c10::intrusive_ptr<ProcessGroup>& pg,
    CompletionCallback callback)
    : pg_(pg),
      hook_id_(next_hook_id++),
      dispatcher_(std::make_shared<Dispatcher>(std::move(callback))) {}

CompletionCallbackHook::~CompletionCallbackHook() {
  // Only reached once the group dropped its hooks (or remove() ran). Deliver
  // what is already queued and let the thread exit on its own; joining here
  // could deadlock, since this may run under the GIL or on that thread.
  dispatcher_->stop(/*discard=*/false, /*wait=*/false);
}

void CompletionCallbackHook::remove() {
  // Unregistering drops the group's references, which are normally the only
  // ones keeping this hook alive.
  auto keepalive = weak_from_this().lock();
  c10::intrusive_ptr<ProcessGroup> pg;
  {
    std::lock_guard<std::mutex> lock(mutex_);
    attached_ = false;
    if (!on_dispatch_thread) {
      pg = pg_.lock();
      pg_.reset();
    }
    pending_.clear();
    pending_order_.clear();
    early_.clear();
    early_order_.clear();
  }
  // A callback (of any hook) runs concurrently with ops issued on the group,
  // which read the group's post-hook table without a lock, so leave it to a
  // later remove() from another thread. Do not wait either: two callbacks
  // removing each other's hooks would deadlock.
  if (on_dispatch_thread) {
    dispatcher_->stop(/*discard=*/true, /*wait=*/false);
    return;
  }
  // Not under mutex_: a completion hook may be blocked on mutex_ right now
  // while the backend holds the lock guarding its hook table.
  if (pg) {
    pg->unregisterPostHook(hook_id_);
    pg->unregisterCompletionHook(hook_id_);
  }
  dispatcher_->stop(/*discard=*/true, /*wait=*/true);
}

void CompletionCallbackHook::stopAll() {
  std::vector<std::shared_ptr<Dispatcher>> live;
  {
    auto& r = registry();
    std::lock_guard<std::mutex> lock(r.mutex);
    for (const auto& weak : r.dispatchers) {
      if (auto d = weak.lock()) {
        live.push_back(std::static_pointer_cast<Dispatcher>(std::move(d)));
      }
    }
    r.dispatchers.clear();
    r.stopped = true;
  }
  for (const auto& d : live) {
    d->stop(/*discard=*/true, /*wait=*/false);
  }
  for (const auto& d : live) {
    d->waitExited();
  }
}

void CompletionCallbackHook::onPost(const PostHookArgs& args) {
  auto prev = last_op_id_.load();
  while (prev < args.op_id &&
         !last_op_id_.compare_exchange_weak(prev, args.op_id)) {
  }
  if (!args.work) {
    return;
  }
  auto key = args.work->getCompletionKey();
  std::optional<CompletedOpInfo> done;
  {
    std::lock_guard<std::mutex> lock(mutex_);
    if (!attached_) {
      return;
    }
    if (auto it = early_.find(key); it != early_.end()) {
      done = CompletedOpInfo{args.name, args.op_id, it->second};
      early_.erase(it);
    } else {
      auto& op = pending_[key];
      // A key reused before its previous op completed (a sentinel Work that
      // never completes on its own); the newer op wins.
      if (op.op_id >= 0) {
        pending_order_.erase(op.op_id);
      }
      op = PendingOp{args.op_id, args.name};
      pending_order_.emplace(args.op_id, key);
      if (pending_order_.size() > kMaxPending) {
        auto oldest = pending_order_.begin();
        pending_.erase(oldest->second);
        pending_order_.erase(oldest);
      }
    }
  }
  if (done) {
    dispatcher_->push(*done);
  }
}

void CompletionCallbackHook::onCompletion(const CompletionHookArgs& args) {
  std::optional<CompletedOpInfo> done;
  {
    std::lock_guard<std::mutex> lock(mutex_);
    if (!attached_) {
      return;
    }
    auto it = pending_.find(args.completionKey);
    if (it == pending_.end()) {
      if (early_.emplace(args.completionKey, args.duration_ms).second) {
        early_order_.push_back(args.completionKey);
      }
      if (early_order_.size() > kMaxEarly) {
        early_.erase(early_order_.front());
        early_order_.pop_front();
      }
      return;
    }
    done = CompletedOpInfo{it->second.name, it->second.op_id, args.duration_ms};
    pending_order_.erase(it->second.op_id);
    pending_.erase(it);
  }
  dispatcher_->push(*done);
}

} // namespace c10d
