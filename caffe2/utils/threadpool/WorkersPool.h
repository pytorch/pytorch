#include "caffe2/core/common.h"
#include "caffe2/core/logging.h"
#include <atomic>
#include <thread>

namespace caffe2 {

// Uses code derived from gemmlowp,
// https://github.com/google/gemmlowp/blob/6c91e1ed0c2eff1182d804310b92911fe9c18019/internal/multi_thread_gemm.h
// Changes:
// - allocation-free execute()
// - Use RAII where possible.
// - Run the first task on the main thread (since that is the largest task).
// - removed custom allocator.
// - Removed some ifdef's
// - cache-line align Worker.

constexpr size_t kGEMMLOWPCacheLineSize = 64;

template <typename T>
struct AllocAligned {
  // Allocate a T aligned at an `align` byte address
  template <typename... Args>
  static T* alloc(Args&&... args) {
    void* p = nullptr;
// FIXME: we should just be able to use std::align
#if !defined(__ANDROID__)
    posix_memalign((void**)&p, kGEMMLOWPCacheLineSize, sizeof(T));
#else
    p = memalign(kGEMMLOWPCacheLineSize, sizeof(T));
#endif

    if (p) {
      return new (p) T(std::forward<Args>(args)...);
    }

    return nullptr;
  }

  // Free a T previously allocated via AllocAligned<T>::alloc()
  static void release(T* p) {
    if (p) {
      p->~T();
      free((void*)p);
    }
  }
};

// Deleter object for unique_ptr for an aligned object
template <typename T>
struct AlignedDeleter {
  void operator()(T* p) const { AllocAligned<T>::release(p); }
};

// make_unique that guarantees alignment
template <typename T>
struct MakeAligned {
  template <typename... Args>
  static std::unique_ptr<T, AlignedDeleter<T>> make(Args&&... args) {
    return std::unique_ptr<T, AlignedDeleter<T>>(
        AllocAligned<T>::alloc(std::forward<Args>(args)...));
  }
};

#ifdef __ARM_NEON__
#define GEMMLOWP_ARM_32 1
#else
#define GEMMLOWP_X86 1
#endif

#define GEMMLOWP_USE_BUSYWAIT 1

const int kMaxBusyWaitNOPs = 32 * 1000 * 1000;

#define GEMMLOWP_NOP "nop\n"

#define GEMMLOWP_STRING_CONCAT_4(X) X X X X
#define GEMMLOWP_NOP4 GEMMLOWP_STRING_CONCAT_4(GEMMLOWP_NOP)
#define GEMMLOWP_NOP16 GEMMLOWP_STRING_CONCAT_4(GEMMLOWP_NOP4)
#define GEMMLOWP_NOP64 GEMMLOWP_STRING_CONCAT_4(GEMMLOWP_NOP16)

inline int Do256NOPs() {
  asm volatile(GEMMLOWP_NOP64);
  return 64;
}

#undef GEMMLOWP_STRING_CONCAT_4
#undef GEMMLOWP_NOP256
#undef GEMMLOWP_NOP64
#undef GEMMLOWP_NOP16
#undef GEMMLOWP_NOP4
#undef GEMMLOWP_NOP

inline void WriteBarrier() {
#ifdef GEMMLOWP_ARM_32
  asm volatile("" ::: "memory");
#elif defined(GEMMLOWP_X86)
  asm volatile("sfence" ::: "memory");
#else
#error "Unsupported architecture for WriteBarrier."
#endif
}

inline void ReadBarrier() {
#ifdef GEMMLOWP_ARM_32
  asm volatile("" ::: "memory");
#elif defined(GEMMLOWP_X86)
  asm volatile("lfence" ::: "memory");
#else
#error "Unsupported architecture for ReadBarrier."
#endif
}

// Waits until *var != initial_value.
//
// Returns the new value of *var. The guarantee here is that
// the return value is different from initial_value, and that that
// new value has been taken by *var at some point during the
// execution of this function. There is no guarantee that this is
// still the value of *var when this function returns, since *var is
// not assumed to be guarded by any lock.
//
// First does some busy-waiting for a fixed number of no-op cycles,
// then falls back to passive waiting for the given condvar, guarded
// by the given mutex.
//
// The idea of doing some initial busy-waiting is to help get
// better and more consistent multithreading benefits for small GEMM sizes.
// Busy-waiting help ensuring that if we need to wake up soon after having
// started waiting, then we can wake up quickly (as opposed to, say,
// having to wait to be scheduled again by the OS). On the other hand,
// we must still eventually revert to passive waiting for longer waits
// (e.g. worker threads having finished a GEMM and waiting until the next GEMM)
// so as to avoid permanently spinning.
//
template <typename T>
T WaitForVariableChange(volatile T* var,
                        T initial_value,
                        pthread_cond_t* cond,
                        pthread_mutex_t* mutex) {
  // If we are on a platform that supports it, spin for some time.
  {
    int nops = 0;
    // First, trivial case where the variable already changed value.
    T new_value = *var;
    if (new_value != initial_value) {
      ReadBarrier();
      return new_value;
    }
    // Then try busy-waiting.
    while (nops < kMaxBusyWaitNOPs) {
      nops += Do256NOPs();
      new_value = *var;
      if (new_value != initial_value) {
        ReadBarrier();
        return new_value;
      }
    }
  }

  // Finally, do real passive waiting.
  pthread_mutex_lock(mutex);
  T new_value = *var;
  if (new_value == initial_value) {
    pthread_cond_wait(cond, mutex);
    new_value = *var;
    assert(new_value != initial_value);
  }
  pthread_mutex_unlock(mutex);
  return new_value;
}

// A BlockingCounter lets one thread to wait for N events to occur.
// This is how the master thread waits for all the worker threads
// to have finished working.
class BlockingCounter {
 public:
  BlockingCounter()
      : cond_(PTHREAD_COND_INITIALIZER),
        mutex_(PTHREAD_MUTEX_INITIALIZER),
        count_(0),
        initial_count_(0) {}

  // Sets/resets the counter; initial_count is the number of
  // decrementing events that the Wait() call will be waiting for.
  void Reset(std::size_t initial_count) {
    pthread_mutex_lock(&mutex_);
    assert(count_ == 0);
    initial_count_ = initial_count;
    count_ = initial_count_;
    pthread_mutex_unlock(&mutex_);
  }

  // Decrements the counter; if the counter hits zero, signals
  // the thread that was waiting for that, and returns true.
  // Otherwise (if the decremented count is still nonzero),
  // returns false.
  bool DecrementCount() {
    pthread_mutex_lock(&mutex_);
    assert(count_ > 0);
    count_--;
    WriteBarrier();
    if (count_ == 0) {
      pthread_cond_signal(&cond_);
    }
    bool retval = count_ == 0;
    pthread_mutex_unlock(&mutex_);
    return retval;
  }

  // Waits for the N other threads (N having been set by Reset())
  // to hit the BlockingCounter.
  void Wait() {
    while (count_) {
      ReadBarrier();
      const std::size_t count_value = count_;
      if (count_value) {
        WaitForVariableChange(&count_, count_value, &cond_, &mutex_);
      }
    }
  }

 private:
  pthread_cond_t cond_;
  pthread_mutex_t mutex_;
  std::size_t count_;
  std::size_t initial_count_;
};

// A workload for a worker.
struct Task {
  Task() {}
  virtual ~Task() {}
  virtual void Run() = 0;
};

// A worker thread.
class alignas(kGEMMLOWPCacheLineSize) Worker {
 public:
  enum class State {
    ThreadStartup, // The initial state before the thread main loop runs.
    Ready, // Is not working, has not yet received new work to do.
    HasWork, // Has work to do.
    ExitAsSoonAsPossible // Should exit at earliest convenience.
  };

  explicit Worker(BlockingCounter* counter_to_decrement_when_ready)
      : task_(nullptr),
        state_cond_(PTHREAD_COND_INITIALIZER),
        state_mutex_(PTHREAD_MUTEX_INITIALIZER),
        state_(State::ThreadStartup),
        counter_to_decrement_when_ready_(counter_to_decrement_when_ready) {
    pthread_create(&thread_, nullptr, ThreadFunc, this);
  }

  ~Worker() {
    ChangeState(State::ExitAsSoonAsPossible);
    pthread_join(thread_, nullptr);
  }

  // Changes State; may be called from either the worker thread
  // or the master thread; however, not all state transitions are legal,
  // which is guarded by assertions.
  void ChangeState(State new_state) {
    pthread_mutex_lock(&state_mutex_);
    assert(new_state != state_);
    switch (state_) {
    case State::ThreadStartup:
      assert(new_state == State::Ready);
      break;
    case State::Ready:
      assert(new_state == State::HasWork || new_state == State::ExitAsSoonAsPossible);
      break;
    case State::HasWork:
      assert(new_state == State::Ready || new_state == State::ExitAsSoonAsPossible);
      break;
    default:
      abort();
    }
    state_ = new_state;
    pthread_cond_signal(&state_cond_);
    if (state_ == State::Ready) {
      counter_to_decrement_when_ready_->DecrementCount();
    }
    pthread_mutex_unlock(&state_mutex_);
  }

  // Thread entry point.
  void ThreadFunc() {
    ChangeState(State::Ready);

    // Thread main loop
    while (true) {
      // Get a state to act on
      // In the 'Ready' state, we have nothing to do but to wait until
      // we switch to another state.
      State state_to_act_upon =
          WaitForVariableChange(&state_, State::Ready, &state_cond_, &state_mutex_);

      // We now have a state to act on, so act.
      switch (state_to_act_upon) {
      case State::HasWork:
        // Got work to do! So do it, and then revert to 'Ready' state.
        assert(task_);
        task_->Run();
        task_ = nullptr;
        ChangeState(State::Ready);
        break;
      case State::ExitAsSoonAsPossible:
        return;
      default:
        abort();
      }
    }
  }

  static void* ThreadFunc(void* arg) {
    static_cast<Worker*>(arg)->ThreadFunc();
    return nullptr;
  }

  // Called by the master thead to give this worker work to do.
  // It is only legal to call this if the worker
  void StartWork(Task* task) {
    assert(!task_);
    // task->local_allocator = &local_allocator_;
    task_ = task;
    WriteBarrier();
    assert(state_ == State::Ready);
    ChangeState(State::HasWork);
  }

 private:
  // The underlying thread.
  pthread_t thread_;

  // The task to be worked on.
  Task* task_;

  // The condition variable and mutex guarding state changes.
  pthread_cond_t state_cond_;
  pthread_mutex_t state_mutex_;

  // The state enum tells if we're currently working, waiting for work, etc.
  State state_;

  // pointer to the master's thread BlockingCounter object, to notify the
  // master thread of when this worker switches to the 'Ready' state.
  BlockingCounter* const counter_to_decrement_when_ready_;
};

class WorkersPool {
 public:
  WorkersPool() {}

  void Execute(const std::vector<std::shared_ptr<Task>>& tasks) {
    assert(tasks.size() >= 1);
    // One of the tasks will be run on the current thread.
    int workers_count = tasks.size() - 1;
    CreateWorkers(workers_count);
    assert(workers_count <= workers_.size());
    counter_to_decrement_when_ready_.Reset(workers_count);
    int n = 0;
    std::for_each(++tasks.begin(), tasks.end(), [this, &n](auto& task) {
      workers_[n++]->StartWork(task.get());
    });
    // Execute the remaining workload immediately on the current thread.
    auto& task = tasks.front();
    task->Run();
    // Wait for the workers submitted above to finish.
    counter_to_decrement_when_ready_.Wait();
  }

 private:
  // Ensures that the pool has at least the given count of workers.
  // If any new worker has to be created, this function waits for it to
  // be ready.
  void CreateWorkers(std::size_t workers_count) {
    if (workers_.size() >= workers_count) {
      return;
    }
    counter_to_decrement_when_ready_.Reset(workers_count - workers_.size());
    while (workers_.size() < workers_count) {
      workers_.push_back(MakeAligned<Worker>::make(&counter_to_decrement_when_ready_));
    }
    counter_to_decrement_when_ready_.Wait();
  }

  DISABLE_COPY_AND_ASSIGN(WorkersPool);
  std::vector<std::unique_ptr<Worker, AlignedDeleter<Worker>>> workers_;
  // The BlockingCounter used to wait for the workers.
  BlockingCounter counter_to_decrement_when_ready_;
};
} // namespace caffe2
