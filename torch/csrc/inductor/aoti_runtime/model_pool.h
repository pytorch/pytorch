#pragma once

#include <algorithm>
#include <condition_variable>
#include <memory>
#include <mutex>
#include <unordered_map>
#include <vector>

// WARNING: Be careful when adding new includes here. This header will be used
// in model.so, and should not refer to any aten/c10 headers except the stable
// C ABI defined in torch/csrc/inductor/aoti_torch/c/shim.h. The same rule
// applies to other files under torch/csrc/inductor/aoti_runtime/.
#include <torch/csrc/inductor/aoti_runtime/interface.h>

namespace torch::aot_inductor {

// Owns the model instances that AOTInductorModelContainer hands out to
// concurrent run() calls. Model must provide is_finished() and
// wait_for_completion(); Stream is an opaque device stream handle.
template <typename Model, typename Stream>
class ModelPool {
  struct Slot;

 public:
  struct StreamLock {
    StreamLock() = default;
    explicit StreamLock(const std::shared_ptr<std::mutex>& stream_mutex)
        : stream_mutex_(stream_mutex), lock_(*stream_mutex_) {}

    std::shared_ptr<std::mutex> stream_mutex_;
    std::unique_lock<std::mutex> lock_;
  };

  // Owns a checked-out model. A run that succeeds ends with mark_in_flight():
  // its device work may still be in flight, so the model is reused only once
  // that work has finished. Dropping the lease means the run threw: the model
  // goes back to the available pool at once, without waiting for completion
  // and with any stream binding cleared, so the next checkout may get it while
  // device work the failed run launched is still in flight.
  class Lease {
   public:
    Lease(const Lease&) = delete;
    Lease& operator=(const Lease&) = delete;
    Lease(Lease&& other) noexcept : pool_(other.pool_), slot_(other.slot_) {
      other.slot_ = nullptr;
    }
    Lease& operator=(Lease&&) = delete;

    ~Lease() {
      if (slot_ != nullptr) {
        pool_->return_model_to_available(*slot_);
      }
    }

    Model* operator->() const& {
      AOTI_RUNTIME_CHECK(slot_ != nullptr, "The lease no longer holds a model");
      return slot_->model.get();
    }
    // A temporary lease would return its model as available at the end of the
    // full expression, while the run's device work may still be in flight.
    Model* operator->() && = delete;
    Model* operator->() const&& = delete;

    void mark_in_flight() {
      AOTI_RUNTIME_CHECK(slot_ != nullptr, "The lease no longer holds a model");
      pool_->return_model_to_in_flight(*slot_);
      slot_ = nullptr;
    }

   private:
    friend class ModelPool;
    Lease(ModelPool& pool, Slot& slot) : pool_(&pool), slot_(&slot) {}

    ModelPool* pool_;
    Slot* slot_;
  };

  void reserve(size_t num_models) {
    slots_.reserve(num_models);
  }

  // Takes ownership of a model. The pool is filled before it is shared.
  Model& add_model(std::unique_ptr<Model> model) {
    slots_.push_back(
        Slot{std::move(model), State::kAvailable, nullptr, next_seq_++});
    return *slots_.back().model;
  }

  size_t size() const {
    return slots_.size();
  }

  // The first-added model. Its identity is stable for the pool's lifetime.
  // The pool must not be empty.
  Model& primary_model() {
    return *slots_.front().model;
  }

  const Model& primary_model() const {
    return *slots_.front().model;
  }

  // The caller must keep runs from using the models meanwhile.
  template <typename Fn>
  void for_each_model(Fn&& fn) {
    for (auto& slot : slots_) {
      fn(*slot.model);
    }
  }

  // Stream affinity is intended for controlled multi-stream benchmarking with
  // one model instance per long-lived stream. It deliberately waits for a
  // stream's bound model instead of using another instance, trading host-side
  // pipelining for stable stream-to-model assignment.
  void set_use_stream_affinity(bool use_stream_affinity) {
    std::lock_guard lk(models_mutex_);
    AOTI_RUNTIME_CHECK(
        std::all_of(
            slots_.begin(),
            slots_.end(),
            [](const Slot& slot) { return slot.state == State::kAvailable; }),
        "Stream affinity can only be reconfigured while the model pool is "
        "quiescent");
#if defined(USE_CUDA) || defined(USE_XPU)
    use_stream_affinity_ = use_stream_affinity;
#else
    AOTI_RUNTIME_CHECK(
        !use_stream_affinity,
        "Stream affinity is only supported for device runtimes");
#endif
    for (auto& slot : slots_) {
      slot.stream = nullptr;
    }
    reclaim_unused_stream_mutexes();
  }

  bool use_stream_affinity() const {
    return use_stream_affinity_;
  }

  int64_t get_stream_affinity_model_index(Stream stream) {
    std::lock_guard lk(models_mutex_);
    if (!use_stream_affinity_) {
      return AOTI_STREAM_AFFINITY_DISABLED;
    }
    Slot* slot = bound_slot(stream);
    if (slot == nullptr) {
      return AOTI_STREAM_AFFINITY_UNBOUND;
    }
    return slot - slots_.data();
  }

  [[nodiscard]] Lease checkout() {
    return Lease(*this, get_available_slot());
  }

  [[nodiscard]] Lease checkout(Stream stream) {
    return Lease(*this, get_available_slot(stream));
  }

  StreamLock acquire_stream_lock(Stream stream) {
#if defined(USE_CUDA) || defined(USE_XPU)
    std::shared_ptr<std::mutex> stream_mutex;
    {
      std::lock_guard lk(models_mutex_);
      if (!use_stream_affinity_ || stream == nullptr) {
        return {};
      }
      reclaim_unused_stream_mutexes();
      auto [it, inserted] = stream_mutexes_.try_emplace(stream, nullptr);
      if (inserted) {
        it->second = std::make_shared<std::mutex>();
      }
      stream_mutex = it->second;
    }
    return StreamLock(stream_mutex);
#else
    (void)stream;
    return {};
#endif
  }

 private:
  // Every model is in exactly one of these states, guarded by models_mutex_.
  enum class State : uint8_t {
    // Free to check out.
    kAvailable,
    // Its last run reached the device and may still be executing there.
    kInFlight,
    // Owned by one thread, which is running it or waiting for its device work.
    kCheckedOut,
  };

  struct Slot {
    std::unique_ptr<Model> model;
    State state;
    // The stream bound to this model when stream affinity is enabled, or
    // nullptr. A binding persists across states; borrowing a bound model keeps
    // it, while a failed run clears it so the next call may choose another
    // model.
    Stream stream;
    // When the slot entered its current state: the newest available model is
    // checked out first, and the oldest in-flight one is waited on first.
    uint64_t seq;
  };

  Slot& get_available_slot() {
    std::unique_lock lk(models_mutex_);
    return checkout_available_slot(lk);
  }

  Slot& get_available_slot(Stream stream) {
    std::unique_lock lk(models_mutex_);
#if defined(USE_CUDA) || defined(USE_XPU)
    if (use_stream_affinity_ && stream != nullptr) {
      return get_stream_affine_slot(stream, lk);
    }
#else
    (void)stream;
#endif
    return checkout_available_slot(lk);
  }

  void return_model_to_available(Slot& slot) {
    {
      std::lock_guard lk(models_mutex_);
      // A failed execution invalidates its affinity so a subsequent call can
      // select another model instance.
      slot.stream = nullptr;
      transition(slot, State::kCheckedOut, State::kAvailable);
    }
    model_state_changed_.notify_all();
  }

  void return_model_to_in_flight(Slot& slot) {
    {
      std::lock_guard lk(models_mutex_);
      transition(slot, State::kCheckedOut, State::kInFlight);
    }
    model_state_changed_.notify_all();
  }

  void transition(Slot& slot, State from, State to) {
    AOTI_RUNTIME_CHECK(slot.state == from, "Invalid model pool transition");
    slot.state = to;
    slot.seq = next_seq_++;
  }

  Slot* bound_slot(Stream stream) {
    if (stream == nullptr) {
      return nullptr;
    }
    for (auto& slot : slots_) {
      if (slot.stream == stream) {
        return &slot;
      }
    }
    return nullptr;
  }

  Slot* newest(State state, bool unbound_only = false) {
    Slot* result = nullptr;
    for (auto& slot : slots_) {
      if (slot.state == state && (!unbound_only || slot.stream == nullptr) &&
          (result == nullptr || slot.seq > result->seq)) {
        result = &slot;
      }
    }
    return result;
  }

  Slot* oldest(State state, uint64_t min_seq = 0) {
    Slot* result = nullptr;
    for (auto& slot : slots_) {
      if (slot.state == state && slot.seq >= min_seq &&
          (result == nullptr || slot.seq < result->seq)) {
        result = &slot;
      }
    }
    return result;
  }

  Slot& checkout_available_slot(std::unique_lock<std::mutex>& lk) {
    return take_available_slot(lk, /*prefer_unbound=*/use_stream_affinity_);
  }

  Slot& take_available_slot(
      std::unique_lock<std::mutex>& lk,
      bool prefer_unbound) {
    while (oldest(State::kAvailable) == nullptr) {
      reclaim_finished_models(lk);
    }
    // Preserve existing bindings when possible. If every available model is
    // bound, the caller borrows one while its owning stream waits.
    Slot* slot = prefer_unbound ? newest(State::kAvailable, true) : nullptr;
    if (slot == nullptr) {
      slot = newest(State::kAvailable);
    }
    transition(*slot, State::kAvailable, State::kCheckedOut);
    return *slot;
  }

  Slot& get_stream_affine_slot(
      Stream stream,
      std::unique_lock<std::mutex>& lk) {
    while (true) {
      Slot* slot = bound_slot(stream);
      if (slot == nullptr) {
        break;
      }

      if (slot->state == State::kAvailable) {
        transition(*slot, State::kAvailable, State::kCheckedOut);
        return *slot;
      }

      if (slot->state == State::kInFlight) {
        transition(*slot, State::kInFlight, State::kCheckedOut);
        lk.unlock();
        try {
          slot->model->wait_for_completion();
        } catch (...) {
          return_model_to_available(*slot);
          throw;
        }
        return *slot;
      }

      // A null-stream checkout or pool reclamation can temporarily borrow a
      // bound model. Same non-null stream calls serialize before this point.
      model_state_changed_.wait(
          lk, [slot]() { return slot->state != State::kCheckedOut; });
    }

    // Bind an unbound model on a stream's first use. If all model instances are
    // already bound, reclaim and rebind one for an oversubscribed stream.
    Slot& slot = take_available_slot(lk, /*prefer_unbound=*/true);
    Slot* previous = bound_slot(stream);
    if (previous != nullptr) {
      previous->stream = nullptr;
    }
    slot.stream = stream;
    return slot;
  }

  void reclaim_unused_stream_mutexes() {
    for (auto it = stream_mutexes_.begin(); it != stream_mutexes_.end();) {
      if (bound_slot(it->first) == nullptr && it->second.use_count() == 1) {
        it = stream_mutexes_.erase(it);
      } else {
        ++it;
      }
    }
  }

  void reclaim_finished_models(std::unique_lock<std::mutex>& lk) {
    // Move every finished run back to available, oldest first, so the most
    // recently launched one is reused first. This allocates nothing, and if
    // is_finished() throws, every slot is still in exactly one state.
    bool reclaimed = false;
    uint64_t min_seq = 0;
    while (Slot* slot = oldest(State::kInFlight, min_seq)) {
      // Read before transition() gives the slot a new seq.
      min_seq = slot->seq + 1;
      if (slot->model->is_finished()) {
        transition(*slot, State::kInFlight, State::kAvailable);
        reclaimed = true;
      }
    }
    if (reclaimed) {
      model_state_changed_.notify_all();
      return;
    }

    model_state_changed_.wait(lk, [this]() {
      return oldest(State::kAvailable) != nullptr ||
          oldest(State::kInFlight) != nullptr;
    });
    if (oldest(State::kAvailable) != nullptr) {
      return;
    }
    // Let's make the schedule simple first. We always wait on the oldest
    // in-flight model to be complete.
    Slot* slot = oldest(State::kInFlight);
    transition(*slot, State::kInFlight, State::kCheckedOut);
    lk.unlock();
    try {
      slot->model->wait_for_completion();
    } catch (...) {
      lk.lock();
      slot->stream = nullptr;
      transition(*slot, State::kCheckedOut, State::kAvailable);
      model_state_changed_.notify_all();
      throw;
    }
    lk.lock();
    transition(*slot, State::kCheckedOut, State::kAvailable);
    model_state_changed_.notify_all();
  }

  // One slot per model instance, which it owns, in the container's index
  // order. The vector is filled before the pool is shared, so slot pointers
  // stay valid.
  std::vector<Slot> slots_;
  uint64_t next_seq_{0};

  bool use_stream_affinity_{false};
  // Callers enabling affinity must keep bound streams alive to avoid a reused
  // device handle inheriting an old binding. Unbound mutexes are reclaimed
  // once no in-flight StreamLock shares ownership.
  std::unordered_map<Stream, std::shared_ptr<std::mutex>> stream_mutexes_;

  // Protects the slots and the stream mutexes.
  std::mutex models_mutex_;

  std::condition_variable model_state_changed_;
};

} // namespace torch::aot_inductor
