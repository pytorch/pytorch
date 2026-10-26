#pragma once

#include <algorithm>
#include <condition_variable>
#include <mutex>
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
 public:
  // Owns a checked-out model. Dropping the lease returns the model to the
  // available pool, which is right for a run that never reached the device. A
  // run that did reach the device ends with mark_in_flight() instead: its work
  // may still be in flight, so the model is reused only once it has finished.
  class Lease {
   public:
    Lease(const Lease&) = delete;
    Lease& operator=(const Lease&) = delete;
    Lease(Lease&& other) noexcept : pool_(other.pool_), model_(other.model_) {
      other.model_ = nullptr;
    }
    Lease& operator=(Lease&&) = delete;

    ~Lease() {
      if (model_ != nullptr) {
        pool_->return_model_to_available(model_);
      }
    }

    Model* operator->() const {
      AOTI_RUNTIME_CHECK(
          model_ != nullptr, "The lease no longer holds a model");
      return model_;
    }

    Model& operator*() const {
      return *operator->();
    }

    void mark_in_flight() {
      AOTI_RUNTIME_CHECK(
          model_ != nullptr, "The lease no longer holds a model");
      pool_->return_model_in_flight(model_);
      model_ = nullptr;
    }

   private:
    friend class ModelPool;
    Lease(ModelPool& pool, Model* model) : pool_(&pool), model_(model) {}

    ModelPool* pool_;
    Model* model_;
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

  // Applies fn to every model while none is checked out or in flight, as
  // after drain() with new runs held off.
  template <typename Fn>
  void for_each_model(Fn&& fn) {
    std::lock_guard lk(models_mutex_);
    AOTI_RUNTIME_CHECK(
        quiescent(),
        "The models can only be updated together while the model pool is "
        "quiescent");
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
        quiescent(),
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

  // Checks out a model for a run on `stream`. With stream affinity, calls on
  // one non-null stream share a model and run one at a time: each waits until
  // the stream's model is free and its previous run has finished on the
  // device.
  Lease checkout(Stream stream = nullptr) {
    return Lease(*this, acquire(stream));
  }

  // Waits until no run is still executing on the device. The caller must keep
  // new runs from starting meanwhile.
  void drain() {
    std::unique_lock lk(models_mutex_);
    while (Slot* slot = oldest(State::kInFlight)) {
      transition(*slot, State::kInFlight, State::kCheckedOut);
      lk.unlock();
      wait_for_device(*slot);
      lk.lock();
      transition(*slot, State::kCheckedOut, State::kAvailable);
      model_state_changed_.notify_all();
    }
  }

  // For run_single_threaded(), which bypasses the pool's locking entirely.
  Model* first_available_model_unsynchronized() {
    Slot* slot = oldest(State::kAvailable);
    AOTI_RUNTIME_CHECK(slot != nullptr, "No model is available");
    return slot->model.get();
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

  // Every decision here is made under models_mutex_. Whenever the lock is
  // released to wait, the next iteration decides again from scratch; the only
  // thing carried across an unlock is a slot this thread owns as kCheckedOut.
  Model* acquire(Stream stream) {
    std::unique_lock lk(models_mutex_);
    while (true) {
      // Affinity can be reconfigured while this call waits for a model.
      const bool affine = use_stream_affinity_ && stream != nullptr;
      if (Slot* bound = affine ? bound_slot(stream) : nullptr) {
        if (bound->state == State::kAvailable) {
          transition(*bound, State::kAvailable, State::kCheckedOut);
          return bound->model.get();
        }
        if (bound->state == State::kInFlight) {
          transition(*bound, State::kInFlight, State::kCheckedOut);
          lk.unlock();
          wait_for_device(*bound);
          return bound->model.get();
        }
        // Checked out by an earlier call on this stream, or borrowed by a
        // checkout that found no unbound model.
        model_state_changed_.wait(
            lk, [bound]() { return bound->state != State::kCheckedOut; });
        continue;
      }

      // Preserve existing bindings when possible. If every available model is
      // bound, borrow one; a stream without a binding takes it over.
      Slot* slot =
          use_stream_affinity_ ? newest(State::kAvailable, true) : nullptr;
      if (slot == nullptr) {
        slot = newest(State::kAvailable);
      }
      if (slot != nullptr) {
        transition(*slot, State::kAvailable, State::kCheckedOut);
        if (affine) {
          slot->stream = stream;
        }
        return slot->model.get();
      }

      if (reclaim_finished_runs()) {
        continue;
      }
      Slot* oldest_in_flight = oldest(State::kInFlight);
      if (oldest_in_flight == nullptr) {
        // Every model is checked out.
        model_state_changed_.wait(lk);
        continue;
      }
      // Let's make the schedule simple first. We always wait on the oldest
      // in-flight run to complete.
      transition(*oldest_in_flight, State::kInFlight, State::kCheckedOut);
      lk.unlock();
      wait_for_device(*oldest_in_flight);
      lk.lock();
      transition(*oldest_in_flight, State::kCheckedOut, State::kAvailable);
      model_state_changed_.notify_all();
    }
  }

  // Waits, without the lock, for the device work of a slot this thread owns.
  // On a device failure the slot goes back to available, unbound.
  void wait_for_device(Slot& slot) {
    try {
      slot.model->wait_for_completion();
    } catch (...) {
      return_model_to_available(slot.model.get());
      throw;
    }
  }

  // Moves every finished in-flight run back to available, oldest first, so
  // the most recently launched one is reused first.
  bool reclaim_finished_runs() {
    std::vector<Slot*> in_flight;
    for (auto& slot : slots_) {
      if (slot.state == State::kInFlight) {
        in_flight.push_back(&slot);
      }
    }
    std::sort(
        in_flight.begin(), in_flight.end(), [](const Slot* a, const Slot* b) {
          return a->seq < b->seq;
        });
    bool reclaimed = false;
    for (Slot* slot : in_flight) {
      if (slot->model->is_finished()) {
        transition(*slot, State::kInFlight, State::kAvailable);
        reclaimed = true;
      }
    }
    if (reclaimed) {
      model_state_changed_.notify_all();
    }
    return reclaimed;
  }

  void return_model_to_available(Model* model) {
    {
      std::lock_guard lk(models_mutex_);
      Slot& slot = slot_of(model);
      // A failed execution invalidates its affinity so a subsequent call can
      // select another model instance.
      slot.stream = nullptr;
      transition(slot, State::kCheckedOut, State::kAvailable);
    }
    model_state_changed_.notify_all();
  }

  void return_model_in_flight(Model* model) {
    {
      std::lock_guard lk(models_mutex_);
      transition(slot_of(model), State::kCheckedOut, State::kInFlight);
    }
    model_state_changed_.notify_all();
  }

  void transition(Slot& slot, State from, State to) {
    AOTI_RUNTIME_CHECK(slot.state == from, "Invalid model pool transition");
    slot.state = to;
    slot.seq = next_seq_++;
  }

  bool quiescent() const {
    return std::all_of(slots_.begin(), slots_.end(), [](const Slot& slot) {
      return slot.state == State::kAvailable;
    });
  }

  Slot& slot_of(Model* model) {
    auto it =
        std::find_if(slots_.begin(), slots_.end(), [model](const Slot& s) {
          return s.model.get() == model;
        });
    AOTI_RUNTIME_CHECK(it != slots_.end(), "Model does not belong to the pool");
    return *it;
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

  Slot* oldest(State state) {
    Slot* result = nullptr;
    for (auto& slot : slots_) {
      if (slot.state == state &&
          (result == nullptr || slot.seq < result->seq)) {
        result = &slot;
      }
    }
    return result;
  }

  // One slot per model instance, which it owns, in the container's index
  // order. The vector is filled before the pool is shared, so slot pointers
  // stay valid.
  std::vector<Slot> slots_;
  uint64_t next_seq_{0};

  // Callers enabling affinity must keep bound streams alive to avoid a reused
  // device handle inheriting an old binding.
  bool use_stream_affinity_{false};

  std::mutex models_mutex_;
  std::condition_variable model_state_changed_;
};

} // namespace torch::aot_inductor
