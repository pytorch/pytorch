#pragma once

#include <algorithm>
#include <condition_variable>
#include <memory>
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
  struct ModelEntry;

 public:
  // Owns a checked-out model. A run that succeeds ends with mark_in_flight():
  // its device work may still be in flight, so the model is reused only once
  // that work has finished. Dropping the lease means the run threw: the model
  // goes back to the available pool at once, without waiting for completion
  // and with any stream binding cleared, so the next checkout may get it while
  // device work the failed run launched is still in flight.
  class ModelLease {
   public:
    ModelLease(const ModelLease&) = delete;
    ModelLease& operator=(const ModelLease&) = delete;
    ModelLease(ModelLease&& other) noexcept
        : pool_(other.pool_), entry_(other.entry_) {
      other.entry_ = nullptr;
    }
    ModelLease& operator=(ModelLease&&) = delete;

    ~ModelLease() {
      if (entry_ != nullptr) {
        pool_->return_model_to_available(*entry_);
      }
    }

    Model* operator->() const& {
      AOTI_RUNTIME_CHECK(
          entry_ != nullptr, "The lease no longer holds a model");
      return entry_->model.get();
    }
    // A temporary lease would return its model as available at the end of the
    // full expression, while the run's device work may still be in flight.
    Model* operator->() && = delete;
    Model* operator->() const&& = delete;

    void mark_in_flight() {
      AOTI_RUNTIME_CHECK(
          entry_ != nullptr, "The lease no longer holds a model");
      pool_->return_model_to_draining(*entry_);
      entry_ = nullptr;
    }

   private:
    friend class ModelPool;
    ModelLease(ModelPool& pool, ModelEntry& entry)
        : pool_(&pool), entry_(&entry) {}

    ModelPool* pool_;
    ModelEntry* entry_;
  };

  void reserve(size_t num_models) {
    entries_.reserve(num_models);
  }

  // Takes ownership of a model. The pool is filled before it is shared.
  Model& add_model(std::unique_ptr<Model> model) {
    entries_.push_back(ModelEntry{
        std::move(model), ModelState::kAvailable, nullptr, next_seq_++});
    return *entries_.back().model;
  }

  size_t size() const {
    return entries_.size();
  }

  // The first-added model. Its identity is stable for the pool's lifetime.
  // The pool must not be empty.
  Model& primary_model() {
    return *entries_.front().model;
  }

  const Model& primary_model() const {
    return *entries_.front().model;
  }

  // The caller must keep runs from using the models meanwhile.
  template <typename Fn>
  void for_each_model(Fn&& fn) {
    for (auto& entry : entries_) {
      fn(*entry.model);
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
            entries_.begin(),
            entries_.end(),
            [](const ModelEntry& entry) {
              return entry.state == ModelState::kAvailable;
            }),
        "Stream affinity can only be reconfigured while the model pool is "
        "quiescent");
#if defined(USE_CUDA) || defined(USE_XPU)
    use_stream_affinity_ = use_stream_affinity;
#else
    AOTI_RUNTIME_CHECK(
        !use_stream_affinity,
        "Stream affinity is only supported for device runtimes");
#endif
    for (auto& entry : entries_) {
      entry.stream = nullptr;
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
    ModelEntry* entry = bound_entry(stream);
    if (entry == nullptr) {
      return AOTI_STREAM_AFFINITY_UNBOUND;
    }
    return entry - entries_.data();
  }

  [[nodiscard]] ModelLease checkout() {
    return ModelLease(*this, get_available_entry());
  }

  [[nodiscard]] ModelLease checkout(Stream stream) {
    return ModelLease(*this, get_available_entry(stream));
  }

 private:
  // Every model is in exactly one of these states, guarded by models_mutex_.
  enum class ModelState : uint8_t {
    // Free to check out.
    kAvailable,
    // Held by one thread, which is running it or waiting for its device work.
    kHeld,
    // Held by no thread, but its last run reached the device and may still be
    // executing there, so it is reused only once that work finishes.
    kDraining,
  };

  struct ModelEntry {
    std::unique_ptr<Model> model;
    ModelState state;
    // The stream bound to this model when stream affinity is enabled, or
    // nullptr. A binding persists across states; borrowing a bound model keeps
    // it, while a failed run clears it so the next call may choose another
    // model.
    Stream stream;
    // When the entry entered its current state: the newest available model is
    // checked out first, and the oldest draining one is waited on first.
    uint64_t seq;
  };

  ModelEntry& get_available_entry() {
    std::unique_lock lk(models_mutex_);
    return checkout_available_entry(lk);
  }

  ModelEntry& get_available_entry(Stream stream) {
    std::unique_lock lk(models_mutex_);
#if defined(USE_CUDA) || defined(USE_XPU)
    if (use_stream_affinity_ && stream != nullptr) {
      return get_stream_affine_entry(stream, lk);
    }
#else
    (void)stream;
#endif
    return checkout_available_entry(lk);
  }

  void return_model_to_available(ModelEntry& entry) {
    {
      std::lock_guard lk(models_mutex_);
      // A failed execution invalidates its affinity so a subsequent call can
      // select another model instance.
      entry.stream = nullptr;
      transition(entry, ModelState::kHeld, ModelState::kAvailable);
    }
    model_state_changed_.notify_all();
  }

  void return_model_to_draining(ModelEntry& entry) {
    {
      std::lock_guard lk(models_mutex_);
      transition(entry, ModelState::kHeld, ModelState::kDraining);
    }
    model_state_changed_.notify_all();
  }

  void transition(ModelEntry& entry, ModelState from, ModelState to) {
    AOTI_RUNTIME_CHECK(entry.state == from, "Invalid model pool transition");
    entry.state = to;
    entry.seq = next_seq_++;
  }

  ModelEntry* bound_entry(Stream stream) {
    if (stream == nullptr) {
      return nullptr;
    }
    for (auto& entry : entries_) {
      if (entry.stream == stream) {
        return &entry;
      }
    }
    return nullptr;
  }

  ModelEntry* newest(ModelState state, bool unbound_only = false) {
    ModelEntry* result = nullptr;
    for (auto& entry : entries_) {
      if (entry.state == state && (!unbound_only || entry.stream == nullptr) &&
          (result == nullptr || entry.seq > result->seq)) {
        result = &entry;
      }
    }
    return result;
  }

  ModelEntry* oldest(ModelState state, uint64_t min_seq = 0) {
    ModelEntry* result = nullptr;
    for (auto& entry : entries_) {
      if (entry.state == state && entry.seq >= min_seq &&
          (result == nullptr || entry.seq < result->seq)) {
        result = &entry;
      }
    }
    return result;
  }

  ModelEntry& checkout_available_entry(std::unique_lock<std::mutex>& lk) {
    return take_available_entry(lk, /*prefer_unbound=*/use_stream_affinity_);
  }

  ModelEntry& take_available_entry(
      std::unique_lock<std::mutex>& lk,
      bool prefer_unbound) {
    while (oldest(ModelState::kAvailable) == nullptr) {
      reclaim_finished_models(lk);
    }
    // Preserve existing bindings when possible. If every available model is
    // bound, the caller borrows one while its owning stream waits.
    ModelEntry* entry =
        prefer_unbound ? newest(ModelState::kAvailable, true) : nullptr;
    if (entry == nullptr) {
      entry = newest(ModelState::kAvailable);
    }
    transition(*entry, ModelState::kAvailable, ModelState::kHeld);
    return *entry;
  }

  // Calls on one stream serialize through the model bound to it: a call waits
  // while that model is held, then for its last run to finish. A stream's first
  // call binds a model in the same critical section that picks it, before
  // anything releases the lock, so a second call on the stream finds the
  // binding and waits instead of binding another model.
  ModelEntry& get_stream_affine_entry(
      Stream stream,
      std::unique_lock<std::mutex>& lk) {
    while (true) {
      ModelEntry* entry = bound_entry(stream);
      if (entry == nullptr) {
        entry = entry_to_bind();
        if (entry == nullptr) {
          // Every model is held; wait for one to come back.
          model_state_changed_.wait(lk);
          continue;
        }
        entry->stream = stream;
      }

      if (entry->state == ModelState::kAvailable) {
        transition(*entry, ModelState::kAvailable, ModelState::kHeld);
        return *entry;
      }

      if (entry->state == ModelState::kDraining) {
        transition(*entry, ModelState::kDraining, ModelState::kHeld);
        lk.unlock();
        try {
          entry->model->wait_for_completion();
        } catch (...) {
          return_model_to_available(*entry);
          throw;
        }
        return *entry;
      }

      // Held by an earlier call on this stream, by a checkout without a stream,
      // or by a checkout waiting for its device work.
      model_state_changed_.wait(
          lk, [entry]() { return entry->state != ModelState::kHeld; });
    }
  }

  // The model for a stream's first call: an available one, after reclaiming
  // finished runs if none is, preferring one no other stream is bound to; else
  // the oldest draining one, which the call claims and waits on. nullptr if
  // every model is held. A model bound to another stream is rebound.
  ModelEntry* entry_to_bind() {
    if (oldest(ModelState::kAvailable) == nullptr) {
      reclaim_finished_runs();
    }
    ModelEntry* entry = newest(ModelState::kAvailable, /*unbound_only=*/true);
    if (entry == nullptr) {
      entry = newest(ModelState::kAvailable);
    }
    if (entry == nullptr) {
      entry = oldest(ModelState::kDraining);
    }
    return entry;
  }

  // Moves every finished run back to available, oldest first, so the most
  // recently launched one is reused first. This allocates nothing, and if
  // is_finished() throws, every entry is still in exactly one state.
  bool reclaim_finished_runs() {
    bool reclaimed = false;
    uint64_t min_seq = 0;
    while (ModelEntry* entry = oldest(ModelState::kDraining, min_seq)) {
      // Read before transition() gives the entry a new seq.
      min_seq = entry->seq + 1;
      if (entry->model->is_finished()) {
        transition(*entry, ModelState::kDraining, ModelState::kAvailable);
        reclaimed = true;
      }
    }
    if (reclaimed) {
      model_state_changed_.notify_all();
    }
    return reclaimed;
  }

  void reclaim_finished_models(std::unique_lock<std::mutex>& lk) {
    if (reclaim_finished_runs()) {
      return;
    }

    model_state_changed_.wait(lk, [this]() {
      return oldest(ModelState::kAvailable) != nullptr ||
          oldest(ModelState::kDraining) != nullptr;
    });
    if (oldest(ModelState::kAvailable) != nullptr) {
      return;
    }
    // Let's make the schedule simple first. We always wait on the oldest
    // draining model to be complete.
    ModelEntry* entry = oldest(ModelState::kDraining);
    transition(*entry, ModelState::kDraining, ModelState::kHeld);
    lk.unlock();
    try {
      entry->model->wait_for_completion();
    } catch (...) {
      lk.lock();
      entry->stream = nullptr;
      transition(*entry, ModelState::kHeld, ModelState::kAvailable);
      model_state_changed_.notify_all();
      throw;
    }
    lk.lock();
    transition(*entry, ModelState::kHeld, ModelState::kAvailable);
    model_state_changed_.notify_all();
  }

  // One entry per model instance, which it owns, in the container's index
  // order. The vector is filled before the pool is shared, so entry pointers
  // stay valid.
  std::vector<ModelEntry> entries_;
  uint64_t next_seq_{0};

  // Callers enabling affinity must keep bound streams alive to avoid a reused
  // device handle inheriting an old binding.
  bool use_stream_affinity_{false};

  // Protects the entries. Never held across a call that blocks.
  std::mutex models_mutex_;

  std::condition_variable model_state_changed_;
};

} // namespace torch::aot_inductor
