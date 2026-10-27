#pragma once

#include <algorithm>
#include <condition_variable>
#include <deque>
#include <memory>
#include <mutex>
#include <unordered_map>
#include <unordered_set>
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
  struct StreamLock {
    StreamLock() = default;
    explicit StreamLock(const std::shared_ptr<std::mutex>& stream_mutex)
        : stream_mutex_(stream_mutex), lock_(*stream_mutex_) {}

    std::shared_ptr<std::mutex> stream_mutex_;
    std::unique_lock<std::mutex> lock_;
  };

  void reserve(size_t num_models) {
    models_.reserve(num_models);
    available_models_.reserve(num_models);
  }

  // Takes ownership of a model. The pool is filled before it is shared.
  Model& add_model(std::unique_ptr<Model> model) {
    models_.push_back(std::move(model));
    available_models_.push_back(models_.back().get());
    return *models_.back();
  }

  size_t size() const {
    return models_.size();
  }

  // The first-added model. Its identity is stable for the pool's lifetime.
  // The pool must not be empty.
  Model& primary_model() {
    return *models_.front();
  }

  const Model& primary_model() const {
    return *models_.front();
  }

  // The caller must keep runs from using the models meanwhile.
  template <typename Fn>
  void for_each_model(Fn&& fn) {
    for (auto& model : models_) {
      fn(*model);
    }
  }

  // Stream affinity is intended for controlled multi-stream benchmarking with
  // one model instance per long-lived stream. It deliberately waits for a
  // stream's bound model instead of using another instance, trading host-side
  // pipelining for stable stream-to-model assignment.
  void set_use_stream_affinity(bool use_stream_affinity) {
    std::lock_guard lk(models_mutex_);
    AOTI_RUNTIME_CHECK(
        available_models_.size() == models_.size() && pending_models_.empty() &&
            checked_out_models_.empty(),
        "Stream affinity can only be reconfigured while the model pool is "
        "quiescent");
#if defined(USE_CUDA) || defined(USE_XPU)
    use_stream_affinity_ = use_stream_affinity;
#else
    AOTI_RUNTIME_CHECK(
        !use_stream_affinity,
        "Stream affinity is only supported for device runtimes");
#endif
    stream_models_.clear();
    model_streams_.clear();
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
    const auto stream_it = stream_models_.find(stream);
    if (stream == nullptr || stream_it == stream_models_.end()) {
      return AOTI_STREAM_AFFINITY_UNBOUND;
    }
    for (size_t i = 0; i < models_.size(); ++i) {
      if (models_[i].get() == stream_it->second) {
        return static_cast<int64_t>(i);
      }
    }
    AOTI_RUNTIME_CHECK(false, "Stream affinity refers to an unknown model");
  }

  Model* get_available_model() {
    std::unique_lock lk(models_mutex_);
    return checkout_available_model(lk);
  }

  Model* get_available_model(Stream stream) {
    std::unique_lock lk(models_mutex_);
#if defined(USE_CUDA) || defined(USE_XPU)
    if (use_stream_affinity_ && stream != nullptr) {
      return get_stream_affine_model(stream, lk);
    }
#else
    (void)stream;
#endif
    return checkout_available_model(lk);
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

  void return_model_to_available(Model* model) {
    {
      std::lock_guard lk(models_mutex_);
      // A failed execution invalidates its affinity so a subsequent call can
      // select another model instance.
      unbind_model_from_stream(model);
      checked_out_models_.erase(model);
      available_models_.push_back(model);
    }
    model_state_changed_.notify_all();
  }

  void return_model_to_pending(Model* model) {
    {
      std::lock_guard lk(models_mutex_);
      checked_out_models_.erase(model);
      pending_models_.push_back(model);
    }
    model_state_changed_.notify_all();
  }

  // For run_single_threaded(), which bypasses the pool's locking entirely.
  Model* first_available_model_unsynchronized() {
    return available_models_[0];
  }

 private:
  Model* checkout_available_model(std::unique_lock<std::mutex>& lk) {
    return take_available_model(lk, /*prefer_unbound=*/use_stream_affinity_);
  }

  Model* take_available_model(
      std::unique_lock<std::mutex>& lk,
      bool prefer_unbound) {
    while (available_models_.empty()) {
      reclaim_finished_models(lk);
    }
    auto result_it = available_models_.end() - 1;
    if (prefer_unbound) {
      // Preserve existing bindings when possible. If every available model is
      // bound, the caller borrows one while its owning stream waits.
      for (auto it = available_models_.end();
           it != available_models_.begin();) {
        --it;
        if (model_streams_.find(*it) == model_streams_.end()) {
          result_it = it;
          break;
        }
      }
    }
    auto* result = *result_it;
    available_models_.erase(result_it);
    if (use_stream_affinity_) {
      checked_out_models_.insert(result);
    }
    return result;
  }

  Model* get_stream_affine_model(
      Stream stream,
      std::unique_lock<std::mutex>& lk) {
    while (true) {
      const auto stream_it = stream_models_.find(stream);
      if (stream_it == stream_models_.end()) {
        break;
      }

      auto* model = stream_it->second;
      const auto available_it =
          std::find(available_models_.begin(), available_models_.end(), model);
      if (available_it != available_models_.end()) {
        available_models_.erase(available_it);
        checked_out_models_.insert(model);
        return model;
      }

      const auto pending_it =
          std::find(pending_models_.begin(), pending_models_.end(), model);
      if (pending_it != pending_models_.end()) {
        pending_models_.erase(pending_it);
        checked_out_models_.insert(model);
        lk.unlock();
        try {
          model->wait_for_completion();
        } catch (...) {
          return_model_to_available(model);
          throw;
        }
        return model;
      }

      // A null-stream checkout or pool reclamation can temporarily borrow a
      // bound model. Same non-null stream calls serialize before this point.
      model_state_changed_.wait(lk, [this, model]() {
        return checked_out_models_.find(model) == checked_out_models_.end();
      });
    }

    // Bind an unbound model on a stream's first use. If all model instances are
    // already bound, reclaim and rebind one for an oversubscribed stream.
    auto* model = take_available_model(lk, /*prefer_unbound=*/true);
    bind_model_to_stream(model, stream);
    return model;
  }

  void reclaim_unused_stream_mutexes() {
    for (auto it = stream_mutexes_.begin(); it != stream_mutexes_.end();) {
      if (stream_models_.find(it->first) == stream_models_.end() &&
          it->second.use_count() == 1) {
        it = stream_mutexes_.erase(it);
      } else {
        ++it;
      }
    }
  }

  void unbind_model_from_stream(Model* model) {
    const auto model_it = model_streams_.find(model);
    if (model_it == model_streams_.end()) {
      return;
    }
    stream_models_.erase(model_it->second);
    model_streams_.erase(model_it);
  }

  void bind_model_to_stream(Model* model, Stream stream) {
    unbind_model_from_stream(model);

    const auto stream_it = stream_models_.find(stream);
    if (stream_it != stream_models_.end()) {
      model_streams_.erase(stream_it->second);
      stream_models_.erase(stream_it);
    }

    stream_models_.emplace(stream, model);
    model_streams_.emplace(model, stream);
  }

  void reclaim_finished_models(std::unique_lock<std::mutex>& lk) {
#ifdef __aarch64__
    // push finished model instances to the end of pending_models_
    auto it = std::partition(
        pending_models_.begin(), pending_models_.end(), [](Model* m) {
          return !m->is_finished();
        });
#else
    // push finished model instances to the end of pending_models_
    auto it = std::stable_partition(
        pending_models_.begin(), pending_models_.end(), [](Model* m) {
          return !m->is_finished();
        });
#endif

    if (it != pending_models_.end()) {
      // We have finished model instances that can be pushed into
      // available_models_ so that we don't have to block waiting on
      // model_state_changed_.
      available_models_.insert(
          available_models_.end(), it, pending_models_.end());
      pending_models_.erase(it, pending_models_.end());
      model_state_changed_.notify_all();
      return;
    }

    model_state_changed_.wait(lk, [this]() {
      return !available_models_.empty() || !pending_models_.empty();
    });
    if (!available_models_.empty()) {
      return;
    }
    // Let's make the schedule simple first. We always wait on the first
    // pending_models_ to be complete.
    auto* model = pending_models_.front();
    pending_models_.pop_front();
    if (use_stream_affinity_) {
      checked_out_models_.insert(model);
    }
    lk.unlock();
    try {
      model->wait_for_completion();
    } catch (...) {
      lk.lock();
      unbind_model_from_stream(model);
      checked_out_models_.erase(model);
      available_models_.push_back(model);
      model_state_changed_.notify_all();
      throw;
    }
    lk.lock();
    checked_out_models_.erase(model);
    available_models_.push_back(model);
    model_state_changed_.notify_all();
  }

  // Every model instance, in the container's index order.
  std::vector<std::unique_ptr<Model>> models_;

  // Holds the model instances available for inference. When stream affinity
  // is enabled, each model is in exactly one execution state under
  // models_mutex_: available here, pending device completion below, or checked
  // out by a host thread. Stream bindings are persistent preferences
  // independent of those states; borrowing a bound model preserves its
  // binding, while a failed run clears it so the next call may choose another
  // model.
  std::vector<Model*> available_models_;

  // Holds the model instances that have started running inference and can be
  // placed onto available_models_ upon their completion.
  std::deque<Model*> pending_models_;

  bool use_stream_affinity_{false};
  // Callers enabling affinity must keep bound streams alive to avoid a reused
  // device handle inheriting an old binding. Unbound mutexes are reclaimed
  // once no in-flight StreamLock shares ownership.
  std::unordered_map<Stream, Model*> stream_models_;
  std::unordered_map<Model*, Stream> model_streams_;
  std::unordered_map<Stream, std::shared_ptr<std::mutex>> stream_mutexes_;
  std::unordered_set<Model*> checked_out_models_;

  // Protects the model-state collections and affinity mappings.
  std::mutex models_mutex_;

  std::condition_variable model_state_changed_;
};

} // namespace torch::aot_inductor
