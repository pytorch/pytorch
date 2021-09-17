#pragma once

#include <ATen/ATen.h>


namespace torch { namespace autograd {

// [ Using ForwardGrad ]
// ForwardGrad needs to be a shared_ptr to satisfy constraints of its inner design. But
// this shared_ptr must be uniquely associated with the object that stores it (as of
// writing, either AutogradMeta or SavedVariable). This object is called the "owning object"
// in the discussions below. This owning object must call `ForwardGrad::clear()` when it
// is destroyed to ensure that the ForwardGrad is properly de-allocated.

struct ForwardGrad;

// This file contains two classes that are used to store forward AD gradients and
// ensure that they are scoped properly.
// Because forward AD runs concurrently with the evaluation of the function, we need
// a mechanism to separate different forward AD invocations and be able to compute the
// right gradients. We model such invocations as levels here.
// The particular scoping issue mentioned above has two main drivers:
//   - Ensure that we can conveniently use forward AD within a high level API without
//     leaking the forward AD states outside.
//   - Ensure that we can keep the level that we expose to the user API simple (an integer
//     that represents the nesting depth) while avoiding confusions when the level index
//     is re-used.

// The important external APIs from this file are:
//   - ForwardADLevel::get_next_idx() that can be used to enter a new level and get its index
//   - ForwardADLevel::release_idx() that can be used to exit a given level.
//   - ForwardGrad() can be used to store a given forward gradient that will handle the level
//     tracking automatically.

// The basic implementation strategy is as follows:
// Every tensor has a ForwardGrad, maintaining a map from levels to tangents.
// ForwardGrad is responsible for registering itself to the appropriate ForwardADLevel when a new
// tangent is added to it via ForwardGrad::set_value and to un-register itself from this same level
// if that tangent is removed via ForwardGrad::reset.
// The ForwardADLevel is created when a new level is entered via ForwardADLevel::get_next_idx.
// A reference to the new ForwardADLevel is stored into a global (for the whole process) vector that
// ensure it can be accessed via ForwardADLevel::get_by_idx. This reference is deleted when the index is
// released by the user when calling ForwardADLevel::release_idx.
// When it is destructed, the ForwardADLevel is responsible for clearing all the tangents for its
// level stored in all the ForwardGrad that registered with it.
//
// This process-wide level design, compared to a thread local one, allows us to use very simple user facing
// handle for the level (an int) while enabling cross-thread forward AD.
// The only required synchronization for the user is when entering and exiting the levels.
// Some discussion on alternative design is in https://github.com/pytorch/pytorch/pull/49097#discussion_r543716453
// and can be refined in the future.

// Correctness of concurrency:
// Each class uses its own lock when reading or modifying internal storages. This allows in particular
// to safely remove tangents from ForwardGrad when the ForwardADLevel is being exited.
// We ensure no deadlock by ensuring that a methods never calls into another class's method while
// the local class's lock is held except in one single case: calling from ForwardADLevel's destructor
// into ForwardGrad::reset with update_level=false.

// The lifetime of these objects is as follows:
// The ForwardADLevel can be in three states:
//      - Initialized: where one of its reference is held by the global vector and there may be more
//        references held by temporary variables in ForwardGrad's methods.
//      - About to be destructed: where "release_idx" has been called and the only reason for the
//        ForwardADLevel not to be destructed right away is that some methods in ForwardGrad have
//        owning reference to it. This is done so that a ForwardADLevel can never be destructed when
//        a ForwardGrad is registered with it and in the process of adding something to its internal state.
//      - Being destructed: Here the ForwardADLevel is not referenced anymore and can be safely reset
//        all of the ForwardGrad. Note that we can have more than one reset being called here (which is ok)
//        but we are guaranteed that there is at least one.
// The ForwardGrad is simpler as there is no intermediary state and no special destructor for. The logic to
// unregister it from the different ForwardADLevel is done when the owning object (AutogradMeta or
// SavedVariable) is being destroyed.

// Other considered design:
// To avoid having the ForwardGrad::clear, we considered storing weak_ptr inside the ForwardADLevel. While this
// would work, it would mean that the set inside the ForwardADLevel would only grow unless we do an
// expensive linear scan to remove all the dangling weak pointers. Hence this approach was not used.

// Data structures in this file are optimized for this maximum number of levels.
// The number of levels corresponds to the degree of the gradient being
// computed using forward AD and we don't expect more than second order gradients
// to be common.
#define EXPECTED_MAX_LEVEL 2

struct TORCH_API ForwardADLevel {
  ForwardADLevel(uint64_t idx) : idx_(idx) {}
  ~ForwardADLevel();

  static uint64_t get_next_idx();
  static void release_idx(uint64_t idx);
  static std::shared_ptr<ForwardADLevel> get_by_idx(uint64_t idx);
  static std::shared_ptr<ForwardADLevel> try_get_by_idx(uint64_t idx);

  void erase(const std::shared_ptr<ForwardGrad>& grad) {
    std::lock_guard<std::mutex> lock(mutex_);
    grads_.erase(grad);
    }

    void insert(const std::shared_ptr<ForwardGrad>& grad) {
        std::lock_guard<std::mutex> lock(mutex_);
        grads_.insert(grad);
    }

private:
    std::unordered_set<std::shared_ptr<ForwardGrad>> grads_;
    std::mutex mutex_;
    uint64_t idx_;

};

struct TORCH_API ForwardGrad : std::enable_shared_from_this<ForwardGrad> {
  ForwardGrad() = default;

  // This function must only be called when AutogradMeta or SavedVariable is
  // being destructed as it ensures that:
  //   - The only (potential) other references to this ForwardGrad are the
  //     different level it is registered to
  //   - No other thread will try to call `set_value` or `value` ever from now
  //   on
  //   - Any of the ForwardADLevel that this ForwardGrad is registered with
  //   might
  //     call `reset` at any point during this function
  void clear() {
    c10::SmallVector<uint64_t, EXPECTED_MAX_LEVEL> levels_idx;

    {
      std::lock_guard<std::mutex> lock(mutex_);
      for (auto& c : content_) {
        levels_idx.push_back(c.first);
      }
    }

    for (auto l_idx : levels_idx) {
      // Use "try" version here as another thread might have deleted this
      // level before we got here
      // This is an owning reference as we want to keep the level alive
      // until we successfully unregister ourselves
      auto level = ForwardADLevel::try_get_by_idx(l_idx);
      if (level) {
        level->erase(shared_from_this());
      }
    }
    }

    void set_value(const at::Tensor& value, uint64_t level) {
        // Owning reference to ensure the forward_level is not destroyed
        // while we are updating our internal state
        auto forward_level = ForwardADLevel::get_by_idx(level);
        forward_level->insert(shared_from_this());

        std::lock_guard<std::mutex> lock(mutex_);
        content_.insert({level, value});
    }

    // This function removes the tangent for a given level from this ForwardGrad
    // Use the update_level flag to disable notifying the level about this reset
    // This flag is most notably used by the ForwardADLevel destructor.
    void reset(uint64_t level, bool update_level=true) {
        if (update_level) {
            ForwardADLevel::get_by_idx(level)->erase(shared_from_this());
        }

        std::lock_guard<std::mutex> lock(mutex_);
        content_.erase(level);
    }

    const at::Tensor& value(uint64_t level) const;

    bool contains(uint64_t level) {
        std::lock_guard<std::mutex> lock(mutex_);
        return content_.count(level) > 0;
    }

    bool empty() const {
        return content_.empty();
    }

    static const at::Tensor& undef_grad();


private:
    // TODO(albanD): replace this with a SmallVector
    std::unordered_map<uint64_t, at::Tensor> content_;
    mutable std::mutex mutex_;

};

}} // namespace torch::autograd
