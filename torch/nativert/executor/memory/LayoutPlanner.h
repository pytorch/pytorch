#pragma once

#include <condition_variable>
#include <functional>
#include <thread>

#include <c10/macros/Macros.h>
#include <c10/util/CallOnce.h>
#include <c10/util/FbcodeMaps.h>
#include <c10/util/LeftRight.h>

#include <torch/nativert/executor/memory/AliasAnalyzer.h>
#include <torch/nativert/executor/memory/FunctionSchema.h>
#include <torch/nativert/executor/memory/LayoutPlannerAlgorithm.h>
#include <torch/nativert/executor/memory/LayoutPlannerSettings.h>
#include <torch/nativert/graph/Graph.h>

namespace {
constexpr inline std::memory_order drop_release(std::memory_order m) noexcept {
  return (
      m == std::memory_order_release
          ? std::memory_order_relaxed
          : ((m == std::memory_order_acq_rel || m == std::memory_order_seq_cst)
                 ? std::memory_order_acquire
                 : m));
}
// derivation of
// https://www.open-std.org/jtc1/sc22/wg21/docs/papers/2024/p0493r5.pdf
template <typename T>
void atomic_set_max(
    std::atomic<T>* pv,
    typename std::atomic<T>::value_type v,
    std::memory_order m = std::memory_order_seq_cst) noexcept {
  auto const mr = drop_release(m);
  auto t = (mr != m) ? pv->fetch_add(0, m) : pv->load(mr);
  while (std::max(v, t) != t) {
    if (pv->compare_exchange_weak(t, v, m, mr)) {
      return;
    }
  }
}
} // namespace

namespace torch::nativert {

class LayoutPlanner {
 public:
  explicit LayoutPlanner(
      const Graph& graph,
      const c10::FastMap<std::string /* target */, FunctionSchema>&
          kernelSchemas,
      const std::vector<bool>& persistentValues,
      const torch::nativert::LayoutPlannerSettings& settings);
#if !defined(_MSC_VER)
  TORCH_API // TODO Doesn't work on msvc.
#endif
      ~LayoutPlanner();

  LayoutPlanner(LayoutPlanner&& other) = delete;
  LayoutPlanner(const LayoutPlanner& other) = delete;
  LayoutPlanner operator=(LayoutPlanner&& other) = delete;
  LayoutPlanner& operator=(const LayoutPlanner& other) = delete;

  void start_worker_if_not_started();

  const std::vector<ValueId>& get_planned_values() const;
  const std::vector<ValueId>& get_unplanned_values() const;

#ifndef NDEBUG
  const AliasAnalyzer& get_alias_analyzer() const {
    return alias_analyzer_;
  }
#endif

  size_t num_values() const {
    return managed_values_.size();
  }

  bool is_managed(ValueId id) {
    TORCH_CHECK(static_cast<size_t>(id) < managed_values_.size());
    return managed_values_[id];
  }

  C10_ALWAYS_INLINE void try_update_max_size_at_index(size_t idx, size_t size) {
    atomic_set_max<size_t>(&planned_values_historical_max_nbytes_[idx], size);
  }

  C10_ALWAYS_INLINE
  void with_plan(std::function<void(const LayoutPlan&)>&& cb) {
    plan_.read(
        std::forward<std::function<void(const LayoutPlan&)>>(std::move(cb)));
  }

 private:
#ifdef LayoutPlannerTests_TEST_FRIENDS
  LayoutPlannerTests_TEST_FRIENDS;
#endif

  // we need some way of mapping graph values to other information
  // (e.g.,  allocation spec, max historical size)
  //
  // since there is a 1:1 mapping to/from each of these
  // we can create+initialize them here
  //
  // note: planning algorithms are allowed to change the ordering
  // of allocation specs -- so we pass the index of the spec during
  // it's insertion s.t., each execution frame can use it to
  // reference the correct associated max historical size / underlying
  // tensor value
  void initialize_vectors(
      c10::FastMap<const Value*, AllocationSpec> value_to_allocation_spec);

  void run_periodic(const std::function<void()>& f);
  void create_plan();

  // variables for managing the state of the
  // interval worker thread that refreshes
  // the plan
  std::condition_variable cv_;
  std::mutex mutex_;
  bool stopped_{false};
  std::thread worker_;

  std::vector<ValueId> unplanned_values_;

  std::vector<ValueId> planned_values_;
  std::vector<AllocationSpec> planned_allocation_specs_;
  std::vector<std::atomic_size_t> planned_values_historical_max_nbytes_;

  // managed_values_[value_id] == true
  // if graph.values()[value_id] has
  // an associated allocation spec
  std::vector<bool> managed_values_;

  LayoutPlannerAlgorithm* algorithm_;
  c10::LeftRight<LayoutPlan> plan_;

  c10::once_flag worker_once_flag_;

#ifndef NDEBUG
  AliasAnalyzer alias_analyzer_;
#endif
  torch::nativert::LayoutPlannerSettings settings_;
};

} // namespace torch::nativert
