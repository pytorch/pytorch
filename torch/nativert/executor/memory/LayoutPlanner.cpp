#include <torch/nativert/executor/memory/LayoutPlanner.h>

#include <c10/util/CallOnce.h>
#include <c10/util/Enumerate.h>

#include <torch/nativert/executor/ExecutionPlanner.h>
#include <torch/nativert/executor/memory/AliasAnalyzer.h>
#include <torch/nativert/executor/memory/Bump.h>
#include <torch/nativert/executor/memory/DisjointStorageGroups.h>
#include <torch/nativert/executor/memory/GreedyBySize.h>

namespace torch::nativert {

LayoutPlanner::LayoutPlanner(
    const Graph& graph,
    const c10::FastMap<std::string /* target */, FunctionSchema>& kernelSchemas,
    const std::vector<bool>& persistentValues,
    const torch::nativert::LayoutPlannerSettings& settings)
    : managed_values_(graph.values().size()), settings_(settings) {
  auto value_to_allocation_spec = c10::FastMap<const Value*, AllocationSpec>{};
  auto alias_analyzer = AliasAnalyzer(graph, kernelSchemas);

  std::set<const Value*> input_values_set_;
  for (const auto* nv : graph.userInputs()) {
    if (nv->type() == Type::Kind::Tensor) {
      input_values_set_.insert(nv);
    }
  }

  const auto& tensor_meta = graph.tensorValuesMeta();

  for (auto&& [i, node] : at::enumerate(graph.nodes())) {
    // only manage out variant values
    if (const auto schemaIt = kernelSchemas.find(std::string(node.target()));
        schemaIt == kernelSchemas.end() ||
        schemaIt->second.kernel_kind() != OpKernelKind::kStaticDispatchKernel) {
      VLOG(1) << "not able to plan outputs for node " << node.target()
              << " as it is derived from an unsupported kernel kind.";
      continue;
    }

    for (const auto& output : node.outputs()) {
      // don't manage persistent values
      if (bool is_persistent = persistentValues[output->id()]; is_persistent) {
        VLOG(1)
            << "not planning " << output->name()
            << " as it is a persistent value (likely a weight or const-folded)";
        continue;
      }

      // only manage tensors
      if (bool is_tensor = output->type().kind() == Type::Kind::Tensor;
          !is_tensor) {
        VLOG(1) << "not planning " << output->name()
                << " as it is not a raw tensor. type: " << output->type();
        continue;
      }

      // output storage ownership must be given to the caller.
      if (const auto& values_associated_with_output =
              alias_analyzer.values_associated_with_output_storage();
          values_associated_with_output.find(output) !=
          values_associated_with_output.end()) {
        VLOG(1)
            << "not planning " << output->name()
            << " as its underlying storage may be associated with a graph output";
        continue;
      }

      // inputs are borrowed -- this is merely a sanity check
      if (input_values_set_.find(output) != input_values_set_.end()) {
        VLOG(1) << "not planning " << output->name()
                << " as it is a graph input that is borrowed from the user";
        continue;
      }

      // don't plan aliases -- they don't own the associated dataptr
      if (bool is_alias = alias_analyzer.is_alias(output); is_alias) {
        VLOG(1) << "not planning " << output->name() << " as it is an alias";
        continue;
      }

      if (bool is_consumed = output->users().size() > 0; !is_consumed) {
        VLOG(1) << "not planning " << output->name() << " as it has no users";
        continue;
      }

      if (auto meta_it = tensor_meta.find(std::string(output->name()));
          meta_it != tensor_meta.end()) {
        if (const auto& meta = meta_it->second; meta.device() == c10::kCPU) {
          auto& spec = value_to_allocation_spec[output];
          spec.lifetime = alias_analyzer.lifetime(output);
          managed_values_[output->id()] = true;
          continue;
        } else {
          VLOG(1) << "tensor " << output->name()
                  << " not placed on cpu so we cannot plan it";
        }
      } else /* possible if runtime pass didn't populate meta info */ {
        VLOG(1) << "tensor " << output->name() << " has no meta information";
      }

      managed_values_[output->id()] = true;
      value_to_allocation_spec[output].lifetime =
          alias_analyzer.lifetime(output);
    }
  }

  LOG(INFO) << "layout planner created with " << value_to_allocation_spec.size()
            << " values";

  switch (settings_.algorithmType()) {
    case torch::nativert::LayoutPlannerAlgorithmType::Bump: {
      algorithm_ = &BumpAllocationPlanner;
      break;
    }
    case torch::nativert::LayoutPlannerAlgorithmType::GreedyBySize: {
      algorithm_ = &GreedyBySizeAllocationPlanner;
      break;
    }
    case LayoutPlannerAlgorithmType::DisjointStorageGroups: {
      algorithm_ = &DisjointStorageGroupsPlanner;
      break;
    }
  }

  TORCH_CHECK_NOTNULL(algorithm_);

  initialize_vectors(value_to_allocation_spec);

  auto exec_planner = ExecutionPlanner{graph};
  auto p = exec_planner.createPlan();
  for (const auto& freeable : p->valuesToFree) {
    for (const auto v : freeable) {
      if (!is_managed(v)) {
        unplanned_values_.push_back(v);
      }
    }
  }
}

void LayoutPlanner::initialize_vectors(
    c10::FastMap<const Value*, AllocationSpec> value_to_allocation_spec) {
  size_t num_managed = value_to_allocation_spec.size();

  planned_values_.resize(num_managed);
  planned_allocation_specs_.resize(num_managed);
  planned_values_historical_max_nbytes_ =
      std::vector<std::atomic_size_t>(num_managed);

  size_t i = 0;
  for (auto& [v, spec] : value_to_allocation_spec) {
    TORCH_CHECK_LE(spec.lifetime.start, spec.lifetime.end);

    planned_values_[i] = v->id();
    planned_values_historical_max_nbytes_[i] = spec.size;
    planned_allocation_specs_[i] = std::move(spec);

    i++;
  }

  // for sanity in case anyone tries to use this after this method
  // is called with a bunch of junk (i.e., moved specs) in it
  value_to_allocation_spec.clear();
}

const std::vector<ValueId>& LayoutPlanner::get_planned_values() const {
  return planned_values_;
}

const std::vector<ValueId>& LayoutPlanner::get_unplanned_values() const {
  return unplanned_values_;
}

void LayoutPlanner::start_worker_if_not_started() {
  static c10::once_flag flag;
  c10::call_once(flag, [&]() {
    // make sure plan is populated by the time this
    // returns for the first time :P
    create_plan();
    worker_ = std::thread([this]() {
      run_periodic(std::bind(&LayoutPlanner::create_plan, this));
    });
  });
}

LayoutPlanner::~LayoutPlanner() {
  {
    std::unique_lock<std::mutex> l(mutex_);
    stopped_ = true;
  }
  cv_.notify_one();
  if (worker_.joinable()) {
    worker_.join();
  }
}

void LayoutPlanner::run_periodic(const std::function<void()>& f) {
  std::unique_lock<std::mutex> l(mutex_);
  while (!cv_.wait_for(
      l, settings_.planningInterval(), [&]() { return stopped_; })) {
    f();
  }
}

void LayoutPlanner::create_plan() {
  // update spec sizes to use historical maximums set
  // by execution frames before creating the new plan
  for (const auto i : c10::irange(planned_allocation_specs_.size())) {
    auto& spec = planned_allocation_specs_[i];
    spec.size = planned_values_historical_max_nbytes_[i].load(
        std::memory_order_relaxed);
  }
  plan_.write([p_new = (*algorithm_)(planned_allocation_specs_)](
                  LayoutPlan& plan) { plan = p_new; });
}

} // namespace torch::nativert
