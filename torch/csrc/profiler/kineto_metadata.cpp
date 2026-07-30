#include <torch/csrc/profiler/kineto_metadata.h>

#include <algorithm>
#include <cstdint>
#include <optional>
#include <string>
#include <type_traits>
#include <vector>

#include <c10/util/Logging.h>
#include <c10/util/irange.h>
#include <c10/util/overloaded.h>
#include <fmt/format.h>

#include <torch/csrc/profiler/kineto_shim.h>
#include <torch/csrc/profiler/util.h>

#ifdef USE_KINETO
#include <MetadataFieldCatalog.h>
#include <libkineto.h>
#endif // USE_KINETO

namespace torch::autograd::profiler {

using torch::profiler::impl::concreteInputsToStrList;
using torch::profiler::impl::EventType;
using torch::profiler::impl::ExtraFields;
using torch::profiler::impl::get_record_concrete_inputs_enabled;
using torch::profiler::impl::ivalueListToStr;
using torch::profiler::impl::ivalueToStr;
using torch::profiler::impl::joinStacks;
using torch::profiler::impl::parseArgData;
using torch::profiler::impl::PyExtraFieldsBase;
using torch::profiler::impl::Result;
using torch::profiler::impl::shapesToInputShapes;
using torch::profiler::impl::variantShapesTruncated;

#ifdef USE_KINETO

namespace fields = libkineto::GenericMetadataFields;
namespace collective_fields = libkineto::CollectiveMetadataFields;

namespace {

template <typename>
inline constexpr bool always_false_v = false;

struct MetadataBase {
  /* implicit */ MetadataBase(const std::shared_ptr<Result>& result)
      : kinetoActivity_{result->kineto_activity_} {
    if (std::holds_alternative<ExtraFields<EventType::Kineto>>(
            result->extra_fields_)) {
      // In order to add metadata we have to downcast from
      // `libkineto::ITraceActivity` to `libkineto::GenericTraceActivity`. We
      // know that all activities provided by PyTorch are of the correct type,
      // however Kineto profilers can (and do) add events that inherit directly
      // from ITraceActivity. As a result, any Result which was constructed from
      // an event that Kineto provided is unsafe to cast.
      if (!(SOFT_ASSERT(!hasKinetoActivity()))) {
        result->kineto_activity_ = nullptr;
      }
      kinetoActivity_ = result->kineto_activity_;
    }
  }

  // Stringly-typed metadata (dynamic keys, string values). Kept for fields
  // whose key or type is only known at runtime.
  void addMetadata(
      const std::string& key,
      const std::string& value,
      bool quote = false) {
    if (kinetoActivity_ && !value.empty() && value != "\"\"") {
      torch::profiler::impl::kineto::addMetadata(
          mutableActivity(), key, value, quote);
    }
  }

  // Typed metadata: the value keeps its declared type all the way into
  // libkineto instead of being pre-stringified. Empty string values are
  // dropped to match the stringly-typed overload above.
  template <typename T>
  void addMetadata(const libkineto::MetadataField<T>& field, const T& value) {
    if (!kinetoActivity_) {
      return;
    }
    if constexpr (std::is_same_v<T, std::string>) {
      if (value.empty()) {
        return;
      }
    }
    mutableActivity()->addMetadata(field, value);
  }

  template <typename T>
  void addCollectiveMetadata(
      const libkineto::MetadataField<T>& field,
      const torch::profiler::impl::collective_meta_t& metadata) {
    const auto it = metadata.find(std::string{field.name});
    if (it == metadata.end()) {
      return;
    }
    if constexpr (std::is_same_v<T, std::string>) {
      addMetadata(field, it->second.toStringRef());
    } else if constexpr (std::is_same_v<T, int64_t>) {
      addMetadata(field, it->second.toInt());
    } else if constexpr (std::is_same_v<T, uint64_t>) {
      addMetadata(field, it->second.toUInt());
    } else if constexpr (std::is_same_v<T, bool>) {
      addMetadata(field, it->second.toBool());
    } else {
      static_assert(always_false_v<T>, "Unsupported collective metadata type");
    }
  }

  bool hasKinetoActivity() const {
    return kinetoActivity_ != nullptr;
  }

 private:
  torch::profiler::impl::kineto::activity_t* mutableActivity() const {
    // NOLINTNEXTLINE(cppcoreguidelines-pro-type-const-cast)
    return const_cast<torch::profiler::impl::kineto::activity_t*>(
        kinetoActivity_);
  }

  const torch::profiler::impl::kineto::activity_t* kinetoActivity_{nullptr};
};

struct AddTensorboardFields : public MetadataBase {
  AddTensorboardFields(
      const std::shared_ptr<Result>& result,
      c10::ArrayRef<std::string> module_hierarchy,
      c10::ArrayRef<std::string> stack)
      : MetadataBase(result) {
    result->visit(*this);
    addMetadata(
        fields::kModuleHierarchy, joinStacks(module_hierarchy.vec(), "."));
    addMetadata(fields::kCallStack, joinStacks(stack.vec(), ";"));

    result->visit_if_base<PyExtraFieldsBase>([&, this](const auto& i) -> void {
      this->addMetadata(fields::kPythonId, static_cast<uint64_t>(i.id_));

      std::optional<uint64_t> parent_id;
      std::shared_ptr<Result> parent = result->parent_.lock();
      while (parent && !parent_id.has_value()) {
        parent->visit_if_base<PyExtraFieldsBase>(
            [&](const auto& j) { parent_id = static_cast<uint64_t>(j.id_); });
        parent = parent->parent_.lock();
      }
      if (parent_id.has_value()) {
        this->addMetadata(fields::kPythonParentId, *parent_id);
      }
      if (i.caller_.line_no_ > 0) {
        this->addMetadata(
            fields::kCallFrom,
            fmt::format(
                "{}:{}", i.caller_.filename_.str(), i.caller_.line_no_));
      }
    });
  }

  void operator()(const ExtraFields<EventType::PyCall>& py_call) {
    if (py_call.module_.has_value()) {
      addMetadata(
          fields::kPythonModuleId, static_cast<uint64_t>(py_call.module_->id_));
    }
  }

  template <typename T>
  void operator()(const T& /*unused*/) {}
};

struct AddGenericMetadata : public MetadataBase {
  AddGenericMetadata(
      std::shared_ptr<Result>& result,
      const torch::profiler::impl::ProfilerConfig* config)
      : MetadataBase(result), config_(config) {
    result->visit(*this);
    if (config->experimental_config.verbose) {
      result->visit_if_base<PyExtraFieldsBase>(
          [&, this](const auto& i) -> void {
            this->addMetadata(
                fields::kPythonThread, static_cast<uint64_t>(i.python_tid_));
          });
    }
  }

  void operator()(ExtraFields<EventType::TorchOp>& op_event) {
    const auto arg_data =
        parseArgData(op_event.inputs_, op_event.concrete_inputs_);

    if (arg_data.hasData) {
      if (get_record_concrete_inputs_enabled()) {
        addMetadata(
            fields::kInputDims, variantShapesTruncated(arg_data.shapes));
      } else {
        addMetadata(
            fields::kInputDims,
            shapesToInputShapes(arg_data.shapesForKinetoEvent));
      }
      addMetadata(
          fields::kInputStrides, variantShapesTruncated(arg_data.strides));
      addMetadata(fields::kInputType, arg_data.dtypes);
      if (!arg_data.concreteInputs.empty()) {
        addMetadata(
            fields::kConcreteInputs,
            concreteInputsToStrList(arg_data.concreteInputs));
      }
    }

    // Add metadata for kwinputs if exist
    for (const auto& [key, val] : op_event.kwinputs_) {
      if (key == "stream" && !val.isInt()) {
        LOG(WARNING) << "Inputted stream is not an int for op: "
                     << op_event.name_ << " skipping";
        continue;
      }

      // Until needed, let's limit the kwargs to only ints, doubles, strings,
      // bools, and list of strings
      bool isValidType =
          val.isInt() || val.isDouble() || val.isString() || val.isBool();
      bool isStringList = false;

      if (!isValidType && val.isList()) {
        // Check if it's a list of strings
        auto list = val.toListRef();
        isStringList = std::ranges::all_of(
            list, [](const c10::IValue& item) { return item.isString(); });
      }

      if (!isValidType && !isStringList) {
        LOG(WARNING)
            << "Inputted kwarg: " << key
            << " is not an int, double, string, bool, or list of strings for op: "
            << op_event.name_ << " skipping";
        continue;
      }

      if (isStringList) {
        // For list of strings, use ivalueListToStr
        auto list = val.toListRef();
        std::vector<c10::IValue> stringList(list.begin(), list.end());
        addMetadata(key, ivalueListToStr(stringList));
      } else {
        bool isString = val.isString();
        addMetadata(key, ivalueToStr(val, isString));
      }
    }
    const auto& metadata = op_event.collective_meta_;
    if (!metadata.empty()) {
      addCollectiveMetadata(collective_fields::kCollectiveName, metadata);
      addCollectiveMetadata(collective_fields::kDtype, metadata);
      // These are manually set for now because Kineto uses getMetadataValue to
      // fetch metadata by key. Adding by key enforces that the MetadataField
      // key matches. Soon NCCL metadata should be written by a visitor
      // pattern and we can remove this.
      addCollectiveMetadata(collective_fields::kInMsgNelems, metadata);
      addCollectiveMetadata(collective_fields::kOutMsgNelems, metadata);
      addCollectiveMetadata(collective_fields::kInSplit, metadata);
      addCollectiveMetadata(collective_fields::kOutSplit, metadata);
      addCollectiveMetadata(collective_fields::kGlobalRankStart, metadata);
      addCollectiveMetadata(collective_fields::kGlobalRankStride, metadata);
      addCollectiveMetadata(collective_fields::kGroupSize, metadata);
      addCollectiveMetadata(collective_fields::kProcessGroupName, metadata);
      addCollectiveMetadata(collective_fields::kProcessGroupDesc, metadata);
      addCollectiveMetadata(collective_fields::kGroupRanks, metadata);
      addCollectiveMetadata(collective_fields::kRank, metadata);
      addCollectiveMetadata(collective_fields::kP2pSrc, metadata);
      addCollectiveMetadata(collective_fields::kP2pDst, metadata);
      addCollectiveMetadata(collective_fields::kSeqNum, metadata);
      addCollectiveMetadata(collective_fields::kInTensorsStart, metadata);
      addCollectiveMetadata(collective_fields::kOutTensorsStart, metadata);
      addCollectiveMetadata(collective_fields::kIsAsynchronizedOp, metadata);
      addCollectiveMetadata(collective_fields::kCommsId, metadata);
    }

    if (config_ && !config_->experimental_config.performance_events.empty()) {
      auto& event_names = config_->experimental_config.performance_events;
      for (const auto i : c10::irange(op_event.perf_event_counters_->size())) {
        addMetadata(
            event_names[i],
            std::to_string((*op_event.perf_event_counters_)[i]));
      }
    }

    // add information about an associated forward op, if a sequence number
    // is available (e.g. during training)
    if (op_event.sequence_number_ >= 0) {
      addMetadata(fields::kFwdThreadId, op_event.forward_tid_);
      addMetadata(fields::kSequenceNumber, op_event.sequence_number_);
    }
    addMetadata(fields::kRecordFunctionId, op_event.record_function_id_);
  }

  void operator()(ExtraFields<EventType::Backend>& backend_event) {
    if (!backend_event.backend_.empty()) {
      addMetadata(fields::kBackend, backend_event.backend_);
    }
  }

  void operator()(const ExtraFields<EventType::Allocation>& alloc) {
    addMetadata(
        fields::kDeviceType,
        static_cast<int64_t>(static_cast<int8_t>(alloc.device_type_)));
    addMetadata(fields::kDeviceId, static_cast<int64_t>(alloc.device_index_));
    addMetadata(
        fields::kAddr,
        static_cast<uint64_t>(reinterpret_cast<uintptr_t>(alloc.ptr_)));
    addMetadata(fields::kBytes, alloc.alloc_size_);
    addMetadata(
        fields::kTotalAllocated, static_cast<uint64_t>(alloc.total_allocated_));
    addMetadata(
        fields::kTotalReserved, static_cast<uint64_t>(alloc.total_reserved_));
  }

  void operator()(const ExtraFields<EventType::OutOfMemory>& alloc) {
    addMetadata(
        fields::kDeviceType,
        static_cast<int64_t>(static_cast<int8_t>(alloc.device_type_)));
    addMetadata(fields::kDeviceId, static_cast<int64_t>(alloc.device_index_));
    addMetadata(fields::kBytes, alloc.alloc_size_);
    addMetadata(
        fields::kTotalAllocated, static_cast<uint64_t>(alloc.total_allocated_));
    addMetadata(
        fields::kTotalReserved, static_cast<uint64_t>(alloc.total_reserved_));
  }

  template <typename T>
  void operator()(const T& /*unused*/) {}

 private:
  /* To get names of the performance events */
  const torch::profiler::impl::ProfilerConfig* config_;
};

} // namespace

void addTensorboardFields(
    const std::shared_ptr<Result>& result,
    c10::ArrayRef<std::string> module_hierarchy,
    c10::ArrayRef<std::string> stack) {
  AddTensorboardFields{result, module_hierarchy, stack};
}

void addGenericMetadata(
    std::shared_ptr<Result>& result,
    const torch::profiler::impl::ProfilerConfig* config) {
  AddGenericMetadata{result, config};
}

#else // USE_KINETO

void addTensorboardFields(
    const std::shared_ptr<Result>& /*result*/,
    c10::ArrayRef<std::string> /*module_hierarchy*/,
    c10::ArrayRef<std::string> /*stack*/) {}

void addGenericMetadata(
    std::shared_ptr<Result>& /*result*/,
    const torch::profiler::impl::ProfilerConfig* /*config*/) {}

#endif // USE_KINETO

void addTraceMetadata(
    std::vector<std::shared_ptr<Result>>& events,
    [[maybe_unused]] const torch::profiler::impl::ProfilerConfig& config,
    int64_t trace_end_ns) {
  for (auto& e : events) {
    // Unfinished events automatically have end time set to trace end time
    if (!e->finished_) {
      e->visit(c10::overloaded(
          [trace_end_ns](ExtraFields<EventType::TorchOp>& i) {
            i.end_time_ns_ = trace_end_ns;
          },
          [](auto&) {}));
    }

    if (!e->kineto_activity_) {
      continue;
    }
#ifdef USE_KINETO
    AddGenericMetadata add_generic(e, &config);

    // Subset of AddTensorboardFields that doesn't require KinetoEvent or
    // parent chain (no python_stack_, no Python parent id).
    MetadataBase tb(e);
    e->visit(c10::overloaded(
        [&](const ExtraFields<EventType::TorchOp>& i) {
          tb.addMetadata(
              fields::kModuleHierarchy, joinStacks(i.jit_modules_, "."));
          tb.addMetadata(fields::kCallStack, joinStacks(i.jit_stack_, ";"));
        },
        [&](const ExtraFields<EventType::Backend>& i) {
          tb.addMetadata(
              fields::kModuleHierarchy, joinStacks(i.jit_modules_, "."));
          tb.addMetadata(fields::kCallStack, joinStacks(i.jit_stack_, ";"));
        },
        [](const auto&) {}));
    e->visit_if_base<PyExtraFieldsBase>([&](const auto& i) {
      tb.addMetadata(fields::kPythonId, static_cast<uint64_t>(i.id_));
    });
    e->visit(c10::overloaded(
        [&](const ExtraFields<EventType::PyCall>& py_call) {
          if (py_call.module_.has_value()) {
            tb.addMetadata(
                fields::kPythonModuleId,
                static_cast<uint64_t>(py_call.module_->id_));
          }
        },
        [](const auto&) {}));

    e->kineto_activity_ = nullptr;
#endif // USE_KINETO
  }
}

} // namespace torch::autograd::profiler
