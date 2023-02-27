#include <instrumentation.h>
#include <python_frontend/fusion_cache.h>
#include <python_frontend/fusion_definition.h>
#include <utils.h>

// Require namespace for perf scope instrumentation
using namespace nvfuser::inst;

namespace nvfuser::python_frontend {

const char* dtypeToPyString(PrimDataType t) {
  switch (t) {
    case DataType::Bool:
      return "DataType.Bool";
    case DataType::Double:
      return "DataType.Double";
    case DataType::Float:
      return "DataType.Float";
    case DataType::Half:
      return "DataType.Half";
    case DataType::BFloat16:
      return "DataType.Bfloat16";
    case DataType::Int:
      return "DataType.Int";
    case DataType::Int32:
      return "DataType.Int32";
    case DataType::ComplexFloat:
      return "DataType.ComplexFloat";
    case DataType::ComplexDouble:
      return "DataType.ComplexDouble";
    case DataType::Null:
      return "DataType.Null";
    default:
      break;
  }
  TORCH_INTERNAL_ASSERT(false, "No string found for data type.");
  return nullptr;
}

bool State::operator==(const State& other) const {
  return (index == other.index) && (stype == other.stype);
}

bool State::operator!=(const State& other) const {
  return (index != other.index) || (stype != other.stype);
}

// Generalized printing of State
std::ostream& operator<<(std::ostream& os, const State& state) {
  if (state.stype == StateType::Scalar) {
    os << "S";
  } else if (state.stype == StateType::Tensor) {
    os << "T";
  } else if (state.stype == StateType::None) {
    os << "None";
  } else {
    TORCH_INTERNAL_ASSERT(false, "Unsupported StateType");
  }
  os << state.index;
  return os;
}

FusionDefinition::FusionDefinition(c10::optional<size_t> id, size_t max_length)
    : FusionState(),
      max_length_(max_length),
      fusion_id_(id),
      fusion_cache_(FusionCache::get()),
      recording_state_(),
      prev_fusion_(nullptr),
      user_sched_(nullptr),
      ops(this),
      sched(this) {}

FusionCache* FusionDefinition::fusionCache() const {
  TORCH_INTERNAL_ASSERT(
      fusion_cache_ != nullptr, "FusionCache pointer is null!");
  return fusion_cache_;
}

FusionDefinition* FusionDefinition::setupDefinition() {
  TORCH_CHECK(max_length_ > 0, "Can't make a FusionDefinition with 0 records!");
  TORCH_CHECK(!id().has_value(), "Fusion Schedule is already found!");
  fusionCache()->resetTriePtr();
  return this;
}

void FusionDefinition::finalizeDefinition() {
  FUSER_PERF_SCOPE("FusionDefinition::finalizeDefinition");
  auto cache_entry = fusionCache()->queryChildren(end_record_.get());
  if (!cache_entry.has_value()) {
    if (isDebugDumpEnabled(DebugDumpOption::PythonFrontendDebug)) {
      std::cout << "\nFusionDefinition: Terminal Node not found.\n";
    }
    fusion_id_ = fusionCache()->createChild(end_record_.get());
    TORCH_CHECK(id().has_value(), "Invalid fusion id!");
    fusionCache()->traverseTrie(end_record_.get());

    if (isDebugDumpEnabled(DebugDumpOption::PythonDefinition)) {
      print(std::cout);
    }

    buildFusionIr(preschedFusion());

    if (isDebugDumpEnabled(DebugDumpOption::FusionIrPresched)) {
      printIr();
    }
  } else {
    if (isDebugDumpEnabled(DebugDumpOption::PythonFrontendDebug)) {
      std::cout << "\nFusionDefinition: Terminal Node found!\n";
    }
    fusion_id_ = c10::optional<size_t>(cache_entry.value()->fusion_id);
    fusionCache()->traverseTrie(end_record_.get());
  }
}

void FusionDefinition::setupSchedule(const at::ArrayRef<c10::IValue>& inputs) {
  FUSER_PERF_SCOPE("FusionDefinition::setupSchedule");
  TORCH_CHECK(id().has_value(), "FusionDefinition definition does not exist!");
  auto& scheds = fusionCache()->queryFusionSchedules(id().value());
  auto device = getCommonDeviceCUDA(inputs);
  TORCH_CHECK(
      inputs.size() == 0 || device > -1,
      "Inputs are not all on the same device!");
  TORCH_CHECK(user_sched_ == nullptr, "Expected User Scheduler to be null!");
  user_sched_ = fusionCache()->createUserSchedule(scheds, inputs, device);

  // Building a new Fusion container for scheduling with definition such that
  // the definition's tensor data members refer to the corresponding IR objects
  // needed for scheduling. A simple copy of the container would mean the data
  // members that represent tensors would refer to the IR objects in the
  // original and not the copy needed for scheduling.
  buildFusionIr(user_sched_->schedule.get());

  // Manually setting the fusion guard as there is not a good way of using a
  // guard in a local scope across the schedule function
  prev_fusion_ = FusionGuard::getCurFusion();
  FusionGuard::setCurFusion(user_sched_->schedule.get());
}
void FusionDefinition::finalizeSchedule(
    const at::ArrayRef<c10::IValue>& inputs) {
  FUSER_PERF_SCOPE("FusionDefinition::finalizeSchedule");
  FusionGuard::setCurFusion(prev_fusion_);
  prev_fusion_ = nullptr;

  user_sched_->executor->compileFusion(user_sched_->schedule.get(), inputs);
  user_sched_ = nullptr;
}

void FusionDefinition::print(std::ostream& os) const {
  if (id().has_value()) {
    os << "\ndef nvfuser_fusion_id" << id().value();
  } else {
    os << "\ndef nvfuser_incomplete_fusion";
  }
  os << "(fd : FusionDefinition) -> None :\n";
  os << std::dec;
  for (auto& rec : recording_) {
    os << "    ";
    rec->print(os);
    os << "\n";
  }
  os << std::endl;
}

std::vector<at::Tensor> FusionDefinition::execute(
    const at::ArrayRef<c10::IValue>& inputs,
    bool override_user_schedule) const {
  TORCH_CHECK(id().has_value(), "Valid fusion schedule is not available!");

  auto& scheds = fusionCache()->queryFusionSchedules(id().value());

  if (!override_user_schedule) {
    auto device = getCommonDeviceCUDA(inputs);
    TORCH_CHECK(
        inputs.size() == 0 || device > -1,
        "Inputs are not all on the same device!");
    auto user_sched_id =
        fusionCache()->queryUserScheduleId(scheds, inputs, device);
    if (user_sched_id.has_value()) {
      auto& user_sched = fusionCache()->queryUserSchedule(
          scheds, user_sched_id.value(), device);
      return user_sched.executor->runFusion(inputs);
    }
  }

  return scheds.auto_gen_schedules->runFusionWithInputs(inputs);
}

c10::optional<size_t> FusionDefinition::id() const {
  return fusion_id_;
}

Scalar FusionDefinition::defineScalar() {
  FUSER_PERF_SCOPE("FusionDefinition::defineScalar");
  Scalar out(recording_state_.size(), this);
  recording_state_.emplace_back(out(), StateType::Scalar);
  return out;
}

Tensor FusionDefinition::defineTensor(size_t dims) {
  FUSER_PERF_SCOPE("FusionDefinition::defineTensor");
  Tensor out(recording_state_.size(), dims, this);
  recording_state_.emplace_back(out(), StateType::Tensor);
  return out;
}

void FusionDefinition::defineRecord(RecordFunctor* record) {
  FUSER_PERF_SCOPE("FusionDefinition::defineRecord");
  TORCH_CHECK(
      (recording_.size() + 1) <= max_length_,
      "The fusion definition has exceeded ",
      max_length_,
      "operations.  The max_length for FusionDefintion's might need to be ",
      "increased if the definition is created as expected.");
  addRecord(record);
  auto cache_entry = fusionCache()->queryChildren(recording_.back().get());
  // If the Record is found in the cache, the FusionDefinition and the Cache
  // will not share Record given the Record had to be created in order to
  // match it but it also already existed in the cache.
  if (cache_entry.has_value()) {
    if (isDebugDumpEnabled(DebugDumpOption::PythonFrontendDebug)) {
      std::cout << "\nFusionDefinition: Record (hash: 0x" << std::hex
                << record->hash() << ") hit in Fusion Cache.\n";
    }
    // The FusionDefinition and the Cache will share the Record
  } else {
    if (isDebugDumpEnabled(DebugDumpOption::PythonFrontendDebug)) {
      std::cout << "\nFusionDefinition: Record (hash: 0x" << std::hex
                << record->hash() << ") missed in Fusion Cache.\n";
    }
    fusionCache()->createChild(recording_.back().get());
  }
  fusionCache()->traverseTrie(recording_.back().get());
}

Fusion* FusionDefinition::preschedFusion() {
  TORCH_CHECK(
      fusion_id_.has_value(),
      "FusionDefinition does not contain a definition, yet!");
  return fusionCache()
      ->queryFusionSchedules(fusion_id_.value())
      .preschedFusion();
}

void FusionDefinition::printMathIr() {
  return preschedFusion()->printMath();
}

State FusionDefinition::recordingState(size_t index) const {
  return recording_state_.at(index);
}

} // namespace nvfuser::python_frontend
