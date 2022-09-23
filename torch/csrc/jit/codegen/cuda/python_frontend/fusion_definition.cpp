#include <torch/csrc/jit/codegen/cuda/instrumentation.h>
#include <torch/csrc/jit/codegen/cuda/python_frontend/fusion_cache.h>
#include <torch/csrc/jit/codegen/cuda/python_frontend/fusion_definition.h>
#include <torch/csrc/jit/codegen/cuda/python_frontend/fusion_interface.h>
#include <torch/csrc/jit/codegen/cuda/utils.h>

// Require namespace for perf scope instrumentation
using namespace torch::jit::fuser::cuda::inst;

namespace nvfuser {

const char* dtypeToPyString(Nvf::DataType t) {
  switch (t) {
    case Nvf::DataType::Bool:
      return "DataType.Bool";
    case Nvf::DataType::Double:
      return "DataType.Double";
    case Nvf::DataType::Float:
      return "DataType.Float";
    case Nvf::DataType::Half:
      return "DataType.Half";
    case Nvf::DataType::BFloat16:
      return "DataType.Bfloat16";
    case Nvf::DataType::Int:
      return "DataType.Int";
    case Nvf::DataType::Int32:
      return "DataType.Int32";
    case Nvf::DataType::ComplexFloat:
      return "DataType.ComplexFloat";
    case Nvf::DataType::ComplexDouble:
      return "DataType.ComplexDouble";
    case Nvf::DataType::Null:
      return "DataType.Null";
    default:
      break;
  }
  TORCH_INTERNAL_ASSERT(false, "No string found for data type.");
  return nullptr;
}

FusionDefinition::FusionDefinition(FusionInterface* fusion, size_t max_length)
    : max_length_(max_length),
      fusion_(fusion),
      fusion_cache_(FusionCache::get()),
      end_record_(new EndRecord()),
      recording_(),
      recording_state_(),
      fusion_state_(),
      ops(this) {}

void FusionDefinition::buildFusionIr() {
  FUSER_PERF_SCOPE("FusionDefinition::buildFusionIr");
  auto fusion_guard = fusionInterfacePtr()->guard();
  fusion_state_.resize(recording_state_.size(), nullptr);
  for (auto& record : recording_) {
    auto functor = record.get();
    (*functor)(*this);
  }
}

FusionCache* FusionDefinition::fusionCachePtr() const {
  TORCH_INTERNAL_ASSERT(
      fusion_cache_ != nullptr, "FusionCache pointer is null!");
  return fusion_cache_;
}

FusionInterface* FusionDefinition::fusionInterfacePtr() const {
  TORCH_INTERNAL_ASSERT(fusion_ != nullptr, "FusionInterface pointer is null!");
  return fusion_;
}

FusionDefinition* FusionDefinition::enter() {
  TORCH_CHECK(max_length_ > 0, "Can't make a FusionDefinition with 0 records!");
  TORCH_CHECK(
      !fusionInterfacePtr()->defined(), "Fusion Interface is already defined!");
  fusionCachePtr()->resetFusionCachePtr();
  return this;
}

void FusionDefinition::exit() {
  FUSER_PERF_SCOPE("FusionDefinition::exit");
  auto cache_entry =
      fusionCachePtr()->lookupFusionCacheEntry(end_record_.get());
  if (!cache_entry.has_value()) {
    if (Nvf::isDebugDumpEnabled(Nvf::DebugDumpOption::PythonFrontendDebug)) {
      std::cout << "\nFusionDefinition: Terminal Node not found.\n";
    }
    auto fusion_id =
        fusionCachePtr()->createFusionCacheEntry(end_record_.get());
    TORCH_CHECK(fusion_id.has_value(), "Invalid fusion id!");
    fusionInterfacePtr()->define(fusion_id.value());
    fusionCachePtr()->traverseFusionCache(end_record_.get());

    if (Nvf::isDebugDumpEnabled(Nvf::DebugDumpOption::PythonDefinition)) {
      print(std::cout);
    }

    buildFusionIr();

    if (Nvf::isDebugDumpEnabled(Nvf::DebugDumpOption::FusionIrPresched)) {
      fusionInterfacePtr()->print();
    }
  } else {
    if (Nvf::isDebugDumpEnabled(Nvf::DebugDumpOption::PythonFrontendDebug)) {
      std::cout << "\nFusionDefinition: Terminal Node found!\n";
    }
    fusionInterfacePtr()->define(cache_entry.value()->fusion_id);
    fusionCachePtr()->traverseFusionCache(end_record_.get());
  }
}

void FusionDefinition::print(std::ostream& os) const {
  os << "\ndef nvfuser_fusion_id" << fusion_->id();
  os << "(fd : FusionDefinition) -> None :\n";
  os << std::dec;
  for (auto& rec : recording_) {
    os << "    ";
    rec->print(os);
    os << "\n";
  }
  os << "\n";
}

Scalar FusionDefinition::defineScalar() {
  FUSER_PERF_SCOPE("FusionDefinition::defineScalar");
  Scalar out(recording_state_.size());
  recording_state_.emplace_back(out(), StateType::Scalar);
  return out;
}

Tensor FusionDefinition::defineTensor() {
  FUSER_PERF_SCOPE("FusionDefinition::defineTensor");
  Tensor out(recording_state_.size());
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
  recording_.emplace_back(record);
  auto cache_entry =
      fusionCachePtr()->lookupFusionCacheEntry(recording_.back().get());
  // If the Record is found in the cache, the FusionDefinition and the Cache
  // will not share Record given the Record had to be created in order to
  // match it but it also already existed in the cache.
  if (cache_entry.has_value()) {
    if (Nvf::isDebugDumpEnabled(Nvf::DebugDumpOption::PythonFrontendDebug)) {
      std::cout << "\nFusionDefinition: Record (hash: 0x" << std::hex
                << record->hash() << ") hit in Fusion Cache.\n";
    }
    // The FusionDefinition and the Cache will share the Record
  } else {
    if (Nvf::isDebugDumpEnabled(Nvf::DebugDumpOption::PythonFrontendDebug)) {
      std::cout << "\nFusionDefinition: Record (hash: 0x" << std::hex
                << record->hash() << ") missed in Fusion Cache.\n";
    }
    fusionCachePtr()->createFusionCacheEntry(recording_.back().get());
  }
  fusionCachePtr()->traverseFusionCache(recording_.back().get());
}

void FusionDefinition::addInput(Nvf::Val* input) {
  fusionInterfacePtr()->addInput(input);
}
void FusionDefinition::addOutput(Nvf::Val* output) {
  fusionInterfacePtr()->addOutput(output);
}

Nvf::Val* FusionDefinition::getFusionState(size_t index) const {
  return fusion_state_.at(index);
}
void FusionDefinition::setFusionState(size_t index, Nvf::Val* val) {
  fusion_state_.at(index) = val;
}

State FusionDefinition::recordingState(size_t index) const {
  return recording_state_.at(index);
}

} // namespace nvfuser
