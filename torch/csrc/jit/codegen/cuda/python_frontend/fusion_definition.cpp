#ifdef USE_CUDA
#include <torch/csrc/jit/codegen/cuda/instrumentation.h>
#include <torch/csrc/jit/codegen/cuda/python_frontend/fusion_definition.h>
#include <torch/csrc/jit/codegen/cuda/python_frontend/fusion_manager.h>
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

FusionDefinition::FusionDefinition(
    FusionManager* fusion_manager,
    size_t max_length)
    : max_length_(max_length),
      fusion_manager_(fusion_manager),
      end_record_(new EndRecord()),
      recording_(),
      recording_state_(),
      fusion_state_(),
      ops(this) {}

void FusionDefinition::buildFusionIr() {
  FUSER_PERF_SCOPE("FusionDefinition::buildFusionIr");
  Nvf::FusionGuard::setCurFusion(fusionManagerPtr()->fusionPtr());
  fusion_state_.resize(recording_state_.size(), nullptr);
  for (auto& record : recording_) {
    auto functor = record.get();
    (*functor)(*this);
  }
  Nvf::FusionGuard::setCurFusion(nullptr);
}

FusionManager* FusionDefinition::fusionManagerPtr() const {
  TORCH_INTERNAL_ASSERT(
      fusion_manager_ != nullptr, "FusionManager pointer is null!");
  return fusion_manager_;
}

FusionDefinition* FusionDefinition::enter() {
  fusionManagerPtr()->resetFusionCachePtr();
  return this;
}
void FusionDefinition::exit() {
  FUSER_PERF_SCOPE("FusionDefinition::exit");
  auto cache_entry = fusionManagerPtr()->lookupFusionCacheEntry(end_record_);
  if (!cache_entry.has_value()) {
    if (Nvf::isDebugDumpEnabled(Nvf::DebugDumpOption::PythonFrontendDebug)) {
      std::cout << "\nFusionDefinition: Terminal Node not found.\n";
    }
    fusionManagerPtr()->createTerminalFusionCacheEntry(end_record_);
    fusionManagerPtr()->traverseFusionCache(end_record_);

    if (Nvf::isDebugDumpEnabled(Nvf::DebugDumpOption::PythonDefinition)) {
      print(std::cout);
    }

    buildFusionIr();

    if (Nvf::isDebugDumpEnabled(Nvf::DebugDumpOption::FusionIrPresched)) {
      fusionManagerPtr()->printIr();
    }
  } else {
    if (Nvf::isDebugDumpEnabled(Nvf::DebugDumpOption::PythonFrontendDebug)) {
      std::cout << "\nFusionDefinition: Terminal Node found!\n";
    }
    fusionManagerPtr()->traverseFusionCache(end_record_);
  }
}

void FusionDefinition::print(std::ostream& os) const {
  os << "\ndef nvfuser_fusion(fd : FusionDefinition) -> None :\n";
  os << std::dec;
  for (auto& rec : recording_) {
    os << "    ";
    rec->print(os);
    os << "\n";
  }
  os << "\n";
}

Scalar* FusionDefinition::defineScalar() {
  FUSER_PERF_SCOPE("FusionDefinition::defineScalar");
  Scalar* out = new nvfuser::Scalar(recording_state_.size());
  recording_state_.emplace_back(out);
  return out;
}
Tensor* FusionDefinition::defineTensor() {
  FUSER_PERF_SCOPE("FusionDefinition::defineTensor");
  Tensor* out = new nvfuser::Tensor(recording_state_.size());
  recording_state_.emplace_back(out);
  return out;
}
void FusionDefinition::defineRecord(RecordFunctor* record) {
  FUSER_PERF_SCOPE("FusionDefinition::defineRecord");
  TORCH_CHECK(
      recording_.size() <= max_length_,
      "The fusion definition has exceeded ",
      max_length_,
      "operations.  The max_length for FusionDefintion's might need to be ",
      "increased if the definition is created as expected.");
  recording_.emplace_back(record);
  auto cache_entry =
      fusionManagerPtr()->lookupFusionCacheEntry(recording_.back());
  if (cache_entry.has_value()) {
    if (Nvf::isDebugDumpEnabled(Nvf::DebugDumpOption::PythonFrontendDebug)) {
      std::cout << "\nFusionDefinition: Record (hash: 0x" << std::hex
                << record->hash() << ") hit in Fusion Cache.\n";
    }
  } else {
    if (Nvf::isDebugDumpEnabled(Nvf::DebugDumpOption::PythonFrontendDebug)) {
      std::cout << "\nFusionDefinition: Record (hash: 0x" << std::hex
                << record->hash() << ") missed in Fusion Cache.\n";
    }
    fusionManagerPtr()->createFusionCacheEntry(recording_.back());
  }
  fusionManagerPtr()->traverseFusionCache(recording_.back());
}

void FusionDefinition::addInput(Nvf::Val* input) {
  fusionManagerPtr()->fusionPtr()->addInput(input);
}
void FusionDefinition::addOutput(Nvf::Val* output) {
  fusionManagerPtr()->fusionPtr()->addOutput(output);
}

Nvf::Val* FusionDefinition::getFusionState(size_t index) const {
  return fusion_state_.at(index);
}
void FusionDefinition::setFusionState(size_t index, Nvf::Val* val) {
  fusion_state_.at(index) = val;
}

} // namespace nvfuser

#endif // USE_CUDA
