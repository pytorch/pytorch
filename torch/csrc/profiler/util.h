#pragma once

#include <cstddef>
#include <cstdint>
#include <list>
#include <string>
#include <unordered_map>
#include <vector>

#include <ATen/record_function.h>
#include <c10/macros/Macros.h>
#include <c10/util/hash.h>
#include <torch/csrc/Export.h>
#include <torch/csrc/jit/frontend/source_range.h>
#include <optional>

// TODO: replace with pytorch/rfcs#43 when it is ready.
#define SOFT_ASSERT(cond, ...)                         \
  [&]() -> bool {                                      \
    if (C10_UNLIKELY(!(cond))) {                       \
      torch::profiler::impl::logSoftAssert(            \
          __func__,                                    \
          __FILE__,                                    \
          static_cast<uint32_t>(__LINE__),             \
          #cond,                                       \
          ::c10::str(__VA_ARGS__));                    \
      if (torch::profiler::impl::softAssertRaises()) { \
        TORCH_INTERNAL_ASSERT(cond, __VA_ARGS__);      \
      } else {                                         \
        TORCH_WARN_ONCE(__VA_ARGS__);                  \
      }                                                \
      return false;                                    \
    }                                                  \
    return true;                                       \
  }()

namespace torch::profiler::impl {
TORCH_API bool softAssertRaises();
TORCH_API void setSoftAssertRaises(std::optional<bool> value);
TORCH_API void logSoftAssert(
    const char* func,
    const char* file,
    uint32_t line,
    const char* cond,
    const char* args);
inline void logSoftAssert(
    const char* func,
    const char* file,
    uint32_t line,
    const char* cond,
    ::c10::detail::CompileTimeEmptyString args) {
  logSoftAssert(func, file, line, cond, (const char*)args);
}
TORCH_API void logSoftAssert(
    const char* func,
    const char* file,
    uint32_t line,
    const char* cond,
    const std::string& args);

using shape =
    std::variant<std::vector<int64_t>, std::vector<std::vector<int64_t>>>;
constexpr int TENSOR_LIST_DISPLAY_LENGTH_LIMIT = 30;

std::string getNvtxStr(
    const char* name,
    int64_t sequence_nr,
    const std::vector<std::vector<int64_t>>& shapes,
    at::RecordFunctionHandle op_id = 0,
    const std::list<std::pair<at::RecordFunctionHandle, int>>& input_op_ids =
        {});

struct TORCH_API FileLineFunc {
  std::string filename;
  size_t line;
  std::string funcname;
};

struct TORCH_API SaveNcclMetaConfig {
  bool truncate;
  bool introspectMetadata;
  bool introspectInputs;
  bool introspectOutputs;

  // Default constructor with default values
  SaveNcclMetaConfig()
      : truncate(true),
        introspectMetadata(true),
        introspectInputs(false),
        introspectOutputs(false) {}

  SaveNcclMetaConfig(
      bool truncate,
      bool introspectMetadata,
      bool introspectInputs,
      bool introspectOutputs)
      : truncate(truncate),
        introspectMetadata(introspectMetadata),
        introspectInputs(introspectInputs),
        introspectOutputs(introspectOutputs) {}
};

TORCH_API std::vector<FileLineFunc> prepareCallstack(
    const std::vector<jit::StackEntry>& cs);
TORCH_API std::vector<std::string> callstackStr(
    const std::vector<FileLineFunc>& cs);
TORCH_API std::string stacksToStr(
    const std::vector<std::string>& stacks,
    const char* delim);
TORCH_API std::vector<std::vector<int64_t>> inputSizes(
    const at::RecordFunction& fn,
    const bool flatten_list_enabled = false);
TORCH_API std::string variantShapesToStr(const std::vector<shape>& shapes);
TORCH_API std::string shapesToStr(
    const std::vector<std::vector<int64_t>>& shapes);
TORCH_API std::string strListToStr(const std::vector<std::string>& types);
TORCH_API std::string inputOpIdsToStr(
    const std::list<std::pair<at::RecordFunctionHandle, int>>& input_op_ids);
TORCH_API std::string ivalueToStr(const c10::IValue& val, bool isString);
TORCH_API std::string ivalueListToStr(const std::vector<c10::IValue>& list);
TORCH_API std::vector<std::string> inputTypes(const at::RecordFunction& fn);

std::unordered_map<std::string, c10::IValue> TORCH_API
saveExtraArgs(const at::RecordFunction& fn);
std::unordered_map<std::string, std::string> TORCH_API saveNcclMeta(
    const at::RecordFunction& fn,
    const SaveNcclMetaConfig& config = SaveNcclMetaConfig());
int getTensorStartHint(const at::Tensor& t);
bool checkFunctionOutputsForLogging(const at::RecordFunction& fn);
bool checkFunctionInputsForLogging(const at::RecordFunction& fn);
std::pair<bool, std::variant<int, std::vector<int>>> findStartAddrForTensors(
    const c10::IValue& val);
uint64_t TORCH_API computeFlops(
    const std::string& op_name,
    const std::unordered_map<std::string, c10::IValue>& extra_args);

std::string shapeToStr(const std::vector<int64_t>& shape);

template <typename T>
class TORCH_API GlobalStateManager {
 public:
  static GlobalStateManager& singleton() {
    /* library-local */ static GlobalStateManager singleton_;
    return singleton_;
  }

  static void push(std::shared_ptr<T>&& state) {
    if (singleton().state_) {
      LOG(WARNING) << "GlobalStatePtr already exists!";
    } else {
      singleton().state_ = std::move(state);
    }
  }

  static auto* get() {
    return singleton().state_.get();
  }

  static std::shared_ptr<T> pop() {
    auto out = singleton().state_;
    singleton().state_.reset();
    return out;
  }

 private:
  GlobalStateManager() = default;

  std::shared_ptr<T> state_;
};

struct HashCombine {
  template <typename T0, typename T1>
  size_t operator()(const std::pair<T0, T1>& i) {
    return c10::get_hash((*this)(i.first), (*this)(i.second));
  }

  template <typename... Args>
  size_t operator()(const std::tuple<Args...>& i) {
    return c10::get_hash(i);
  }

  template <typename T>
  size_t operator()(const T& i) {
    return c10::get_hash(i);
  }
};

#ifdef USE_DISTRIBUTED
constexpr auto kCommsName = "Collective name";
constexpr auto kDtype = "dtype";
constexpr auto kInMsgNelems = "In msg nelems";
constexpr auto kOutMsgNelems = "Out msg nelems";
constexpr auto kInSplit = "In split size";
constexpr auto kOutSplit = "Out split size";
constexpr auto kGlobalRankStart = "Global rank start";
constexpr auto kGlobalRankStride = "Global rank stride";
constexpr auto kGroupSize = "Group size";
constexpr auto kProcessGroupName = "Process Group Name";
constexpr auto kProcessGroupDesc = "Process Group Description";
constexpr auto kGroupRanks = "Process Group Ranks";
constexpr auto kRank = "Rank";
constexpr auto kP2pSrc = "Src Rank";
constexpr auto kP2pDst = "Dst Rank";
constexpr auto kInTensorsStart = "Input Tensors start";
constexpr auto kOutTensorsStart = "Output Tensors start";
constexpr auto kIsAsynchronizedOp = "Is asynchronized op";
#endif // USE_DISTRIBUTED

} // namespace torch::profiler::impl
