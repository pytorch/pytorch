#include <torch/csrc/autograd/function.h>
#include <torch/csrc/profiler/collection.h>
#include <torch/csrc/profiler/kineto_shim.h>
#include <torch/csrc/profiler/util.h>

#include <c10/util/ArrayRef.h>
#include <c10/util/irange.h>
#include <fmt/format.h>
#include <fmt/ranges.h>

#ifdef USE_KINETO
#include <libkineto.h>
#endif
#ifdef USE_DISTRIBUTED
#include <torch/csrc/distributed/c10d/ParamCommsUtils.hpp>
#endif // USE_DISTRIBUTED

namespace torch::profiler::impl {

namespace {
std::optional<bool> soft_assert_raises_;
} // namespace

void setSoftAssertRaises(std::optional<bool> value) {
  soft_assert_raises_ = value;
}

bool softAssertRaises() {
  return soft_assert_raises_.value_or(false);
}

void logSoftAssert(
    // @lint-ignore CLANGTIDY
    const char* func,
    // @lint-ignore CLANGTIDY
    const char* file,
    // @lint-ignore CLANGTIDY
    uint32_t line,
    // @lint-ignore CLANGTIDY
    const char* cond,
    // @lint-ignore CLANGTIDY
    const char* args) {
#ifdef USE_KINETO
  std::string error;
  error = fmt::format(
      "{} SOFT ASSERT FAILED at {}:{}, func: {}, args: {}",
      cond,
      file,
      line,
      func,
      args);
  // TODO: Implement profile_id and group_profile_id as 3rd/4th arguments.
  kineto::logInvariantViolation(cond, error, "", "");
#endif
}

void logSoftAssert(
    // @lint-ignore CLANGTIDY
    const char* func,
    // @lint-ignore CLANGTIDY
    const char* file,
    // @lint-ignore CLANGTIDY
    uint32_t line,
    // @lint-ignore CLANGTIDY
    const char* cond,
    // @lint-ignore CLANGTIDY
    const std::string& args) {
#ifdef USE_KINETO
  std::string error;
  error = fmt::format(
      "{} SOFT ASSERT FAILED at {}:{}, func: {}, args: {}",
      cond,
      file,
      line,
      func,
      args);
  // TODO: Implement profile_id and group_profile_id as 3rd/4th arguments.
  kineto::logInvariantViolation(cond, error, "", "");
#endif
}

// ----------------------------------------------------------------------------
// -- NVTX --------------------------------------------------------------------
// ----------------------------------------------------------------------------
std::string getNvtxStr(
    const char* name,
    int64_t sequence_nr,
    const std::vector<std::vector<int64_t>>& shapes,
    at::RecordFunctionHandle op_id,
    const std::list<std::pair<at::RecordFunctionHandle, int>>& input_op_ids) {
  if (sequence_nr >= -1 || !shapes.empty()) {
    std::string str;
    if (sequence_nr >= 0) {
      str = fmt::format("{}, seq = {}", name, sequence_nr);
    } else if (sequence_nr == -1) {
      str = name;
    } else {
#if defined(USE_ROCM)
      // Only ROCM supports < -1 sequence_nr
      str = name;
#endif
    }
    if (op_id > 0) {
      str = fmt::format("{}, op_id = {}", str, op_id);
    }
    if (!shapes.empty()) {
      str = fmt::format("{}, sizes = {}", str, shapesToStr(shapes));
    }
    // Include the op ids of the input edges so
    // you can build the network graph
    if (!input_op_ids.empty()) {
      str = fmt::format(
          "{}, input_op_ids = {}", str, inputOpIdsToStr(input_op_ids));
    }
    return str;
  } else {
    return name;
  }
}

// ----------------------------------------------------------------------------
// -- Op context (shapes, call stack) -----------------------------------------
// ----------------------------------------------------------------------------
std::vector<FileLineFunc> prepareCallstack(
    const std::vector<jit::StackEntry>& cs) {
  std::vector<FileLineFunc> entries;
  entries.reserve(cs.size());
  for (const auto& entry : cs) {
    auto& range = entry.range;
    if (range.source()) {
      auto& src = range.source();
      if (src && src->filename()) {
        auto line =
            src->starting_line_no() + src->lineno_for_offset(range.start());
        entries.emplace_back(
            // NOLINTNEXTLINE(bugprone-unchecked-optional-access)
            FileLineFunc{*(src->filename()), line, entry.filename});
      }
    }
  }
  return entries;
}

std::vector<std::string> callstackStr(const std::vector<FileLineFunc>& cs) {
  std::vector<std::string> cs_str;
  cs_str.reserve(cs.size());
  for (const auto& entry : cs) {
    std::stringstream loc;
    loc << entry.filename << "(" << entry.line << "): " << entry.funcname;
    cs_str.push_back(loc.str());
  }
  return cs_str;
}

std::string stacksToStr(
    const std::vector<std::string>& stacks,
    const char* delim) {
  std::ostringstream oss;
  std::transform(
      stacks.begin(),
      stacks.end(),
      std::ostream_iterator<std::string>(oss, delim),
      [](std::string s) -> std::string {
#ifdef _WIN32
        // replace the windows backslash with forward slash
        std::replace(s.begin(), s.end(), '\\', '/');
#endif
        return s;
      });
  auto rc = oss.str();
  return "\"" + rc + "\"";
}

static std::vector<std::vector<int64_t>> flattenList(
    const c10::List<c10::IValue>& list) {
  std::vector<std::vector<int64_t>> tensor_dims;
  for (const c10::IValue& input : list) {
    if (input.isTensor()) {
      const at::Tensor& tensor = input.toTensor();
      if (tensor.defined()) {
        tensor_dims.push_back(input.toTensor().sizes().vec());
      }
    }
  }
  return tensor_dims;
}

std::vector<std::vector<int64_t>> inputSizes(
    const at::RecordFunction& fn,
    bool flatten_list_enabled) {
  std::vector<std::vector<int64_t>> sizes;
  sizes.reserve(fn.inputs().size());
  for (const c10::IValue& input : fn.inputs()) {
    if (input.isTensor()) {
      const at::Tensor& tensor = input.toTensor();
      if (tensor.defined()) {
        sizes.push_back(input.toTensor().sizes().vec());
      } else {
        sizes.emplace_back();
      }
    } else if (input.isList()) {
      std::vector<std::vector<int64_t>> tmp_sizes;
      if (flatten_list_enabled) {
        tmp_sizes = flattenList(input.toList());
      }
      // Extend the current sizes array by the array returned from input sizes
      if (!tmp_sizes.empty()) {
        sizes.insert(sizes.end(), tmp_sizes.begin(), tmp_sizes.end());
      } else {
        sizes.emplace_back();
      }
    } else {
      sizes.emplace_back();
    }
  }
  return sizes;
}

std::string shapesToStr(const std::vector<std::vector<int64_t>>& shapes) {
  std::string str("[");
  for (const auto t_idx : c10::irange(shapes.size())) {
    if (t_idx > 0) {
      str = fmt::format("{}, ", str);
    }
    str = fmt::format("{}{}", str, shapeToStr(shapes[t_idx]));
  }
  str = fmt::format("{}]", str);
  return str;
}

std::string variantShapesToStr(const std::vector<shape>& shapes) {
  std::string str("[");
  for (const auto t_idx : c10::irange(shapes.size())) {
    if (t_idx > 0) {
      str = fmt::format("{}, ", str);
    }
    if (std::holds_alternative<std::vector<int64_t>>(shapes[t_idx])) {
      const auto& shape = std::get<std::vector<int64_t>>(shapes[t_idx]);
      str = fmt::format("{}{}", str, shapeToStr(shape));
    } else if (std::holds_alternative<std::vector<std::vector<int64_t>>>(
                   shapes[t_idx])) {
      const auto& tensor_shape =
          std::get<std::vector<std::vector<int64_t>>>(shapes[t_idx]);
      if (tensor_shape.size() > TENSOR_LIST_DISPLAY_LENGTH_LIMIT) {
        // skip if the tensor list is too long
        str = fmt::format("{}[]", str);
        continue;
      }
      str = fmt::format("{}[", str);
      for (const auto s_idx : c10::irange(tensor_shape.size())) {
        if (s_idx > 0) {
          str = fmt::format("{}, ", str);
        }
        str = fmt::format("{}{}", str, shapeToStr(tensor_shape[s_idx]));
      }
      str = fmt::format("{}]", str);
    }
  }
  str = fmt::format("{}]", str);
  return str;
}

std::string shapeToStr(const std::vector<int64_t>& shape) {
  std::string str("[");
  for (const auto s_idx : c10::irange(shape.size())) {
    if (s_idx > 0) {
      str = fmt::format("{}, ", str);
    }
    str = fmt::format("{}{}", str, shape[s_idx]);
  }
  str = fmt::format("{}]", str);
  return str;
}

std::string inputOpIdsToStr(
    const std::list<std::pair<at::RecordFunctionHandle, int>>& input_op_ids) {
  std::string str("[");
  int idx = 0;

  for (const auto& op_id_info_pair : input_op_ids) {
    if (idx++ > 0) {
      str = fmt::format("{}, ", str);
    }
    // (OpId,OutputNr)
    str = fmt::format(
        "{}({},{})", str, op_id_info_pair.first, op_id_info_pair.second);
  }
  str = fmt::format("{}]", str);
  return str;
}

std::string strListToStr(const std::vector<std::string>& types) {
  if (types.empty()) {
    return "[]";
  } else {
    std::ostringstream oss;
    std::transform(
        types.begin(),
        types.end(),
        std::ostream_iterator<std::string>(oss, ", "),
        [](const std::string& s) -> std::string { return "\"" + s + "\""; });
    auto rc = oss.str();
    rc.erase(rc.length() - 2); // remove last ", "
    return "[" + rc + "]";
  }
}
std::string ivalueToStr(const c10::IValue& val, bool isString) {
  std::stringstream ss;
  if (val.isNone()) {
    return "\"None\"";
  } else {
    ss.str("");
    if (isString) {
      ss << "\"";
    }
    ss << val;
    if (isString) {
      ss << "\"";
    }
    std::string mystr = ss.str();

    // For boolean the values that ivalue gives is "True" and "False" but
    // json only takes "true" and "false" so we convert the string to lower case
    if (val.isBool()) {
      for (char& c : mystr) {
        c = static_cast<char>(std::tolower(c));
      }
    }

    // A double quote can cause issues with the chrome tracing so force
    // all inputs to not contain more than the 2 we add in this function
    auto count = std::count(mystr.begin(), mystr.end(), '"');
    return count > 2 ? "\"None\"" : mystr;
  }
}

std::string ivalueListToStr(const std::vector<c10::IValue>& list) {
  std::vector<std::string> concrete_str_inputs;
  std::stringstream ss;
  for (const auto& val : list) {
    if (val.isNone()) {
      concrete_str_inputs.emplace_back("");
    } else {
      ss.str("");
      ss << val;
      concrete_str_inputs.emplace_back(ss.str());
    }
  }
  return strListToStr(concrete_str_inputs);
}

std::vector<std::string> inputTypes(const at::RecordFunction& fn) {
  std::vector<std::string> types;
  types.reserve(fn.inputs().size());
  for (const c10::IValue& input : fn.inputs()) {
    if (input.isTensor()) {
      const at::Tensor& tensor = input.toTensor();
      if (tensor.defined()) {
        types.push_back(
            static_cast<std::string>(input.toTensor().dtype().name()));
      } else {
        types.emplace_back();
      }
    } else if (input.isScalar() || input.isList()) {
      types.push_back(input.tagKind());
    } else {
      types.emplace_back();
    }
  }
  return types;
}

// ----------------------------------------------------------------------------
// -- NCCL Metadata -----------------------------------------------------------
// ----------------------------------------------------------------------------

static constexpr int32_t kTruncatLength = 30;

template <typename ListLikeType>
static inline std::string format_list(
    ListLikeType list,
    bool truncate,
    bool with_escaped_quotes = true) {
  if (truncate && list.size() > kTruncatLength) {
    if (with_escaped_quotes == true) {
      auto x = fmt::format(
          "\"[{}, ..., {}]\"",
          fmt::join(list.begin(), list.begin() + kTruncatLength - 1, ", "),
          *std::prev(list.end()));
      return x;
    } else {
      auto x = fmt::format(
          "[{}, ..., {}]",
          fmt::join(list.begin(), list.begin() + kTruncatLength - 1, ", "),
          *std::prev(list.end()));
      return x;
    }
  }
  if (with_escaped_quotes == true) {
    auto x = fmt::format("\"[{}]\"", fmt::join(list.begin(), list.end(), ", "));
    return x;
  } else {
    auto x = fmt::format("[{}]", fmt::join(list.begin(), list.end(), ", "));
    return x;
  }
}

std::pair<bool, std::variant<int, std::vector<int>>> findStartAddrForTensors(
    const c10::IValue& val) {
  if (val.isTensor()) {
    // Store hints about where the input starts in memory.
    // Useful for debugging memory access patterns.
    const auto& tensor = val.toTensor();
    const int result = getTensorStartHint(tensor);
    return {false, result};
  } else if (val.isTuple()) {
    const auto& val_tuple = val.toTupleRef().elements();
    size_t tuple_size = val_tuple.size();
    std::vector<int> responses;
    responses.reserve(tuple_size);
    for (const auto j : c10::irange(tuple_size)) {
      auto [is_list, res] = findStartAddrForTensors(val_tuple[j]);
      if (is_list) {
        const auto& vec_res = std::get<std::vector<int>>(res);
        responses.insert(responses.end(), vec_res.begin(), vec_res.end());
      } else {
        responses.push_back(std::get<int>(res));
      }
    }
    return {true, responses};
  } else if (val.isList()) {
    const auto& val_list = val.toList();
    size_t list_size = val_list.size();
    std::vector<int> responses;
    responses.reserve(list_size);
    for (const auto j : c10::irange(list_size)) {
      auto [is_list, res] = findStartAddrForTensors(val_list[j]);
      if (is_list) {
        auto const& vec_res = std::get<std::vector<int>>(res);
        responses.insert(responses.end(), vec_res.begin(), vec_res.end());
      } else {
        responses.push_back(std::get<int>(res));
      }
    }
    return {true, responses};
  } else {
    // push back an invalid value for indices representing non-tensor inputs
    return {false, -1};
  }
}

std::unordered_map<std::string, std::string> saveNcclMeta(
    // @lint-ignore CLANGTIDY
    const at::RecordFunction& fn,
    // @lint-ignore CLANGTIDY
    const SaveNcclMetaConfig& config) {
  std::unordered_map<std::string, std::string> map;
#ifdef USE_DISTRIBUTED
  auto debugInfo = dynamic_cast<ParamCommsDebugInfo*>(
      c10::ThreadLocalDebugInfo::get(c10::DebugInfoKind::PARAM_COMMS_INFO));

  if (config.introspectMetadata) {
    if (debugInfo == nullptr) {
      LOG(WARNING) << "ParamCommsDebugInfo not available for function: "
                   << fn.name();
      return map;
    }
    auto& collective_name = debugInfo->getCollectiveName();
    map.emplace(kCommsName, fmt::format("\"{}\"", collective_name));
    map.emplace(
        kDtype, fmt::format("\"{}\"", c10::toString(debugInfo->getDType())));
    map.emplace(kInMsgNelems, std::to_string(debugInfo->getInMessageNelems()));
    map.emplace(
        kOutMsgNelems, std::to_string(debugInfo->getOutMessageNelems()));

    auto& inSplitSizes = debugInfo->getInputSplitSizes();
    map.emplace(kInSplit, format_list(inSplitSizes, config.truncate));

    auto& outSplitSizes = debugInfo->getOutputSplitSizes();
    map.emplace(kOutSplit, format_list(outSplitSizes, config.truncate));

    auto globalRankStart = debugInfo->getGlobalRankStart();
    if (globalRankStart >= 0) {
      map.emplace(kGlobalRankStart, std::to_string(globalRankStart));
    }
    auto globalRankStride = debugInfo->getGlobalRankStride();
    if (globalRankStride > 0) {
      map.emplace(kGlobalRankStride, std::to_string(globalRankStride));
    }
    map.emplace(kGroupSize, std::to_string(debugInfo->getWorldSize()));
    auto& group_name = debugInfo->getProcessGroupName();
    if (!group_name.empty()) {
      map.emplace(kProcessGroupName, fmt::format("\"{}\"", group_name));
    }
    auto& group_desc = debugInfo->getProcessGroupDesc();
    if (!group_desc.empty()) {
      map.emplace(kProcessGroupDesc, fmt::format("\"{}\"", group_desc));
    }
    auto& groupRanks = debugInfo->getGroupRanks();
    map.emplace(kGroupRanks, format_list(groupRanks, config.truncate));

    auto rank = debugInfo->getRank();
    map.emplace(kRank, std::to_string(rank));
    int nRanks = static_cast<int>(groupRanks.size());
    if (collective_name == "send") {
      if (rank >= 0 && rank < nRanks) {
        map.emplace(kP2pDst, std::to_string(groupRanks[rank]));
      }
    } else if (collective_name == "recv") {
      if (rank >= 0 && rank < nRanks) {
        map.emplace(kP2pSrc, std::to_string(groupRanks[rank]));
      }
    }
  }

  if (get_record_tensor_addrs_enabled()) {
    std::vector<std::string> addressList;
    if (config.introspectInputs) {
      auto num_inputs = fn.num_inputs();
      const auto inputs = fn.inputs();
      if (checkFunctionInputsForLogging(fn)) {
        // need to account for Stack mode where the inputs are at the end.
        size_t input_start = inputs.size() - num_inputs;
        for (const auto i : c10::irange(input_start, inputs.size())) {
          const c10::IValue& val = inputs[i];
          auto [is_list, result] = findStartAddrForTensors(val);
          if (is_list) {
            auto const& list_result = std::get<std::vector<int>>(result);
            addressList.push_back(
                format_list(list_result, config.truncate, false));
          } else {
            auto scalar_result = std::get<int>(result);
            addressList.push_back(std::to_string(scalar_result));
          }
          // today we record a lot of metadata in record_param_comms that shows
          // up as inputs. here we only need the addresses of the first inputs,
          // which are the real tensor inputs to the collective call. So let's
          // break out of the loop here.
          break;
        }
        map.emplace(kInTensorsStart, format_list(addressList, false));
        addressList.clear();
      }
    }
    if (config.introspectOutputs) {
      const auto& outputs = fn.outputs();
      auto num_outputs = fn.num_outputs();
      if (checkFunctionOutputsForLogging(fn)) {
        // need to account for Stack mode where the outputs are at the end.
        size_t output_start = outputs.size() - num_outputs;
        for (const auto i : c10::irange(output_start, outputs.size())) {
          const c10::IValue& val = outputs[i];
          auto [is_list, result] = findStartAddrForTensors(val);
          if (is_list) {
            auto const& list_result = std::get<std::vector<int>>(result);
            addressList.push_back(
                format_list(list_result, config.truncate, false));
          } else {
            auto scalar_result = std::get<int>(result);
            addressList.push_back(std::to_string(scalar_result));
          }
        }
        map.emplace(kOutTensorsStart, format_list(addressList, false));
        addressList.clear();
      }
    }
  }
#endif // USE_DISTRIBUTED
  return map;
}

// ----------------------------------------------------------------------------
// -- FLOPS -------------------------------------------------------------------
// ----------------------------------------------------------------------------
static constexpr auto kConv2dStride = 3;
static constexpr auto kConv2dPadding = 4;
static constexpr auto kConv2dDilation = 5;
static constexpr auto kConv2dGroups = 6;

// List of supported operators
static constexpr auto kConv2dOp = "aten::conv2d";
static constexpr auto kMMOp = "aten::mm";
static constexpr auto kAddMMOp = "aten::addmm";
static constexpr auto kMulOp = "aten::mul";
static constexpr auto kAddOp = "aten::add";
static constexpr auto kBMMOp = "aten::bmm";
static constexpr auto kBAddBMMOp = "aten::baddbmm";

static constexpr auto kInputSize = "input_size";
static constexpr auto kWeightSize = "weight_size";
static constexpr auto kGroups = "groups";
static constexpr auto kPadding = "padding";
static constexpr auto kStride = "stride";
static constexpr auto kDilation = "dilation";
static constexpr auto kMatSize = "mat_size";
static constexpr auto kMat1Size = "mat1_size";
static constexpr auto kMat2Size = "mat2_size";

static std::vector<c10::IntArrayRef> getInputSizes(
    const std::string& op_name,
    size_t min_size,
    c10::ArrayRef<const c10::IValue> inputs,
    const c10::ArrayRef<int>& should_be_tensor) {
  std::stringstream ss;
  if (inputs.size() < min_size) {
    ss << "Failed to save extra arguments for flops computation of op "
       << op_name << ", min size: " << min_size
       << ", actual size: " << inputs.size();
    TORCH_WARN(ss.str());
    return {};
  }
  std::vector<c10::IntArrayRef> inputSizes = {};
  for (auto index : should_be_tensor) {
    if (!inputs[index].isTensor()) {
      ss << "Failed to save extra arguments for flops computation of op "
         << op_name << ", input[" << index << "] must be a tensor.";
      TORCH_WARN(ss.str());
      return {};
    }
    at::Tensor t = inputs[index].toTensor();
    if (t.is_nested()) {
      ss << "Failed to save extra arguments for flops computation of op "
         << op_name << " with input[" << index << "] as nested tensor.";
      TORCH_WARN(ss.str());
      return {};
    }
    inputSizes.emplace_back(t.sizes());
  }
  return inputSizes;
}

std::unordered_map<std::string, c10::IValue> saveExtraArgs(
    const at::RecordFunction& fn) {
  // for specific types of fn, return the saved extra args for computing flops
  std::unordered_map<std::string, c10::IValue> map;
  auto inputs = fn.inputs();
  std::string fname(fn.name());

  if (inputs.empty()) {
    // Input shape is unavailable, return empty map
    return map;
  }

  if (fname == kConv2dOp) {
    const auto inputSizes =
        getInputSizes(fname, kConv2dGroups + 1, inputs, {0, 1});
    if (inputSizes.empty()) {
      return map;
    }
    if (inputSizes[1].size() != 4) {
      TORCH_WARN(
          "Failed to compute flops for op aten::conv2d because it requires a 4D kernel tensor.");
      return map;
    }
    map[kInputSize] = at::IValue(inputSizes[0]);
    map[kWeightSize] = at::IValue(inputSizes[1]);
    map[kStride] = inputs[kConv2dStride];
    map[kPadding] = inputs[kConv2dPadding];
    map[kDilation] = inputs[kConv2dDilation];
    map[kGroups] = inputs[kConv2dGroups];
  } else if (fname == kMMOp) {
    const auto inputSizes = getInputSizes(fname, 2, inputs, {0, 1});
    if (inputSizes.empty()) {
      return map;
    }

    map[kMat1Size] = at::IValue(inputSizes[0]);
    map[kMat2Size] = at::IValue(inputSizes[1]);
  } else if (fname == kAddMMOp) {
    const auto inputSizes = getInputSizes(fname, 3, inputs, {0, 1, 2});
    if (inputSizes.empty()) {
      return map;
    }
    // Exact FLOP count depends on scaling factors alpha and beta but
    // just assume these are +=1.
    // (similar to http://www.netlib.org/lapack/lawnspdf/lawn41.pdf,
    // "Operations Count for the BLAS and LAPACK", Table 3, SGEMM)
    map[kMat1Size] = at::IValue(inputSizes[1]);
    map[kMat2Size] = at::IValue(inputSizes[2]);
  } else if (fname == kMulOp) {
    const auto inputSizes = getInputSizes(fname, 1, inputs, {0});
    if (inputSizes.empty()) {
      return map;
    }
    map[kMatSize] = at::IValue(inputSizes[0]);
  } else if (fname == kAddOp) {
    const auto inputSizes = getInputSizes(fname, 1, inputs, {0});
    if (inputSizes.empty()) {
      return map;
    }
    map[kMatSize] = at::IValue(inputSizes[0]);
  } else if (fname == kBMMOp) {
    const auto inputSizes = getInputSizes(fname, 2, inputs, {0, 1});
    if (inputSizes.empty()) {
      return map;
    }

    map[kMat1Size] = at::IValue(inputSizes[0]);
    map[kMat2Size] = at::IValue(inputSizes[1]);
  } else if (fname == kBAddBMMOp) {
    const auto inputSizes = getInputSizes(fname, 3, inputs, {0, 1, 2});
    if (inputSizes.empty()) {
      return map;
    }

    // Exact FLOP count depends on scaling factors alpha and beta but
    // just assume these are +=1.
    // (similar to http://www.netlib.org/lapack/lawnspdf/lawn41.pdf,
    // "Operations Count for the BLAS and LAPACK", Table 3, SGEMM)
    map[kMat1Size] = at::IValue(inputSizes[1]);
    map[kMat2Size] = at::IValue(inputSizes[2]);
  }

  return map;
}

uint64_t computeFlops(
    const std::string& op_name,
    const std::unordered_map<std::string, c10::IValue>& extra_args) {
  if (op_name == kConv2dOp) {
    if (extra_args.find(kInputSize) == extra_args.end() ||
        extra_args.find(kWeightSize) == extra_args.end() ||
        extra_args.find(kGroups) == extra_args.end() ||
        extra_args.find(kPadding) == extra_args.end() ||
        extra_args.find(kStride) == extra_args.end() ||
        extra_args.find(kDilation) == extra_args.end()) {
      TORCH_WARN(
          "Calculating flops for aten::conv2d requires groups, padding, stride, dilation, input_size, and weight_size in saved arguments.");
      return 0;
    }
    auto input_sizes_ref = extra_args.at(kInputSize);
    auto kernel_sizes_ref = extra_args.at(kWeightSize);
    auto groups_ref = extra_args.at(kGroups);
    auto padding_ref = extra_args.at(kPadding);
    auto stride_ref = extra_args.at(kStride);
    auto dilation_ref = extra_args.at(kDilation);
    if (!input_sizes_ref.isIntList() || !kernel_sizes_ref.isIntList()) {
      TORCH_WARN(
          "Failed to compute flops for op aten::conv2d because it requires input and weight tensor sizes.");
      return 0;
    }
    if (!padding_ref.isIntList() || !stride_ref.isIntList() ||
        !dilation_ref.isIntList()) {
      TORCH_WARN(
          "Failed to compute flops for op aten::conv2d because it requires padding, stride, and dilation values.");
      return 0;
    }

    const auto input_sizes = input_sizes_ref.toDimVector();
    const auto kernel_sizes = kernel_sizes_ref.toDimVector();
    const uint64_t groups = groups_ref.toInt();
    const std::vector<int64_t> padding = padding_ref.toIntVector();
    const std::vector<int64_t> stride = stride_ref.toIntVector();
    const std::vector<int64_t> dilation = dilation_ref.toIntVector();
    if (input_sizes.size() != 4 || kernel_sizes.size() != 4) {
      TORCH_WARN(
          "Failed to compute flops for op aten::conv2d because both input and weight must be size 4.");
      return 0;
    }
    if (!groups) {
      TORCH_WARN(
          "Failed to compute flops for op aten::conv2d because group size must not be 0.");
      return 0;
    }
    if (padding.size() != 2 || dilation.size() != 2) {
      TORCH_WARN(
          "Failed to compute flops for op aten::conv2d because both padding and dilation must be size 2.");
      return 0;
    }
    if (stride.size() != 2 || (stride[0] * stride[1] == 0)) {
      TORCH_WARN(
          "Failed to compute flops for op aten::conv2d because stride must be size 2 and cannot be 0.");
      return 0;
    }
    // format of the input is defined in
    // torch.ao.nn.quantized.functional.conv2d()
    const uint64_t conv2d_multiply_factor = 2;
    auto [minibatch, in_channels, input_h, input_w] = std::make_tuple(
        input_sizes[0], input_sizes[1], input_sizes[2], input_sizes[3]);
    auto [out_channels, _, kernel_h, kernel_w] = std::make_tuple(
        kernel_sizes[0], kernel_sizes[1], kernel_sizes[2], kernel_sizes[3]);
    uint64_t output_h =
        (input_h + 2 * padding[0] - dilation[0] * (kernel_h - 1) - 1) /
            stride[0] +
        1;
    uint64_t output_w =
        (input_w + 2 * padding[1] - dilation[1] * (kernel_w - 1) - 1) /
            stride[1] +
        1;

    return conv2d_multiply_factor * minibatch * output_h * output_w * kernel_h *
        kernel_w * in_channels * out_channels / groups;
  } else if (op_name == kMMOp || op_name == kAddMMOp) {
    if (extra_args.find(kMat1Size) == extra_args.end() ||
        extra_args.find(kMat2Size) == extra_args.end()) {
      TORCH_WARN(
          "Calculating flops for ",
          op_name,
          " requires mat1_size and mat2_size in saved arguments.");
      return 0;
    }
    auto mat1_sizes_ref = extra_args.at(kMat1Size);
    auto mat2_sizes_ref = extra_args.at(kMat2Size);
    if (!mat1_sizes_ref.isIntList() || !mat2_sizes_ref.isIntList()) {
      TORCH_WARN(
          "Failed to compute flops for op ",
          op_name,
          " because it requires mat1_size and mat2_size to be IntList.");
      return 0;
    }

    const auto mat1_size = mat1_sizes_ref.toDimVector();
    const auto mat2_size = mat2_sizes_ref.toDimVector();
    if (mat1_size.empty()) {
      return 0;
    }

    int64_t overlap_dim = mat1_size.back();
    if (overlap_dim == 0) {
      return 0;
    }

    const uint64_t gemm_multiply_factor = 2;
    uint64_t flops = 1;
    for (int64_t dim : mat1_size) {
      flops *= dim;
    }
    flops /= overlap_dim;
    for (int64_t dim : mat2_size) {
      flops *= dim;
    }
    flops *= gemm_multiply_factor;
    return flops;
  } else if (op_name == kBMMOp || op_name == kBAddBMMOp) {
    if (extra_args.find(kMat1Size) == extra_args.end() ||
        extra_args.find(kMat2Size) == extra_args.end()) {
      TORCH_WARN(
          "Calculating flops for ",
          op_name,
          " requires mat1_size and mat2_size in saved arguments.");
      return 0;
    }
    auto mat1_sizes_ref = extra_args.at(kMat1Size);
    auto mat2_sizes_ref = extra_args.at(kMat2Size);
    if (!mat1_sizes_ref.isIntList() || !mat2_sizes_ref.isIntList()) {
      TORCH_WARN(
          "Failed to compute flops for op ",
          op_name,
          " because it requires mat1_size and mat2_size to be IntList.");
      return 0;
    }

    const auto mat1_size = mat1_sizes_ref.toDimVector();
    const auto mat2_size = mat2_sizes_ref.toDimVector();
    if (mat1_size.empty()) {
      return 0;
    }

    int64_t batch_size = mat1_size.front();
    if (batch_size == 0) {
      return 0;
    }

    int64_t overlap_dim = mat1_size.back();
    if (overlap_dim == 0) {
      return 0;
    }

    const uint64_t gemm_multiply_factor = 2;
    uint64_t flops = 1;
    for (int64_t dim : mat1_size) {
      flops *= dim;
    }
    flops /= overlap_dim;
    flops /= batch_size;
    for (int64_t dim : mat2_size) {
      flops *= dim;
    }
    flops *= gemm_multiply_factor;
    return flops;
  } else if (op_name == kMulOp) {
    if (extra_args.find(kMatSize) == extra_args.end()) {
      TORCH_WARN(
          "Calculating flops for aten::mul.Tensor requires mat_size in saved arguments.");
      return 0;
    }
    auto mat_sizes = extra_args.at(kMatSize);
    if (!mat_sizes.isIntList()) {
      TORCH_WARN(
          "Failed to compute flops for op aten::mul because it requires mat_size to be IntList.");
      return 0;
    }

    const auto mat_size = mat_sizes.toDimVector();
    uint64_t flops = 1;
    for (int64_t dim : mat_size) {
      flops *= dim;
    }
    return flops;
  } else if (op_name == kAddOp) {
    if (extra_args.find(kMatSize) == extra_args.end()) {
      TORCH_WARN(
          "Calculating flops for aten::add.Tensor requires mat_size in saved arguments.");
      return 0;
    }
    auto mat_sizes = extra_args.at(kMatSize);
    if (!mat_sizes.isIntList()) {
      TORCH_WARN(
          "Failed to compute flops for op aten::add because it requires mat_size to be IntList.");
      return 0;
    }

    const auto mat_size = mat_sizes.toDimVector();
    uint64_t flops = 1;
    for (int64_t dim : mat_size) {
      flops *= dim;
    }
    return flops;
  }
  return 0;
}

// A function that takes an IValue
// and returns a conventional string representation of the IValue
// Currently it returns int representation of the last 20 bits of the address
// value
int getTensorStartHint(const at::Tensor& t) {
  const auto tensor_impl = t.unsafeGetTensorImpl();
  uintptr_t storage_addr = 0;
  storage_addr = reinterpret_cast<uintptr_t>(tensor_impl->storage().data());
  int last_bits = static_cast<int>(storage_addr & 0xFFFFF);
  return last_bits;
}

bool checkFunctionOutputsForLogging(const at::RecordFunction& fn) {
  const auto& outputs = fn.outputs();
  auto num_outputs = fn.num_outputs();
  VLOG(2) << "outputs: " << num_outputs << " " << outputs.size() << '\n';
  // We have two cases: for unboxed kernel, we have num_outputs ==
  // outputs.size() for boxed kernel using stack, there could be more elements
  // on the stack from previous ops.
  // TORCH_INTERNAL_ASSERT(num_outputs <= outputs.size());
  if (num_outputs > outputs.size()) {
    return false;
  }
  return true;
}

bool checkFunctionInputsForLogging(const at::RecordFunction& fn) {
  auto num_inputs = fn.num_inputs();
  const auto inputs = fn.inputs();
  VLOG(2) << "inputs: " << num_inputs << " " << inputs.size() << '\n';
  // We have two cases: for unboxed kernel, we have num_inputs ==
  // inputs.size() for boxed kernel using stack, there could be more elements
  // on the stack from previous ops.
  // TORCH_INTERNAL_ASSERT(num_inputs <= inputs.size());
  if (num_inputs > inputs.size()) {
    return false;
  }
  return true;
}
} // namespace torch::profiler::impl
