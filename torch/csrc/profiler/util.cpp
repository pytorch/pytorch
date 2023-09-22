#include <torch/csrc/autograd/function.h>
#include <torch/csrc/profiler/kineto_shim.h>
#include <torch/csrc/profiler/util.h>

#include <c10/util/ArrayRef.h>
#include <c10/util/irange.h>
#include <fmt/format.h>

#ifdef USE_KINETO
#include <libkineto.h>
#endif

namespace torch {
namespace profiler {
namespace impl {

ApproximateClockToUnixTimeConverter::ApproximateClockToUnixTimeConverter()
    : start_times_(measurePairs()) {}

ApproximateClockToUnixTimeConverter::UnixAndApproximateTimePair
ApproximateClockToUnixTimeConverter::measurePair() {
  // Take a measurement on either side to avoid an ordering bias.
  auto fast_0 = getApproximateTime();
  auto wall = std::chrono::system_clock::now();
  auto fast_1 = getApproximateTime();

  TORCH_INTERNAL_ASSERT(fast_1 >= fast_0, "getCount is non-monotonic.");
  auto t = std::chrono::duration_cast<std::chrono::nanoseconds>(
      wall.time_since_epoch());

  // `x + (y - x) / 2` is a more numerically stable average than `(x + y) / 2`.
  return {t.count(), fast_0 + (fast_1 - fast_0) / 2};
}

ApproximateClockToUnixTimeConverter::time_pairs
ApproximateClockToUnixTimeConverter::measurePairs() {
  static constexpr auto n_warmup = 5;
  for (C10_UNUSED const auto _ : c10::irange(n_warmup)) {
    getApproximateTime();
    steady_clock_t::now();
  }

  time_pairs out;
  for (const auto i : c10::irange(out.size())) {
    out[i] = measurePair();
  }
  return out;
}

std::function<time_t(approx_time_t)> ApproximateClockToUnixTimeConverter::
    makeConverter() {
  auto end_times = measurePairs();

  // Compute the real time that passes for each tick of the approximate clock.
  std::array<long double, replicates> scale_factors{};
  for (const auto i : c10::irange(replicates)) {
    auto delta_ns = end_times[i].t_ - start_times_[i].t_;
    auto delta_approx = end_times[i].approx_t_ - start_times_[i].approx_t_;
    scale_factors[i] = (double)delta_ns / (double)delta_approx;
  }
  std::sort(scale_factors.begin(), scale_factors.end());
  long double scale_factor = scale_factors[replicates / 2 + 1];

  // We shift all times by `t0` for better numerics. Double precision only has
  // 16 decimal digits of accuracy, so if we blindly multiply times by
  // `scale_factor` we may suffer from precision loss. The choice of `t0` is
  // mostly arbitrary; we just need a factor that is the correct order of
  // magnitude to bring the intermediate values closer to zero. We are not,
  // however, guaranteed that `t0_approx` is *exactly* the getApproximateTime
  // equivilent of `t0`; it is only an estimate that we have to fine tune.
  auto t0 = start_times_[0].t_;
  auto t0_approx = start_times_[0].approx_t_;
  std::array<double, replicates> t0_correction{};
  for (const auto i : c10::irange(replicates)) {
    auto dt = start_times_[i].t_ - t0;
    auto dt_approx =
        (double)(start_times_[i].approx_t_ - t0_approx) * scale_factor;
    t0_correction[i] = dt - (time_t)dt_approx;
  }
  t0 += t0_correction[t0_correction.size() / 2 + 1];

  return [=](approx_time_t t_approx) {
    // See above for why this is more stable than `A * t_approx + B`.
    return (time_t)((double)(t_approx - t0_approx) * scale_factor) + t0;
  };
}

namespace {
c10::optional<bool> soft_assert_raises_;
} // namespace

void setSoftAssertRaises(c10::optional<bool> value) {
  soft_assert_raises_ = value;
}

bool softAssertRaises() {
  return soft_assert_raises_.value_or(false);
}

void logSoftAssert(
    const char* func,
    const char* file,
    uint32_t line,
    const char* cond,
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
    const char* func,
    const char* file,
    uint32_t line,
    const char* cond,
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
    str = fmt::format("{}[", str);
    for (const auto s_idx : c10::irange(shapes[t_idx].size())) {
      if (s_idx > 0) {
        str = fmt::format("{}, ", str);
      }
      str = fmt::format("{}{}", str, shapes[t_idx][s_idx]);
    }
    str = fmt::format("{}]", str);
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

static bool validateInput(
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
    return false;
  }
  for (auto index : should_be_tensor) {
    if (!inputs[index].isTensor()) {
      ss << "Failed to save extra arguments for flops computation of op "
         << op_name << ", input[" << index << "] must be a tensor.";
      TORCH_WARN(ss.str());
      return false;
    }
  }
  return true;
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
    bool check = validateInput(fname, kConv2dGroups + 1, inputs, {0, 1});
    if (!check) {
      return map;
    }

    at::Tensor input = inputs[0].toTensor();
    at::Tensor weight = inputs[1].toTensor();
    if (weight.sizes().size() != 4) {
      TORCH_WARN(
          "Failed to compute flops for op aten::conv2d because it requires a 4D kernel tensor.");
      return map;
    }
    map[kInputSize] = at::IValue(input.sizes());
    map[kWeightSize] = at::IValue(weight.sizes());
    map[kStride] = inputs[kConv2dStride];
    map[kPadding] = inputs[kConv2dPadding];
    map[kDilation] = inputs[kConv2dDilation];
    map[kGroups] = inputs[kConv2dGroups];
  } else if (fname == kMMOp) {
    bool check = validateInput(fname, 2, inputs, {0, 1});
    if (!check) {
      return map;
    }

    at::Tensor left = inputs[0].toTensor();
    at::Tensor right = inputs[1].toTensor();
    map[kMat1Size] = at::IValue(left.sizes());
    map[kMat2Size] = at::IValue(right.sizes());
  } else if (fname == kAddMMOp) {
    bool check = validateInput(fname, 3, inputs, {0, 1, 2});
    if (!check) {
      return map;
    }

    // Exact FLOP count depends on scaling factors alpha and beta but
    // just assume these are +=1.
    // (similar to http://www.netlib.org/lapack/lawnspdf/lawn41.pdf,
    // "Operations Count for the BLAS and LAPACK", Table 3, SGEMM)
    at::Tensor left = inputs[1].toTensor();
    at::Tensor right = inputs[2].toTensor();
    map[kMat1Size] = at::IValue(left.sizes());
    map[kMat2Size] = at::IValue(right.sizes());
  } else if (fname == kMulOp) {
    bool check = validateInput(fname, 1, inputs, {0});
    if (!check) {
      return map;
    }

    at::Tensor mat = inputs[0].toTensor();
    map[kMatSize] = at::IValue(mat.sizes());
  } else if (fname == kAddOp) {
    bool check = validateInput(fname, 1, inputs, {0});
    if (!check) {
      return map;
    }

    at::Tensor mat = inputs[0].toTensor();
    map[kMatSize] = at::IValue(mat.sizes());
  } else if (fname == kBMMOp) {
    bool check = validateInput(fname, 2, inputs, {0, 1});
    if (!check) {
      return map;
    }

    at::Tensor left = inputs[0].toTensor();
    at::Tensor right = inputs[1].toTensor();
    map[kMat1Size] = at::IValue(left.sizes());
    map[kMat2Size] = at::IValue(right.sizes());
  } else if (fname == kBAddBMMOp) {
    bool check = validateInput(fname, 3, inputs, {0, 1, 2});
    if (!check) {
      return map;
    }

    // Exact FLOP count depends on scaling factors alpha and beta but
    // just assume these are +=1.
    // (similar to http://www.netlib.org/lapack/lawnspdf/lawn41.pdf,
    // "Operations Count for the BLAS and LAPACK", Table 3, SGEMM)
    at::Tensor left = inputs[1].toTensor();
    at::Tensor right = inputs[2].toTensor();
    map[kMat1Size] = at::IValue(left.sizes());
    map[kMat2Size] = at::IValue(right.sizes());
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
    uint64_t minibatch = 0, in_channels = 0, input_h = 0, input_w = 0;
    uint64_t out_channels = 0, kernel_h = 0, kernel_w = 0;
    const uint64_t conv2d_multiply_factor = 2;
    std::tie(minibatch, in_channels, input_h, input_w) = std::make_tuple(
        input_sizes[0], input_sizes[1], input_sizes[2], input_sizes[3]);
    std::tie(out_channels, std::ignore, kernel_h, kernel_w) = std::make_tuple(
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

} // namespace impl
} // namespace profiler
} // namespace torch
