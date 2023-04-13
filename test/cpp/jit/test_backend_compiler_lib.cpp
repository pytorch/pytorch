#include <ATen/Utils.h>
#include <c10/core/TensorImpl.h>
#include <torch/csrc/jit/backends/backend.h>
#include <torch/csrc/jit/backends/backend_exception.h>

#ifndef NO_PROFILING
#include <torch/csrc/jit/mobile/profiler_edge.h>
#endif

namespace torch {
namespace jit {

// Implementation of a PyTorch Backend that can process, compile and execute
// TorchScript Modules composed of 'add' and 'sub' operators. It just supports
// for modules that implement a sum or subtraction of 2 inputs (i.e. in1 + in2
// or in1 - in2). Hence the methods of the models expect exactly 2 inputs of
// type Tensor. This backend is used to demonstrate the flow of compilation and
// execution with minimum amount of work. It's not intended to a practical
// backend that can be used for actual inference.

// Implementation details:
//
// Compilation
// 1. A backend with minimum compilation features, "backend_with_compiler_demo"
// is added.
// 2. The compilation happens AOT in the preprocess function registered to this
// backend.
// 3. Compiled results are stored in a string blob for each method. They are
// serialized to the lowered module with __getstate__ function.
// 4. Error message with model source code is thrown, for features not handled
// by the backend compiler.
//
// Runtime
// 1. The compiled blob is loaded in __setstate__ method.
// 2. The compile function of the backend: parse the preprocessed blob to the
// format (a list of tokens) that the backend can understand.
// 3. The execute function of the backend executes the specified method
// (handle).

namespace {
std::vector<std::tuple<std::string, int64_t>> parseMethodHandle(
    const std::string& blob) {
  std::vector<std::tuple<std::string, int64_t>> result;
  std::stringstream s_stream(blob);
  constexpr char debug_handle_token[] = "<debug_handle>";
  while (s_stream.good()) {
    std::string substr;
    getline(s_stream, substr, ',');
    auto debug_handle_pos = substr.find(debug_handle_token);
    int64_t debug_handle{-1};
    auto instruction = substr.substr(0);
    if (debug_handle_pos != std::string::npos) {
      instruction = substr.substr(0, debug_handle_pos);
      debug_handle = stoi(substr.substr(debug_handle_pos + 14));
    }
    result.push_back(std::make_tuple(instruction, debug_handle));
  }
  return result;
}

float* float_data_ptr(const at::Tensor& t) {
  return t.data_ptr<float>();
}
} // namespace

class BackendWithCompiler : public PyTorchBackendInterface {
 public:
  // Constructor.
  // NOLINTNEXTLINE(modernize-use-equals-default)
  explicit BackendWithCompiler() {}
  // NOLINTNEXTLINE(modernize-use-override)
  virtual ~BackendWithCompiler() = default;

  bool is_available() override {
    return true;
  }

  // Since the actual compilation is done AOT for this backend, compile just
  // forwards everything along. In a non toy setup this could grab information
  // from that runtime that might be relevant to execute, such as build flags
  // the resolution of the devices camera, or basically any runtime specific
  // information that wouldnt be available server side where preprocess is
  // called.
  c10::impl::GenericDict compile(
      c10::IValue processed,
      c10::impl::GenericDict method_compile_spec) override {
    auto dict = processed.toGenericDict();
    auto handles =
        c10::Dict<std::string, std::vector<std::tuple<std::string, int64_t>>>();
    for (const auto& kv : dict) {
      auto tokens = parseMethodHandle(kv.value().toStringRef());
      handles.insert(kv.key().toStringRef(), tokens);
    }
    return c10::impl::toGenericDict(handles);
  }

  // Function that actually executes the model in the backend. Here there is
  // nothing to dispatch to, so the backend is implemented locally within
  // execute and it only supports add, subtract, and constant. In a non toy
  // backend you can imagine how this function could be used to actually
  // dispatch the inputs to the relevant backend/device.
  c10::impl::GenericList execute(
      c10::IValue
          handle, // example: [('prim::Constant#1', 14), ('aten::add', 15)]
      c10::impl::GenericList inputs) override {
    TORCH_INTERNAL_ASSERT(inputs.size() == 2);
    c10::IValue val0 = inputs[0];
    at::Tensor x = val0.toTensor();
    c10::IValue val1 = inputs[1];
    at::Tensor h = val1.toTensor();
    std::vector<std::tuple<int64_t, int64_t, std::string>> op_runtimes_us;
    op_runtimes_us.reserve(handle.toList().size());

    c10::List<at::Tensor> output_list;
#ifndef NO_PROFILING
    auto start_us = torch::profiler::impl::getTime() / 1000;
#endif
    for (const auto& token : handle.toList()) {
      IValue val = token;
      auto instruction = val.toTupleRef().elements()[0].toStringRef();
      auto debug_handle = val.toTupleRef().elements()[1].toInt();
#ifndef NO_PROFILING
      auto start_time_us = torch::profiler::impl::getTime() / 1000;
#endif
      try {
        if (instruction.rfind("prim::Constant", 0) == 0) {
          // 15 is the length of 'prim::Constant#' the constant val comes after
          TORCH_CHECK(
              instruction.size() > 15,
              "Constant value is expected in ",
              instruction);
          // NOLINTNEXTLINE(cppcoreguidelines-avoid-magic-numbers)
          auto sub = instruction.substr(15);
        } else if (instruction == "aten::add" || instruction == "aten::sub") {
          TORCH_CHECK(x.sizes() == h.sizes());
          if (x.dim() > 1 || (x.dim() == 1 && x.size(0) > 1)) {
            TORCH_WARN(
                "Only the first elements of the tensors are added or subbed.");
          }
          TORCH_CHECK(
              (x.scalar_type() == c10::ScalarType::Float &&
               h.scalar_type() == c10::ScalarType::Float),
              "Only float tensors are compatible for add and sub.");
          at::Tensor y = at::detail::empty_cpu(x.sizes(), at::kFloat);
          auto x_ptr = float_data_ptr(x);
          auto h_ptr = float_data_ptr(h);
          auto y_ptr = float_data_ptr(y);
#ifndef NO_PROFILING
          RECORD_BACKEND_MEMORY_EVENT_TO_EDGE_PROFILER(
              x_ptr,
              x.numel() * sizeof(float),
              x.numel() * sizeof(float),
              x.numel() * sizeof(float) + y.numel() * sizeof(float) +
                  h.numel() * sizeof(float),
              c10::Device(c10::kCPU));
#endif
          if (instruction == "aten::add") {
            y_ptr[0] = x_ptr[0] + h_ptr[0];
          } else {
            y_ptr[0] = x_ptr[0] - h_ptr[0];
          }
          output_list.emplace_back(y);
        } else {
          TORCH_CHECK(
              false,
              "Instruction, ",
              instruction,
              " is not supported. ",
              "Contact the backend POC for details. ");
        }
      } catch (c10::Error& e) {
        TORCH_DELEGATED_BACKEND_THROW(false, e.what(), debug_handle);
      }
#ifndef NO_PROFILING
      auto end_time_us = torch::profiler::impl::getTime() / 1000;
      auto duration = end_time_us - start_time_us;
      op_runtimes_us.emplace_back(duration, debug_handle, instruction);
#endif
    }
#ifndef NO_PROFILING
    for (const auto& tup : op_runtimes_us) {
      RECORD_BACKEND_EVENT_TO_EDGE_PROFILER(
          start_us,
          start_us + std::get<0>(tup),
          std::get<1>(tup),
          std::get<2>(tup),
          "test_backend");
      start_us = start_us + std::get<0>(tup);
    }
#endif
    return c10::impl::toList(output_list);
  }
};

namespace {
constexpr auto backend_name = "backend_with_compiler_demo";
static auto cls = torch::jit::backend<BackendWithCompiler>(backend_name);
} // namespace

} // namespace jit
} // namespace torch
