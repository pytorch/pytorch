#include <torch/csrc/distributed/c10d/frontend_cuda.hpp>

#ifdef USE_C10D_NCCL

#include <c10/util/Exception.h>
#include <c10d/ProcessGroupNCCL.hpp>
#include <torch/csrc/distributed/c10d/frontend.hpp>
#include <torch/csrc/distributed/c10d/quantization/quantization_gpu.h>
#include <torch/library.h>

namespace c10d {

void initCustomClassBindingsNccl() {
  // XXX: Ideally the Options of ProcessGroupNCCL should be
  // bound using `def_readwrite` like in pybind11, but we
  // didn't do that because: 1. no milisecond support yet
  // 2. no def_readwrite or property support yet.
  // TODO: make this binding the same as pybind11
  static const auto ProcessGroupNCCLOptionsTorchBind =
      torch::class_<::c10d::ProcessGroupNCCL::Options>(
          "dist_c10d", "ProcessGroupNCCLOptions")
          .def(torch::init([](int64_t timeout, bool isHighPriorityStream) {
            auto opTimeout = std::chrono::milliseconds(timeout);
            auto opts =
                ::c10d::ProcessGroupNCCL::Options::create(isHighPriorityStream);
            opts->timeout = opTimeout;
            return opts;
          }));

  static const auto ProcessGroupNCCLTorchBind =
      torch::class_<::c10d::ProcessGroupNCCL>("dist_c10d", "ProcessGroupNCCL")
          .def_pickle(
              [](const c10::intrusive_ptr<::c10d::ProcessGroupNCCL>& self) {
                auto base_process_group =
                    ::c10::static_intrusive_pointer_cast<::c10d::ProcessGroup>(
                        self);
                auto name =
                    ::c10d::DistributedC10d::get()->getNameOfProcessGroup(self);
                return std::vector<std::string>{name};
              },
              [](std::vector<std::string> state) {
                TORCH_CHECK(
                    state.size() == 1,
                    "Expecting exactly 1 state when restoring ProcessGroupNCCL, got: ",
                    state.size());
                const auto& process_group_name = state.front();
                auto base_process_group =
                    ::c10d::DistributedC10d::get()->getProcessGroupByName(
                        process_group_name);
                TORCH_CHECK(
                    base_process_group.defined(),
                    "Needed process group not found, ",
                    "please create a process group with name: ",
                    process_group_name);
                c10::intrusive_ptr<::c10d::ProcessGroupNCCL>
                    process_group_nccl = ::c10::dynamic_intrusive_pointer_cast<
                        ::c10d::ProcessGroupNCCL>(base_process_group);
                TORCH_CHECK(
                    process_group_nccl.defined(),
                    "Process group ",
                    process_group_name,
                    " isn't configured for NCCL backend");
                return process_group_nccl;
              })
          .def(torch::init(
              [](const c10::intrusive_ptr<::c10d::Store>& store,
                 int64_t rank,
                 int64_t size,
                 c10::intrusive_ptr<::c10d::ProcessGroupNCCL::Options> options,
                 const std::string& name) {
                auto pg = c10::make_intrusive<::c10d::ProcessGroupNCCL>(
                    store, rank, size, options);
                ::c10d::DistributedC10d::get()->registerProcessGroupName(
                    pg, name);
                return pg;
              }))
          .def(
              "alltoall_base",
              [](const c10::intrusive_ptr<::c10d::ProcessGroupNCCL>& self,
                 at::Tensor output,
                 at::Tensor input,
                 std::vector<int64_t> outputSplitSizes,
                 std::vector<int64_t> inputSplitSizes) {
                return self->alltoall_base(
                    output,
                    input,
                    outputSplitSizes,
                    inputSplitSizes,
                    ::c10d::AllToAllOptions());
              })
          .def(
              "size",
              [](const c10::intrusive_ptr<::c10d::ProcessGroupNCCL>& self) {
                return (int64_t)self->getSize();
              })
          .def(
              "rank",
              [](const c10::intrusive_ptr<::c10d::ProcessGroupNCCL>& self) {
                return (int64_t)self->getRank();
              });
}

namespace {
struct RealNCCLProcessGroupProvider : public NCCLProcessGroupProvider {
  c10::intrusive_ptr<ProcessGroup> get(
      c10::intrusive_ptr<PrefixStore> prefix_store,
      int64_t rank,
      int64_t world_size,
      std::chrono::milliseconds timeout) const override {
    auto options = ProcessGroupNCCL::Options::create();
    options->is_high_priority_stream = false;
    options->timeout = timeout;
    return c10::make_intrusive<ProcessGroupNCCL>(
        prefix_store, rank, world_size, options);
  }
};

struct RegisterNCCLProcessGroupProvider {
  RegisterNCCLProcessGroupProvider() {
    static RealNCCLProcessGroupProvider provider;
    registerNCCLProcessGroupProvider(&provider);
  }
};

RegisterNCCLProcessGroupProvider reg;

} // namespace
#define DISPATCH_TO_CUDA(name, function) \
    m.impl(name, torch::dispatch(c10::DispatchKey::CUDA, TORCH_FN(function)))
TORCH_LIBRARY_IMPL(q, CUDA, m) {
    DISPATCH_TO_CUDA("_Bfloat16QuantizedToFloat", ::torch::distributed::c10d::quantization::_bfloat16_to_float_cuda);
    DISPATCH_TO_CUDA("_FloatToBfloat16Quantized", ::torch::distributed::c10d::quantization::_float_to_bfloat16_cuda);
}
} // namespace c10d

#endif // USE_C10D_NCCL
