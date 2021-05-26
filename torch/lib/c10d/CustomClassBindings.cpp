#include <c10d/Store.hpp>
#include <c10d/FileStore.hpp>
#include <c10d/TCPStore.hpp>
#include <c10d/PrefixStore.hpp>
#include <c10d/ProcessGroup.hpp>
#include <c10d/frontend.hpp>
#ifdef USE_C10D_NCCL
#include <c10d/ProcessGroupNCCL.hpp>
#endif

#include <torch/custom_class.h>

namespace c10 {
namespace {


// NOTE: Below are TorchBind bindings for c10d, these bindings will
// live together with those pybind11 bindings above until we resolve
// all the TorchBind issues and merge these two together. we shouldn't
// document this until we finish the migration.

static const auto StoreTorchBind =
    torch::class_<::c10d::Store>("dist_c10d", "Store");

static const auto FileStoreTorchBind =
    torch::class_<::c10d::FileStore>("dist_c10d", "FileStore")
        .def(torch::init([](const std::string& path, int64_t num_workers) {
          return c10::make_intrusive<::c10d::FileStore>(path, num_workers);
        }));

static const auto TCPStoreTorchBind =
    torch::class_<::c10d::TCPStore>("dist_c10d", "TCPStore")
        .def(torch::init([](const std::string& host_name,
                            int64_t port,
                            int64_t world_size,
                            bool is_master,
                            int64_t timeout) {
          auto timeout_miliseconds = std::chrono::milliseconds(timeout);
          return c10::make_intrusive<::c10d::TCPStore>(
              host_name, port, world_size, is_master, timeout_miliseconds);
        }));

// TODO: This should really take Store as constructor argument instead of
// TCPStore, but the fact that TorchScript does not support polymorphism
// forced us to cast in C++ instead of automatic casting
static const auto PrefixStoreTorchBind =
    torch::class_<::c10d::PrefixStore>("dist_c10d", "PrefixStore")
        .def(torch::init([](const std::string& prefix,
                            const c10::intrusive_ptr<::c10d::Store>& store) {
          return c10::make_intrusive<::c10d::PrefixStore>(prefix, store);
        }));

// Torchbind the ProcessGroup to make it available in TorchScript
static const auto ProcessGroupWorkTorchBind =
    torch::class_<::c10d::ProcessGroup::Work>("dist_c10d", "Work")
        .def(torch::init<>())
        .def(
            "wait",
            [](const c10::intrusive_ptr<::c10d::ProcessGroup::Work>& work)
                -> bool {
              // TODO: make std::chrono::millisecond works with TorchBind to
              // provide the full API in python
              return work->wait();
            })
        .def("result", &::c10d::ProcessGroup::Work::result);

// TODO: Support argument names in Python API.
static const auto ProcessGroupTorchBind =
    torch::class_<::c10d::ProcessGroup>("dist_c10d", "ProcessGroup")
        .def_pickle(
            [](const c10::intrusive_ptr<::c10d::ProcessGroup>& self) {
              auto name =
                  ::c10d::DistributedC10d::get()->getNameOfProcessGroup(self);
              return std::vector<std::string>{name};
            },
            [](std::vector<std::string> state) {
              TORCH_CHECK(
                  state.size() == 1,
                  "Expecting exactly 1 state when restoring ProcessGroup, got: ",
                  state.size());
              const auto& process_group_name = state.front();
              auto process_group =
                  ::c10d::DistributedC10d::get()->getProcessGroupByName(
                      process_group_name);
              TORCH_CHECK(
                  process_group.defined(),
                  "Needed process group not found, ",
                  "please create a process group with name: ",
                  process_group_name);
              return process_group;
            })
        .def(
            "rank",
            [](const c10::intrusive_ptr<::c10d::ProcessGroup>& self) {
              return static_cast<int64_t>(self->getRank());
            })
        .def(
            "size",
            [](const c10::intrusive_ptr<::c10d::ProcessGroup>& self) {
              return static_cast<int64_t>(self->getSize());
            })
        // TODO: make BroadcastOptions compatible with TorchBind to provide
        // the full API in python.
        /*
        // TODO: Enable this method when TorchBind supports overloading.
        .def(
            "broadcast",
            [](const c10::intrusive_ptr<::c10d::ProcessGroup>& self,
               std::vector<at::Tensor> data) { return self->broadcast(data); })
        */
        .def(
            "broadcast",
            [](const c10::intrusive_ptr<::c10d::ProcessGroup>& self,
               at::Tensor tensor,
               int64_t rootRank) {
              ::c10d::BroadcastOptions opts;
              opts.rootRank = rootRank;
              std::vector<at::Tensor> tensors = {std::move(tensor)};
              return self->broadcast(tensors, opts);
            })
        // TODO: make AllreduceOptions compatible with TorchBind to provide
        // the full API in python.
        .def(
            "allreduce",
            [](const c10::intrusive_ptr<::c10d::ProcessGroup>& self,
               std::vector<at::Tensor> tensors) {
              return self->allreduce(tensors);
            })
        /*
        // TODO: Enable these methods when TorchBind supports overloading.
        // TODO: Enable these methods when ReduceOp can be torchbinded.
        .def(
            "allreduce",
            [](c10::intrusive_ptr<::c10d::ProcessGroup>& self,
                std::vector<at::Tensor>& tensors,
                c10::intrusive_ptr<::c10d::ReduceOp> op) {
                    ::c10d::AllreduceOptions opts;
                    opts.reduceOp = *op;
                    return self->allreduce(tensors, opts);
                }
        )
        .def(
            "allreduce",
            [](const c10::intrusive_ptr<::c10d::ProcessGroup>& self,
               at::Tensor& tensor,
               c10::intrusive_ptr<::c10d::ReduceOp> op) {
                    ::c10d::AllreduceOptions opts;
                    opts.reduceOp = *op;
                    std::vector<at::Tensor> tensors = {tensor};
                    return self->allreduce(tensors, opts);
               }
         )
        */
        // TODO: make AllreduceCoalescedOptions compatible with TorchBind to
        // provide the full API in python.
        .def(
            "allreduce_coalesced",
            [](const c10::intrusive_ptr<::c10d::ProcessGroup>& self,
               std::vector<at::Tensor> tensors) {
              ::c10d::AllreduceCoalescedOptions opts;
              return self->allreduce_coalesced(tensors, opts);
            })
        .def(
            "reduce",
            [](const c10::intrusive_ptr<::c10d::ProcessGroup>& self,
               std::vector<at::Tensor> tensors) {
              ::c10d::ReduceOptions opts;
              return self->reduce(tensors, opts);
            })
        /*
        // TODO: Enable this when c10d::ReduceOp is TorchBind compatible.
        .def(
            "reduce",
            [](const c10::intrusive_ptr<::c10d::ProcessGroup>& self,
            at::Tensor& tensor,
            int rootRank,
            c10::intrusive_ptr<::c10d::ReduceOp> op) {
            ::c10d::ReduceOptions opts;
            opts.reduceOp = *op;
            opts.rootRank = rootRank;
            std::vector<at::Tensor> tensors = {tensor};
            return self->reduce(tensors, opts);
            })
        */
        .def(
            "allgather",
            [](const c10::intrusive_ptr<::c10d::ProcessGroup>& self,
               std::vector<std::vector<at::Tensor>> outputTensors,
               std::vector<at::Tensor> inputTensors) {
              ::c10d::AllgatherOptions opts;
              return self->allgather(outputTensors, inputTensors, opts);
            })
        /*
        // TODO: Enable these methods when TorchBind supports overloading.
        .def(
            "allgather",
            [](const c10::intrusive_ptr<::c10d::ProcessGroup>& self,
               std::vector<at::Tensor> output,
               at::Tensor input) {
              std::vector<std::vector<at::Tensor>> outputs = {
                  std::move(output)};
              std::vector<at::Tensor> inputs = {std::move(input)};
              ::c10d::AllgatherOptions opts;
              return self->allgather(outputs, inputs, opts);
            })
        */
        .def(
            "allgather_coalesced",
            [](const c10::intrusive_ptr<::c10d::ProcessGroup>& self,
               std::vector<std::vector<at::Tensor>> output_lists,
               std::vector<at::Tensor> input_list) {
              ::c10d::AllgatherOptions opts;
              return self->allgather_coalesced(output_lists, input_list, opts);
            })
        /*
        // TODO: Enable this method when TorchBind supports overloading.
        .def(
            "gather",
            [](const c10::intrusive_ptr<::c10d::ProcessGroup>& self,
               std::vector<std::vector<at::Tensor>> output_tensors,
               std::vector<at::Tensor> input_tensors) {
              ::c10d::GatherOptions opts;
              return self->gather(output_tensors, input_tensors, opts);
            })
        */
        .def(
            "gather",
            [](const c10::intrusive_ptr<::c10d::ProcessGroup>& self,
               std::vector<at::Tensor> output,
               at::Tensor input,
               int64_t rootRank) {
              ::c10d::GatherOptions opts;
              opts.rootRank = rootRank;
              std::vector<std::vector<at::Tensor>> outputs = {
                  std::move(output)};
              std::vector<at::Tensor> inputs = {std::move(input)};
              return self->gather(outputs, inputs, opts);
            })
        /*
        // TODO: Enable this method when TorchBind supports overloading.
        .def(
            "scatter",
            [](const c10::intrusive_ptr<::c10d::ProcessGroup>& self,
               std::vector<at::Tensor> outputTensors,
               std::vector<std::vector<at::Tensor>> inputTensors) {
              ::c10d::ScatterOptions opts;
              self->scatter(outputTensors, inputTensors, opts);
            })
        */
        .def(
            "scatter",
            [](const c10::intrusive_ptr<::c10d::ProcessGroup>& self,
               at::Tensor output,
               std::vector<at::Tensor> input,
               int64_t rootRank) {
              ::c10d::ScatterOptions opts;
              opts.rootRank = rootRank;
              std::vector<std::vector<at::Tensor>> inputs = {std::move(input)};
              std::vector<at::Tensor> outputs = {std::move(output)};
              return self->scatter(outputs, inputs, opts);
            })
        /*
        // TODO: Enable this method when TorchBind supports overloading.
        // TODO: Enable this method when TorchBind supports
        ReduceScatterOptions. .def( "reduce_scatter",
            [](const c10::intrusive_ptr<::c10d::ProcessGroup>& self,
               std::vector<at::Tensor> outputTensors,
               std::vector<std::vector<at::Tensor>> inputTensors) {
              ::c10d::ReduceScatterOptions opts;
              return self->reduce_scatter(outputTensors, inputTensors, opts);
            })
        */
        .def(
            "reduce_scatter",
            [](const c10::intrusive_ptr<::c10d::ProcessGroup>& self,
               at::Tensor output,
               std::vector<at::Tensor> input) {
              std::vector<at::Tensor> outputs = {std::move(output)};
              std::vector<std::vector<at::Tensor>> inputs = {std::move(input)};
              ::c10d::ReduceScatterOptions opts;
              return self->reduce_scatter(outputs, inputs, opts);
            })
        .def(
            "alltoall_base",
            [](const c10::intrusive_ptr<::c10d::ProcessGroup>& self,
               at::Tensor outputTensor,
               at::Tensor inputTensor,
               std::vector<int64_t> outputSplitSizes,
               std::vector<int64_t> inputSplitSizes) {
              ::c10d::AllToAllOptions opts;
              return self->alltoall_base(
                  outputTensor,
                  inputTensor,
                  outputSplitSizes,
                  inputSplitSizes,
                  opts);
            })
        .def(
            "alltoall",
            [](const c10::intrusive_ptr<::c10d::ProcessGroup>& self,
               std::vector<at::Tensor> outputTensors,
               std::vector<at::Tensor> inputTensors) {
              ::c10d::AllToAllOptions opts;
              return self->alltoall(outputTensors, inputTensors, opts);
            })
        .def(
            "send",
            [](const c10::intrusive_ptr<::c10d::ProcessGroup>& self,
               std::vector<at::Tensor> tensors,
               int64_t dstRank,
               int64_t tag) {
              return self->send(
                  tensors, static_cast<int>(dstRank), static_cast<int>(tag));
            })
        .def(
            "recv",
            [](const c10::intrusive_ptr<::c10d::ProcessGroup>& self,
               std::vector<at::Tensor> tensors,
               int64_t srcRank,
               int64_t tag) {
              return self->recv(
                  tensors, static_cast<int>(srcRank), static_cast<int>(tag));
            })
        .def(
            "recv_anysource",
            [](const c10::intrusive_ptr<::c10d::ProcessGroup>& self,
               std::vector<at::Tensor> tensors,
               int64_t tag) {
              return self->recvAnysource(tensors, static_cast<int>(tag));
            })
        .def(
            "barrier",
            [](const c10::intrusive_ptr<::c10d::ProcessGroup>& self) {
              ::c10d::BarrierOptions opts;
              return self->barrier(opts);
            });

#ifdef USE_C10D_NCCL
// XXX: Ideally the Options of ProcessGroupNCCL should be
// bound using `def_readwrite` like in pybind11, but we
// didn't do that because: 1. no milisecond support yet
// 2. no def_readwrite or property support yet.
// TODO: make this binding the same as pybind11
static const auto ProcessGroupNCCLOptionsTorchBind =
    torch::class_<::c10d::ProcessGroupNCCL::Options>(
        "dist_c10d",
        "ProcessGroupNCCLOptions")
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
                  static_intrusive_pointer_cast<::c10d::ProcessGroup>(self);
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
              c10::intrusive_ptr<::c10d::ProcessGroupNCCL> process_group_nccl =
                  dynamic_intrusive_pointer_cast<::c10d::ProcessGroupNCCL>(
                      base_process_group);
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
#endif

static const auto DistributedC10dFrontendTorchBind =
    torch::class_<::c10d::DistributedC10d>("dist_c10d", "frontend")
        .def(torch::init([]() { return ::c10d::DistributedC10d::get(); }))
        .def(
            "new_process_group_helper",
            &::c10d::DistributedC10d::newProcessGroupHelper)
        .def(
            "get_process_group_by_name",
            &::c10d::DistributedC10d::getProcessGroupByName)
        .def(
            "get_name_of_process_group",
            &::c10d::DistributedC10d::getNameOfProcessGroup);

} // namespace
}
