// Standalone Python extension (torch._nccl_ep) for the NCCL EP bindings.
// Loaded lazily by torch.distributed._token_switch, which sets JIT paths.
#include <torch/csrc/distributed/c10d/symm_mem/nccl_ep.hpp>
#include <torch/csrc/utils/pybind.h>

#include <pybind11/pybind11.h>

namespace py = pybind11;

PYBIND11_MODULE(_nccl_ep, m) {
  using namespace c10d::nccl_ep;

  py::enum_<NcclEpLayout>(m, "Layout")
      .value("FLAT", NcclEpLayout::Flat)
      .value("EXPERT_MAJOR", NcclEpLayout::ExpertMajor)
      .value("RANK_MAJOR", NcclEpLayout::RankMajor);

  py::class_<NcclEpGroup, c10::intrusive_ptr<NcclEpGroup>>(m, "_NcclEpGroup")
      .def_static(
          "create",
          &nccl_ep_create_group,
          py::arg("pg"),
          py::arg("num_experts"),
          py::arg("max_dispatch_tokens_per_rank"),
          py::arg("max_recv_tokens_per_rank"),
          py::arg("max_token_bytes"));

  py::class_<NcclEpHandle, c10::intrusive_ptr<NcclEpHandle>>(m, "_NcclEpHandle")
      .def_static(
          "create",
          &nccl_ep_create_handle,
          py::arg("group"),
          py::arg("topk_idx"),
          py::arg("recv_expert_counter") = py::none(),
          py::arg("layout"))
      .def("get_num_recv_tokens", &nccl_ep_handle_get_num_recv_tokens);

  m.def(
      "_nccl_ep_dispatch",
      &nccl_ep_dispatch,
      py::arg("handle"),
      py::arg("tokens"),
      py::arg("topk_weights"),
      py::arg("out_tokens"),
      py::arg("out_topk_weights"),
      py::arg("out_topk_idx"));

  m.def(
      "_nccl_ep_combine",
      &nccl_ep_combine,
      py::arg("handle"),
      py::arg("expert_tokens"),
      py::arg("out_tokens"));
}
