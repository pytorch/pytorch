// Standalone Python extension (torch._nccl_ep) for the NCCL EP bindings.
//
// The EP code lives in this optional extension -- which NEEDED-links libnccl_ep
// -- rather than in libtorch_cuda / libtorch_python's init.cpp. It is imported
// lazily by torch/distributed/_token_switch.py, so the normal Python
// extension-import machinery loads libnccl_ep (and raises ImportError if the
// optional nccl4py wheel that provides it is absent). libtorch_cuda therefore
// never references ncclEp* and torch imports with or without nccl4py.
#include <torch/csrc/distributed/c10d/symm_mem/nccl_ep.hpp>
#include <torch/csrc/utils/pybind.h>

#include <pybind11/pybind11.h>

#include <cstdlib>

namespace py = pybind11;

PYBIND11_MODULE(_nccl_ep, m) {
  using namespace c10d::nccl_ep;

#ifdef NCCL_EP_JIT_HOME
  // Point nccl-ep's runtime JIT at the in-tree headers baked at build time
  // (NCCL_EP_HOME -> include/nccl_ep, NCCL_HOME -> include/nccl.h, both under
  // this dir). overwrite=0 so an explicit user setting wins.
  setenv("NCCL_EP_HOME", NCCL_EP_JIT_HOME, /*overwrite=*/0);
  setenv("NCCL_HOME", NCCL_EP_JIT_HOME, /*overwrite=*/0);
#endif

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
          py::arg("recv_expert_counter") = py::none())
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
