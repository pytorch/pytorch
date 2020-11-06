#include <torch/csrc/autograd/python_cpp_function.h>
#include <torch/csrc/distributed/autograd/autograd.h>
#include <torch/csrc/jit/python/pybind_utils.h>
#include <torch/csrc/python_headers.h>
#include <torch/csrc/utils/object_ptr.h>
#include <torch/csrc/utils/pybind.h>
#include <torch/types.h>

namespace torch {
namespace distributed {
namespace autograd {

namespace {

template <typename T>
using shared_ptr_class_ = py::class_<T, std::shared_ptr<T>>;

PyObject* dist_autograd_init(PyObject* _unused, PyObject* noargs) {
  auto autograd_module =
      THPObjectPtr(PyImport_ImportModule("torch.distributed.autograd"));
  if (!autograd_module) {
    throw python_error();
  }

  auto torch_C_module = THPObjectPtr(PyImport_ImportModule("torch._C"));
  if (!torch_C_module) {
    throw python_error();
  }

  auto torch_C_m = py::handle(torch_C_module).cast<py::module>();
  auto m = torch_C_m.def_submodule("_distributed_autograd", "distributed autograd bindings");

  auto module = py::handle(m).cast<py::module>();

  auto distAutogradContext =
      shared_ptr_class_<DistAutogradContext>(module, "DistAutogradContext")
          .def(
              "_context_id",
              &DistAutogradContext::contextId,
              py::call_guard<py::gil_scoped_release>())
          .def(
              "_recv_functions",
              [](const DistAutogradContext& ctx) {
                std::map<int64_t, py::object> funcs;
                for (const auto& map_entry : ctx.recvFunctions()) {
                  funcs.emplace(
                      map_entry.first,
                      py::reinterpret_steal<py::object>(
                          torch::autograd::functionToPyObject(
                              map_entry.second)));
                }
                return funcs;
              })
          .def(
              "_send_functions",
              [](const ContextPtr& ctx) {
                std::map<int64_t, py::object> funcs;
                for (const auto& map_entry : ctx->sendFunctions()) {
                  funcs.emplace(
                      map_entry.first,
                      py::reinterpret_steal<py::object>(
                          torch::autograd::functionToPyObject(
                              map_entry.second)));
                }
                return funcs;
              })
          .def("_known_worker_ids", &DistAutogradContext::getKnownWorkerIds);

  module.def(
      "_new_context",
      []() -> const ContextPtr {
        return DistAutogradContainer::getInstance().newContext();
      },
      py::return_value_policy::reference);

  module.def(
      "_release_context",
      [](int64_t context_id) {
        return DistAutogradContainer::getInstance().releaseContext(context_id);
      },
      py::call_guard<py::gil_scoped_release>());

  module.def("_get_max_id", []() {
    return DistAutogradContainer::getInstance().getMaxId();
  });

  module.def(
      "_is_valid_context",
      [](int64_t worker_id) {
        DistAutogradContainer::getInstance().isValidContext(worker_id);
      },
      py::call_guard<py::gil_scoped_release>());

  module.def(
      "_retrieve_context",
      [](int64_t context_id) -> const ContextPtr {
        return DistAutogradContainer::getInstance().retrieveContext(context_id);
      },
      py::return_value_policy::reference);

  module.def(
      "_current_context",
      []() -> const ContextPtr {
        return DistAutogradContainer::getInstance().currentContext();
      },
      py::return_value_policy::reference);

  module.def(
      "_init",
      [](int64_t worker_id) { DistAutogradContainer::init(worker_id); },
      py::call_guard<py::gil_scoped_release>());

  module.def(
      "_get_debug_info",
      []() { return DistEngine::getInstance().getDebugInfo(); },
      py::call_guard<py::gil_scoped_release>());

  py::options options;
  options.disable_function_signatures();

  module.def(
      "backward",
      backward,
      R"(
backward(context_id: int, roots: List[Tensor], retain_graph = False) -> None

Kicks off the distributed backward pass using the provided roots. This
currently implements the :ref:`fast-mode-algorithm` which
assumes all RPC messages sent in the same distributed autograd context
across workers would be part of the autograd graph during the backward pass.

We use the provided roots to discover the autograd graph and compute
appropriate dependencies. This method blocks until the entire
autograd computation is done.

We accumulate the gradients in the appropriate
:class:`torch.distributed.autograd.context` on each of the nodes. The autograd
context to be used is looked up given the ``context_id`` that is passed in when
:meth:`torch.distributed.autograd.backward` is called. If there is no valid
autograd context corresponding to the given ID, we throw an error. You can
retrieve the accumulated gradients using the
:meth:`~torch.distributed.autograd.get_gradients` API.

Arguments:
    context_id (int): The autograd context id for which we should retrieve the gradients.
    roots (list): Tensors which represent the roots of the autograd
                  computation. All the tensors should be scalars.
    retain_graph(bool, optional): If False, the graph used to compute the grad
                  will be freed. Note that in nearly all cases setting this
                  option to True is not needed and often can be worked around
                  in a much more efficient way. Usually, you need to set this
                  to True to run backward multiple times.

Example::
    >>> import torch.distributed.autograd as dist_autograd
    >>> with dist_autograd.context() as context_id:
    >>>     pred = model.forward()
    >>>     loss = loss_func(pred, loss)
    >>>     dist_autograd.backward(context_id, loss)
)",
      py::arg("contextId"),
      py::arg("roots"),
      py::arg("retain_graph") = false,
      py::call_guard<py::gil_scoped_release>());

  module.def(
      "get_gradients",
      [](int64_t contextId) -> py::dict {
        const auto& autogradContext =
            DistAutogradContainer::getInstance().retrieveContext(contextId);
        return torch::jit::toPyObject(IValue(autogradContext->getGradients()));
      },
      R"(
get_gradients(context_id: int) -> Dict[Tensor, Tensor]

Retrieves a map from Tensor to the appropriate gradient for that Tensor
accumulated in the provided context corresponding to the given ``context_id``
as part of the distributed autograd backward pass.

Arguments:
    context_id(int): The autograd context id for which we should retrieve the
                     gradients.

Returns:
    A map where the key is the Tensor and the value is the associated gradient
    for that Tensor.

Example::
    >>> import torch.distributed.autograd as dist_autograd
    >>> with dist_autograd.context() as context_id:
    >>>     t1 = torch.rand((3, 3), requires_grad=True)
    >>>     t2 = torch.rand((3, 3), requires_grad=True)
    >>>     loss = t1 + t2
    >>>     dist_autograd.backward(context_id, [loss.sum()])
    >>>     grads = dist_autograd.get_gradients(context_id)
    >>>     print(grads[t1])
    >>>     print(grads[t2])
)",
      py::arg("context_id"));

  Py_RETURN_TRUE;
}
} // namespace

static PyMethodDef methods[] = { // NOLINT
    {"_dist_autograd_init",
     dist_autograd_init,
     METH_NOARGS,
     nullptr},
    {nullptr, nullptr, 0, nullptr}};

PyMethodDef* python_functions() {
  return methods;
}

} // namespace autograd
} // namespace distributed
} // namespace torch
