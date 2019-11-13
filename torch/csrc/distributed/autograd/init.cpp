#include <torch/csrc/autograd/python_cpp_function.h>
#include <torch/csrc/distributed/autograd/context/container.h>
#include <torch/csrc/distributed/autograd/engine/dist_engine.h>
#include <torch/csrc/jit/pybind_utils.h>
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

PyObject* dist_autograd_init(PyObject* /* unused */) {
  auto autograd_module =
      THPObjectPtr(PyImport_ImportModule("torch.distributed.autograd"));
  if (!autograd_module) {
    throw python_error();
  }

  auto module = py::handle(autograd_module).cast<py::module>();

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
              [](const DistAutogradContext& ctx) {
                std::map<int64_t, py::object> funcs;
                for (const auto& map_entry : ctx.sendFunctions()) {
                  funcs.emplace(
                      map_entry.first,
                      py::reinterpret_steal<py::object>(
                          torch::autograd::functionToPyObject(
                              map_entry.second)));
                }
                return funcs;
              })
          .def("_known_worker_ids", [](const DistAutogradContext& ctx) {
            std::vector<rpc::worker_id_t> worker_ids;
            for (const auto worker_id : ctx.getKnownWorkerIds()) {
              worker_ids.push_back(worker_id);
            }
            return worker_ids;
          });

  module.def(
      "_new_context",
      []() -> const DistAutogradContext& {
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
      "_retrieve_context",
      [](int64_t context_id) -> const DistAutogradContext& {
        return DistAutogradContainer::getInstance().retrieveContext(context_id);
      },
      py::return_value_policy::reference);

  module.def(
      "_current_context",
      []() -> const DistAutogradContext& {
        return DistAutogradContainer::getInstance().currentContext();
      },
      py::return_value_policy::reference);

  module.def(
      "_init",
      [](int64_t worker_id) { DistAutogradContainer::init(worker_id); },
      py::call_guard<py::gil_scoped_release>());

  module.def(
      "backward",
      [](const std::vector<torch::Tensor>& roots) {
        torch::autograd::variable_list variables;
        for (const auto& root : roots) {
          variables.emplace_back(root);
        }
        DistEngine::getInstance().execute(variables);
      },
      R"(
Kicks off the distributed backward pass using the provided roots. This
currently implements the "FAST" mode
(see https://github.com/pytorch/pytorch/issues/23110) algorithm which
assumes all RPC messages sent in the same distributed autograd context
across workers would be part of the autograd graph during the backward pass.

We use the provided roots to discover the autograd graph and compute
appropriate dependencies. This method blocks until the entire
autograd computation is done.

We accumulate the gradients in the appropriate "autograd context id" on each
of the nodes. The autograd context id used is the current autograd context
id of this node when backward() is called. If there is no valid autograd
context id, we throw an error. You can retrieve the accumulated gradients
using the ``get_gradients`` API.

Arguments:
    roots: List of tensors which represent the roots of the autograd
        computation. All the tensors should be scalars.

Example::
    >> import torch.distributed.autograd as dist_autograd
    >> with dist_autograd.context() as context_id:
    >>      pred = model.forward()
    >>      loss = loss_func(pred, loss)
    >>      dist_autograd.backward(loss)
)",
      py::call_guard<py::gil_scoped_release>());

  module.def(
      "get_gradients",
      [](int64_t contextId) {
        const auto& autogradContext =
            DistAutogradContainer::getInstance().retrieveContext(contextId);
        return torch::jit::toPyObject(IValue(autogradContext.getGradients()));
      },
      R"(
Retrieves a map from Tensor to the appropriate gradient for that Tensor
accumulated in the provided context_id as part of the distributed autograd
backward pass.

Arguments:
    context_id: The autograd context id for which we should retrieve the
                gradients.

Example::
    >> import torch.distributed.autograd as dist_autograd
    >> with dist_autograd.context() as context_id:
    >>      t1 = torch.rand((3, 3), requires_grad=True)
    >>      t2 = torch.rand((3, 3), requires_grad=True)
    >>      loss = t1 + t2
    >>      dist_autograd.backward([loss.sum()])
    >>      grads = dist_autograd.get_gradients(context_id)
    >>      print (grads[t1])
    >>      print (grads[t2])
)");

  Py_RETURN_TRUE;
}
} // namespace

static PyMethodDef methods[] = { // NOLINT
    {"_dist_autograd_init",
     (PyCFunction)dist_autograd_init,
     METH_NOARGS,
     nullptr},
    {nullptr, nullptr, 0, nullptr}};

PyMethodDef* python_functions() {
  return methods;
}

} // namespace autograd
} // namespace distributed
} // namespace torch
