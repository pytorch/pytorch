#include "torch/csrc/python_headers.h"

#include <c10d/Def.hpp>
#include <c10d/FileStore.hpp>
#include <c10d/ProcessGroup.hpp>
#include <c10d/ProcessGroupGloo.hpp>

#ifdef USE_C10D_NCCL
#include <c10d/ProcessGroupNCCL.hpp>
#endif

#include <c10d/TCPStore.hpp>
#include <gloo/transport/tcp/device.h>
#include <pybind11/chrono.h>

#include "torch/csrc/Exceptions.h"
#include "torch/csrc/utils/object_ptr.h"
#include "torch/csrc/utils/pybind.h"

namespace torch {
namespace distributed {
namespace c10d {

namespace {

template <typename T>
using shared_ptr_class_ = py::class_<T, std::shared_ptr<T>>;

PyObject* c10d_init(PyObject* _unused) {
  auto c10d_module =
      THPObjectPtr(PyImport_ImportModule("torch.distributed.c10d"));
  if (!c10d_module) {
    throw python_error();
  }

  auto module = py::handle(c10d_module).cast<py::module>();

  py::class_<::c10d::BroadcastOptions>(module, "BroadcastOptions")
      .def(py::init<>())
      .def_readwrite("rootRank", &::c10d::BroadcastOptions::rootRank)
      .def_readwrite("rootTensor", &::c10d::BroadcastOptions::rootTensor);

  py::class_<::c10d::AllreduceOptions>(module, "AllreduceOptions")
      .def(py::init<>())
      .def_readwrite("reduceOp", &::c10d::AllreduceOptions::reduceOp);

  py::enum_<::c10d::ReduceOp>(module, "ReduceOp")
      .value("SUM", ::c10d::ReduceOp::SUM)
      .value("PRODUCT", ::c10d::ReduceOp::PRODUCT)
      .value("MIN", ::c10d::ReduceOp::MIN)
      .value("MAX", ::c10d::ReduceOp::MAX);

  auto store =
      shared_ptr_class_<::c10d::Store>(module, "Store")
          // Convert from std::string to std::vector<uint8>.
          .def(
              "set",
              [](::c10d::Store& store,
                 const std::string& key,
                 const std::string& value) {
                std::vector<uint8_t> value_(value.begin(), value.end());
                store.set(key, value_);
              },
              py::call_guard<py::gil_scoped_release>())
          // Convert from std::vector<uint8_t> to py::bytes.
          // The returned value is not guaranteed to be valid UTF-8.
          .def(
              "get",
              [](::c10d::Store& store, const std::string& key) -> py::bytes {
                auto value = store.get(key);
                return py::bytes(
                    reinterpret_cast<char*>(value.data()), value.size());
              },
              py::call_guard<py::gil_scoped_release>())
          .def(
              "add",
              &::c10d::Store::add,
              py::call_guard<py::gil_scoped_release>())
          .def(
              "wait",
              &::c10d::Store::wait,
              py::call_guard<py::gil_scoped_release>());

  shared_ptr_class_<::c10d::FileStore>(module, "FileStore", store)
      .def(py::init<const std::string&>());

  shared_ptr_class_<::c10d::TCPStore>(module, "TCPStore", store)
      .def(py::init<const std::string&, int, bool>());

  auto processGroup =
      shared_ptr_class_<::c10d::ProcessGroup>(module, "ProcessGroup")
          .def("rank", &::c10d::ProcessGroup::getRank)
          .def("size", &::c10d::ProcessGroup::getSize)
          .def(
              "broadcast",
              &::c10d::ProcessGroup::broadcast,
              py::call_guard<py::gil_scoped_release>())
          .def(
              "broadcast",
              [](::c10d::ProcessGroup& pg, at::Tensor& x, int rootRank) {
                ::c10d::BroadcastOptions opts;
                opts.rootRank = rootRank;
                std::vector<at::Tensor> xs = {x};
                return pg.broadcast(xs, opts);
              },
              py::arg("tensor"),
              py::arg("root"),
              py::call_guard<py::gil_scoped_release>())
          .def(
              "allreduce",
              &::c10d::ProcessGroup::allreduce,
              py::call_guard<py::gil_scoped_release>())

          .def(
              "allreduce",
              [](::c10d::ProcessGroup& pg, at::Tensor& x, ::c10d::ReduceOp op) {
                ::c10d::AllreduceOptions opts;
                opts.reduceOp = op;
                std::vector<at::Tensor> xs = {x};
                return pg.allreduce(xs, opts);
              },
              py::arg("tensor"),
              py::arg("op") = ::c10d::ReduceOp::SUM,
              py::call_guard<py::gil_scoped_release>());

  auto processGroupGloo = shared_ptr_class_<::c10d::ProcessGroupGloo>(
      module, "ProcessGroupGloo", processGroup);

  shared_ptr_class_<::gloo::transport::Device>(processGroupGloo, "Device");

  shared_ptr_class_<::c10d::ProcessGroupGloo::Options>(
      processGroupGloo, "Options")
      .def(py::init<>())
      .def_readwrite("devices", &::c10d::ProcessGroupGloo::Options::devices)
      .def_readwrite("timeout", &::c10d::ProcessGroupGloo::Options::timeout)
      .def_readwrite("threads", &::c10d::ProcessGroupGloo::Options::threads)
      .def_readwrite(
          "cacheNumAlgorithmEntries",
          &::c10d::ProcessGroupGloo::Options::cacheNumAlgorithmEntries);

  processGroupGloo.def_static(
      "create_tcp_device",
      [](const std::string& hostname, const std::string& interface)
          -> std::shared_ptr<::gloo::transport::Device> {
        ::gloo::transport::tcp::attr attr;
        if (!hostname.empty()) {
          attr.hostname = hostname;
        } else if (!interface.empty()) {
          attr.iface = interface;
        } else {
          // Neither argument is specified; Gloo itself will use the hostname
          // Nothing specified, default to something useful
        }
        return ::gloo::transport::tcp::CreateDevice(attr);
      },
      py::arg("hostname") = "",
      py::arg("interface") = "");

  processGroupGloo
      .def(py::init<
           const std::shared_ptr<::c10d::Store>&,
           int,
           int,
           ::c10d::ProcessGroupGloo::Options>())
      .def(py::init(
          [](const std::shared_ptr<::c10d::Store>& store, int rank, int size) {
            ::c10d::ProcessGroupGloo::Options options;

            // By default, use the hostname to resolve the network address to
            // use. Note: if the hostname does not resolve to an address (e.g.
            // because of misconfigured /etc/hosts file), this will not work.
            std::array<char, HOST_NAME_MAX> hostname;
            auto rv = gethostname(hostname.data(), hostname.size());
            if (rv != 0) {
              throw std::system_error(errno, std::system_category());
            }

            ::gloo::transport::tcp::attr attr;
            attr.hostname = hostname.data();
            options.devices.push_back(
                ::gloo::transport::tcp::CreateDevice(attr));
            return std::make_shared<::c10d::ProcessGroupGloo>(
                store, rank, size, options);
          }));

#ifdef USE_C10D_NCCL
  shared_ptr_class_<::c10d::ProcessGroupNCCL>(
      module, "ProcessGroupNCCL", processGroup)
      .def(py::init<const std::shared_ptr<::c10d::Store>&, int, int>());
#endif

  shared_ptr_class_<::c10d::ProcessGroup::Work>(module, "Work")
      .def("isCompleted", &::c10d::ProcessGroup::Work::isCompleted)
      .def("isSuccess", &::c10d::ProcessGroup::Work::isSuccess)
      .def("exception", &::c10d::ProcessGroup::Work::exception)
      .def("synchronize", &::c10d::ProcessGroup::Work::synchronize)
      .def(
          "wait",
          &::c10d::ProcessGroup::Work::wait,
          py::call_guard<py::gil_scoped_release>());

  Py_RETURN_TRUE;
}

} // namespace

// c10d methods on torch._C
static PyMethodDef methods[] = {
    {"_c10d_init", (PyCFunction)c10d_init, METH_NOARGS, nullptr},
    {nullptr, nullptr, 0, nullptr}};

PyMethodDef* python_functions() {
  return methods;
}

} // namespace c10d
} // namespace distributed
} // namespace torch
