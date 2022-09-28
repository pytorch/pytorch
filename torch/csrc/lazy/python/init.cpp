#include <torch/csrc/lazy/python/init.h>

#include <ATen/FunctionalTensorWrapper.h>
#include <c10/core/Device.h>
#include <torch/csrc/jit/python/pybind.h>
#include <torch/csrc/lazy/backend/backend_device.h>
#include <torch/csrc/lazy/backend/backend_interface.h>
#include <torch/csrc/lazy/core/config.h>
#include <torch/csrc/lazy/core/debug_util.h>
#include <torch/csrc/lazy/core/internal_ops/ltc_ops.h>
#include <torch/csrc/lazy/core/ir_dump_util.h>
#include <torch/csrc/lazy/core/lazy_graph_executor.h>
#include <torch/csrc/lazy/core/metrics.h>
#include <torch/csrc/lazy/core/trie.h>
#include <torch/csrc/lazy/python/python_util.h>
#if !(defined(FBCODE_CAFFE2) || defined(OVRSOURCE))
#include <torch/csrc/lazy/ts_backend/ts_backend_impl.h>
#include <torch/csrc/lazy/ts_backend/ts_lowering_context.h>
#endif // FBCODE_CAFFE2 || OVRSOURCE
#include <string>
#include <vector>

namespace torch {
namespace lazy {

// TODO(whc) backend 'device' related APIs are not very clear, this code could
// be simplified but it should probably be done together with
// designing/refactoring the overall approach to get/set of default eager/lazy
// device types
torch::lazy::BackendDevice GetDeviceOrCurrent(const std::string& device_str) {
  if (device_str.empty()) {
    getBackend()->GetDefaultDeviceType();
    return torch::lazy::BackendDevice();
  }
  return torch::lazy::atenDeviceToBackendDevice(c10::Device(device_str));
}

std::ptrdiff_t GetTensorId(const at::Tensor& tensor) {
  torch::lazy::LazyTensorPtr lazy_tensor = torch::lazy::TryGetLtcTensor(tensor);
  return lazy_tensor->GetUniqueId();
}

std::string GetTensorsDump(
    const std::vector<at::Tensor>& tensors,
    const std::function<std::string(c10::ArrayRef<torch::lazy::Node*>)>&
        coverter) {
  std::vector<torch::lazy::Node*> nodes;
  std::vector<torch::lazy::Value> values;
  for (auto& tensor : tensors) {
    auto inner = at::functionalization::impl::from_functional_tensor(tensor);
    torch::lazy::LazyTensorPtr lazy_tensor =
        torch::lazy::TryGetLtcTensor(inner);
    values.push_back(lazy_tensor->GetIrValue());
    nodes.push_back(values.back().node.get());
  }
  return coverter(nodes);
}

std::vector<torch::lazy::LazyTensorPtr> GetLtcTensors(
    const std::vector<at::Tensor>& tensors,
    bool want_all) {
  std::vector<torch::lazy::LazyTensorPtr> lazy_tensors;
  lazy_tensors.reserve(tensors.size());
  if (want_all) {
    for (auto& tensor : tensors) {
      lazy_tensors.push_back(torch::lazy::TryGetLtcTensor(tensor));
    }
  } else {
    for (auto& tensor : tensors) {
      auto lazy_tensor = torch::lazy::TryGetLtcTensor(tensor);
      if (lazy_tensor) {
        lazy_tensors.push_back(lazy_tensor);
      }
    }
  }
  return lazy_tensors;
}

std::string GetTensorsBackendGraph(const std::vector<at::Tensor>& tensors) {
  std::vector<torch::lazy::LazyTensorPtr> lazy_tensors =
      GetLtcTensors(tensors, /*want_all=*/false);
  return torch::lazy::LazyGraphExecutor::Get()->DumpBackendComputation(
      lazy_tensors);
}

void SyncTensors(
    const std::vector<at::Tensor>& tensors,
    const std::vector<std::string>& devices,
    bool wait,
    bool sync_ltc_data) {
  std::vector<torch::lazy::LazyTensorPtr> lazy_tensors =
      GetLtcTensors(tensors, /*want_all=*/false);
  torch::lazy::LazyGraphExecutor::Get()->SyncTensorsGraph(
      &lazy_tensors, devices, wait, sync_ltc_data);
}

void initLazyBindings(PyObject* module) {
  auto m = py::handle(module).cast<py::module>();
  auto lazy = m.def_submodule("_lazy");
  auto lazy_ts_backend = m.def_submodule("_lazy_ts_backend");

  lazy.def(
      "_mark_step",
      // TODO(whc) this API should probably change from vector<string> to
      // vector<c10::device> but in a separate PR
      [](const std::string& device_str,
         const std::vector<std::string>& devices,
         bool wait) {
        pybind11::gil_scoped_release no_gil;
        auto backend_device = GetDeviceOrCurrent(device_str);
        torch::lazy::LazyGraphExecutor::Get()->SyncLiveTensorsGraph(
            &backend_device, devices, wait);
        torch::lazy::LazyGraphExecutor::Get()->MarkStep(backend_device);
      },
      py::arg("device") = "",
      py::arg("devices"),
      py::arg("wait") = true);
  lazy.def(
      "_wait_device_ops",
      [](const std::vector<std::string>& devices) {
        pybind11::gil_scoped_release no_gil;
        // TODO: Add support of non-empty devices.
        if (!devices.empty()) {
          LOG(ERROR) << "Non-empty devices are not supported.";
        }
        torch::lazy::LazyGraphExecutor::Get()->WaitDeviceOps({});
      },
      py::arg("devices"));
  lazy.def(
      "_reset_metrics", []() { torch::lazy::MetricsArena::Get()->Reset(); });
  lazy.def("_counter_names", []() { return torch::lazy::GetCounterNames(); });
  lazy.def(
      "_metrics_report", []() { return torch::lazy::CreateMetricReport(); });
  lazy.def("_counter_value", [](const std::string& name) -> py::object {
    torch::lazy::CounterData* data = torch::lazy::GetCounter(name);
    return data != nullptr ? py::cast<int64_t>(data->Value()) : py::none();
  });
  lazy.def("_get_tensor_id", [](const at::Tensor& tensor) {
    return GetTensorId(tensor);
  });

  lazy.def(
      "_get_tensors_text",
      [](const std::vector<at::Tensor>& tensors) -> std::string {
        auto coverter = [](c10::ArrayRef<torch::lazy::Node*> nodes) {
          return torch::lazy::DumpUtil::ToText(nodes);
        };
        return GetTensorsDump(tensors, coverter);
      });
  lazy.def(
      "_get_tensors_dot",
      [](const std::vector<at::Tensor>& tensors) -> std::string {
        auto coverter = [](c10::ArrayRef<torch::lazy::Node*> nodes) {
          return torch::lazy::DumpUtil::ToDot(nodes);
        };
        return GetTensorsDump(tensors, coverter);
      });
  lazy.def(
      "_get_tensors_backend",
      [](const std::vector<at::Tensor>& tensors) -> std::string {
        return GetTensorsBackendGraph(tensors);
      });
  lazy.def("_get_graph_hash", [](const std::vector<at::Tensor>& tensors) {
    std::vector<LazyTensorPtr> xtensors;
    xtensors.reserve(tensors.size());
    for (auto& tensor : tensors) {
      xtensors.push_back(TryGetLtcTensor(tensor));
    }
    auto hash = LazyGraphExecutor::Get()->GetGraphHash(xtensors);
    std::string bin((const char*)&hash, sizeof(hash));
    return py::bytes(bin);
  });
  lazy.def(
      "_sync_multi",
      [](const std::vector<at::Tensor>& tensors,
         const std::vector<std::string>& devices,
         bool wait,
         bool sync_ltc_data) {
        pybind11::gil_scoped_release no_gil;
        SyncTensors(tensors, devices, wait, sync_ltc_data);
      },
      py::arg("tensors"),
      py::arg("devices"),
      py::arg("wait") = true,
      py::arg("sync_ltc_data") = true);

  lazy.def("_get_force_fallback", []() {
    return torch::lazy::getLTCForceFallback();
  });
  lazy.def("_set_force_fallback", [](std::string newval) {
    torch::lazy::getLTCForceFallback() = newval;
  });
  lazy.def("_clear_ir_cache", []() { TrieCache::Get()->Clear(); });
  lazy.def("_dump_ir_cache", [](std::string filename) {
    TrieCache::Get()->DumpToDotFile(filename);
  });
  lazy.def("_set_reuse_ir", [](bool val) { FLAGS_torch_lazy_reuse_ir = val; });
  lazy.def("_set_symbolic_shape_mode", [](bool val) {
    FLAGS_ltc_enable_symbolic_shapes = val;
  });
  lazy.def("_get_symbolic_shape_mode", []() {
    return FLAGS_ltc_enable_symbolic_shapes;
  });
  lazy.def("_get_default_device_type", []() {
    return getBackend()->GetDefaultDeviceType()->toString();
  });

  lazy_ts_backend.def("_init", []() {
#if !(defined(FBCODE_CAFFE2) || defined(OVRSOURCE))
    torch::lazy::InitTorchScriptBackend();
#else
      TORCH_CHECK(false, "TorchScript backend not yet supported in FBCODE/OVRSOURCE builds");
#endif // !(defined(FBCODE_CAFFE2) || defined(OVRSOURCE))
  });

  /*
   * Return tensor ids and tensors for DeviceData nodes.
   * TODO(shunting) revisit this API for XLA
   */
  lazy_ts_backend.def(
      "_get_tensors_ts_device_data_node",
      [](const std::vector<at::Tensor>& tensors)
          -> std::pair<std::vector<int64_t>, std::vector<at::IValue>> {
#if !(defined(FBCODE_CAFFE2) || defined(OVRSOURCE))
        std::vector<Node*> roots;
        for (auto& tensor : tensors) {
          auto xtensor = TryGetLtcTensor(tensor);
          roots.push_back(xtensor->GetIrValue().node.get());
        }
        auto post_order = Util::ComputePostOrder(roots);
        std::vector<int64_t> tensor_ids;
        std::vector<at::IValue> ivalues;

        std::unordered_set<BackendData::Handle> data_handles_;
        for (auto nodeptr : post_order) {
          if (nodeptr->op() == *torch::lazy::ltc_device_data) {
            const auto backend_data =
                getBackend()->GetComputationDataFromNode(nodeptr);

            auto infoptr = backend_data->info();
            auto deviceDataInfoPtr =
                (torch::lazy::LazyGraphExecutor::DeviceDataInfo*)infoptr;
            auto* tsDataPtr = (torch::lazy::TSData*)backend_data.get();

            // dedup DeviceData by handle
            auto handle = tsDataPtr->GetHandle();
            if (!data_handles_.insert(handle).second) {
              continue;
            }
            tensor_ids.push_back(deviceDataInfoPtr->tensor_id);
            /*
             * If the TSData contains a tensor, then the tensor id will uniquely
             * identify the tensor. We use that tensor id to find the tensor in
             * other places: e.g. in the python forward method parameters.
             *
             * If the TSData contains a scalar, the tensor id itself is not
             * important. We reuse the scalar value in future calls.
             */
            if (tsDataPtr->HasValue()) {
              ivalues.emplace_back(tsDataPtr->data());
            } else {
              CHECK(tsDataPtr->scalar.has_value());
              ivalues.emplace_back(tsDataPtr->scalar.value());
            }
          }
        }
        return std::make_pair(tensor_ids, ivalues);
#else
        TORCH_CHECK(
            false, "TorchScript backend not yet supported in FBCODE builds");
        return std::make_pair(
            std::vector<int64_t>(), std::vector<at::IValue>());
#endif // !(defined(FBCODE_CAFFE2) || defined(OVRSOURCE))
      });
  // TODO(shunting) revisit this part for XLA
  lazy_ts_backend.def(
      "_run_cached_graph",
      [](const std::string& hash_str,
         const std::vector<at::IValue>& graph_inputs) {
        std::vector<at::Tensor> result;
#if !(defined(FBCODE_CAFFE2) || defined(OVRSOURCE))
        TORCH_CHECK(hash_str.size() == sizeof(hash_t));
        hash_t hash = *(hash_t*)(hash_str.c_str());
        auto cachedComputation =
            LazyGraphExecutor::Get()->GetComputationCache()->Get(hash);
        TORCH_CHECK(
            cachedComputation,
            "Failed to get computation by hash. Maybe the entry get kicked out of the LRU cache"); // TODO implement a fallback mechanism, or make sure those entries never get kicked out
        auto computationPtr =
            (torch::lazy::TSComputation*)cachedComputation->computation.get();

        std::vector<torch::jit::IValue> stack;
        stack.reserve(graph_inputs.size());
        for (const auto& arg : graph_inputs) {
          stack.emplace_back(arg);
        }
        computationPtr->graph_executor().run(stack);
        result.reserve(stack.size());
        for (torch::jit::IValue elem : stack) {
          result.push_back(elem.toTensor());
        }
#else
        TORCH_CHECK(
            false, "TorchScript backend not yet supported in FBCODE builds");
#endif // !(defined(FBCODE_CAFFE2) || defined(OVRSOURCE))
        return result;
      });
}

} // namespace lazy
} // namespace torch
