#include <c10/core/Device.h>
#include <c10/util/Optional.h>
#include <torch/csrc/lazy/backend/backend_device.h>
#include <torch/csrc/lazy/backend/backend_interface.h>
#include <torch/csrc/lazy/core/helpers.h>
#include <torch/csrc/lazy/core/ir_dump_util.h>
#include <torch/csrc/lazy/core/ir_util.h>
#include <torch/csrc/lazy/core/lazy_graph_executor.h>
#include <torch/csrc/lazy/core/ir.h>
#include <torch/csrc/lazy/core/tensor.h>
#include <torch/csrc/lazy/core/metrics.h>
#include <torch/csrc/lazy/core/multi_wait.h>
#include <torch/csrc/lazy/core/tensor_impl.h>
#include <torch/csrc/lazy/core/tensor_util.h>
#include <torch/csrc/lazy/core/thread_pool.h>
#include <torch/csrc/lazy/core/util.h>
#include <torch/csrc/lazy/python/python_util.h>
#include <torch/csrc/lazy/core/internal_ops/ltc_ops.h>
#include <torch/csrc/lazy/ts_backend/ops/device_data.h>

#include <cstring>
#include <sstream>
#include <string>
#include <thread>
#include <vector>

#include "lazy_tensor_core/csrc/tensor_aten_ops.h"
#include "lazy_tensor_core/csrc/tensor_distributed.h"
#include "lazy_tensor_core/csrc/ts_backend/backend_impl.h"
#include "lazy_tensor_core/csrc/version.h"
#include "lazy_tensors/computation_client/metrics_analysis.h"
#include "lazy_tensors/computation_client/metrics_reader.h"
#include "lazy_tensors/computation_client/sys_util.h"
#include "torch/csrc/autograd/utils/wrap_outputs.h"
#include "torch/csrc/autograd/variable.h"
#include "torch/csrc/jit/python/pybind.h"
#include "torch/csrc/lazy/core/config.h"
#include "torch/csrc/utils/cuda_lazy_init.h"

namespace torch_lazy_tensors {
namespace {

struct NoGilSection {
  NoGilSection() : state(PyEval_SaveThread()) {}
  ~NoGilSection() { PyEval_RestoreThread(state); }
  PyThreadState* state = nullptr;
};

c10::optional<torch::lazy::BackendDevice> GetOptionalDevice(const std::string& device_str) {
  if (device_str.empty()) {
    return c10::nullopt;
  }
  return torch::lazy::atenDeviceToBackendDevice(c10::Device(device_str));
}

torch::lazy::BackendDevice GetDeviceOrCurrent(const std::string& device_str) {
  if (device_str.empty()) {
    return torch::lazy::BackendDevice();
  }
  return torch::lazy::atenDeviceToBackendDevice(c10::Device(device_str));
}

void PrepareToExit() {
  // TODO(whc) should we hook this interface up? It does nothing currently
  torch::lazy::getBackend()->PrepareToExit();
  // TODO(whc) can I call this unconditionally?
  torch::lazy::LazyGraphExecutor::Get()->WaitDeviceOps({});
}

std::string GetTensorsDump(
    const std::vector<at::Tensor>& tensors,
    const std::function<std::string(c10::ArrayRef<torch::lazy::Node*>)>&
        coverter) {
  std::vector<torch::lazy::Node*> nodes;
  std::vector<torch::lazy::Value> values;
  for (auto& tensor : tensors) {
    torch::lazy::LazyTensorPtr lazy_tensor = torch::lazy::TryGetLtcTensor(tensor);
    values.push_back(lazy_tensor->GetIrValue());
    nodes.push_back(values.back().node.get());
  }
  return coverter(nodes);
}

std::vector<std::string> GetLtcDeviceStrings(
    const std::vector<std::string>& devices) {
  std::vector<std::string> ltc_devices;
  ltc_devices.reserve(devices.size());
  for (auto& device_str : devices) {
    auto device = torch::lazy::atenDeviceToBackendDevice(c10::Device(device_str));
    ltc_devices.emplace_back(device.toString());
  }
  return ltc_devices;
}

std::vector<torch::lazy::BackendDevice> GetLtcDevices(const std::vector<std::string>& devices) {
  std::vector<torch::lazy::BackendDevice> ltc_devices;
  ltc_devices.reserve(devices.size());
  for (auto& device_str : devices) {
    ltc_devices.push_back(
        torch::lazy::atenDeviceToBackendDevice(c10::Device(device_str)));
  }
  return ltc_devices;
}

std::vector<torch::lazy::LazyTensorPtr> GetLtcTensors(const std::vector<at::Tensor>& tensors,
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

AllReduceType GetReduceType(const std::string& reduce_type) {
  if (reduce_type == "sum") {
    return AllReduceType::kSum;
  } else if (reduce_type == "mul") {
    return AllReduceType::kMul;
  } else if (reduce_type == "and") {
    return AllReduceType::kAnd;
  } else if (reduce_type == "or") {
    return AllReduceType::kOr;
  } else if (reduce_type == "min") {
    return AllReduceType::kMin;
  } else if (reduce_type == "max") {
    return AllReduceType::kMax;
  }
  LOG(ERROR) << "Unknown AllReduce type: " << reduce_type;
}

std::vector<std::vector<int64_t>> CreateReduceGroups(const py::list& groups) {
  std::vector<std::vector<int64_t>> replica_groups;
  for (auto& group : groups) {
    replica_groups.emplace_back();
    for (auto& replica_id : group.cast<py::list>()) {
      replica_groups.back().push_back(replica_id.cast<int64_t>());
    }
  }
  return replica_groups;
}

std::vector<std::pair<int64_t, int64_t>> CreateSourceTargetPairs(
    const py::list& pairs) {
  std::vector<std::pair<int64_t, int64_t>> source_target_pairs;
  for (auto& pair : pairs) {
    const auto& pylist_pair = pair.cast<py::list>();
    CHECK_EQ(len(pylist_pair), 2);
    source_target_pairs.push_back(
        {pylist_pair[0].cast<int64_t>(), pylist_pair[1].cast<int64_t>()});
  }
  return source_target_pairs;
}

std::shared_ptr<torch::lazy::Value> AllReduceInPlace(
    const std::string& reduce_type, const std::vector<at::Tensor>& tensors,
    const std::shared_ptr<torch::lazy::Value>& token, double scale,
    const std::vector<std::vector<int64_t>>& replica_groups) {
  std::vector<torch::lazy::LazyTensorPtr> lazy_tensors = GetLtcTensors(tensors, /*want_all=*/true);
  return std::make_shared<torch::lazy::Value>(
      lazy_tensor_distributed::all_reduce(&lazy_tensors, *token,
                                          GetReduceType(reduce_type), scale,
                                          replica_groups));
}

std::pair<at::Tensor, std::shared_ptr<torch::lazy::Value>> AllReduce(
    const std::string& reduce_type, const at::Tensor& input,
    const std::shared_ptr<torch::lazy::Value>& token, double scale,
    const std::vector<std::vector<int64_t>>& replica_groups) {
  torch::lazy::LazyTensorPtr result;
  torch::lazy::Value new_token;
  std::tie(result, new_token) = lazy_tensor_distributed::all_reduce(
      torch::lazy::TryGetLtcTensor(input), *token, GetReduceType(reduce_type), scale,
      replica_groups);
  return std::pair<at::Tensor, std::shared_ptr<torch::lazy::Value>>(
      torch::lazy::CreateAtenFromLtcTensor(std::move(result)),
      std::make_shared<torch::lazy::Value>(new_token));
}

std::pair<at::Tensor, std::shared_ptr<torch::lazy::Value>> AllToAll(
    const at::Tensor& input, const std::shared_ptr<torch::lazy::Value>& token,
    int64_t split_dimension, int64_t concat_dimension, int64_t split_count,
    const std::vector<std::vector<int64_t>>& replica_groups) {
  torch::lazy::LazyTensorPtr result;
  torch::lazy::Value new_token;
  std::tie(result, new_token) = lazy_tensor_distributed::all_to_all(
      torch::lazy::TryGetLtcTensor(input), *token, split_dimension, concat_dimension,
      split_count, replica_groups);
  return std::pair<at::Tensor, std::shared_ptr<torch::lazy::Value>>(
      torch::lazy::CreateAtenFromLtcTensor(std::move(result)),
      std::make_shared<torch::lazy::Value>(new_token));
}

std::pair<at::Tensor, std::shared_ptr<torch::lazy::Value>> CollectivePermute(
    const at::Tensor& input, const std::shared_ptr<torch::lazy::Value>& token,
    const std::vector<std::pair<int64_t, int64_t>>& source_target_pairs) {
  torch::lazy::LazyTensorPtr result;
  torch::lazy::Value new_token;
  std::tie(result, new_token) = lazy_tensor_distributed::collective_permute(
      torch::lazy::TryGetLtcTensor(input), *token, source_target_pairs);
  return std::pair<at::Tensor, std::shared_ptr<torch::lazy::Value>>(
      torch::lazy::CreateAtenFromLtcTensor(std::move(result)),
      std::make_shared<torch::lazy::Value>(new_token));
}

void SyncTensors(const std::vector<at::Tensor>& tensors,
                 const std::vector<std::string>& devices, bool wait,
                 bool sync_ltc_data) {
  std::vector<torch::lazy::LazyTensorPtr> lazy_tensors = GetLtcTensors(tensors, /*want_all=*/false);
  torch::lazy::LazyGraphExecutor::Get()->SyncTensorsGraph(&lazy_tensors, devices, wait,
                                             sync_ltc_data);
}

void SyncLiveTensors(const std::string& device_str,
                     const std::vector<std::string>& devices, bool wait) {
  auto opt_device = GetOptionalDevice(device_str);
  torch::lazy::LazyGraphExecutor::Get()->SyncLiveTensorsGraph(
      opt_device ? &opt_device.value() : nullptr, devices, wait);
}

void StepMarker(const std::string& device_str,
                const std::vector<std::string>& devices, bool wait) {
  auto device = GetDeviceOrCurrent(device_str);
  torch::lazy::LazyGraphExecutor::Get()->SyncLiveTensorsGraph(&device, devices, wait);
  torch::lazy::LazyGraphExecutor::Get()->MarkStep(device);
  bool debug_mode = lazy_tensors::sys_util::GetEnvBool("PT_LTC_DEBUG", false);
  if (C10_UNLIKELY(debug_mode)) {
    std::string report = lazy_tensors::metrics::CreatePerformanceReport();
    if (!report.empty()) {
      std::string fout =
          lazy_tensors::sys_util::GetEnvString("PT_LTC_DEBUG_FILE", "");
      if (C10_UNLIKELY(!fout.empty())) {
        std::ofstream out_file(fout, std::ios_base::app);
        out_file << report;
      } else {
        std::cout << report;
      }
    }
  }
}

void SetRngSeed(uint64_t seed, const std::string& device_str) {
  auto device = GetDeviceOrCurrent(device_str);
  torch::lazy::LazyGraphExecutor::Get()->SetRngSeed(device, seed);
}

uint64_t GetRngSeed(const std::string& device_str) {
  return torch::lazy::LazyGraphExecutor::Get()->GetRunningSeed(
      GetDeviceOrCurrent(device_str));
}

std::string GetTensorsBackendGraph(const std::vector<at::Tensor>& tensors) {
  std::vector<torch::lazy::LazyTensorPtr> lazy_tensors = GetLtcTensors(tensors, /*want_all=*/false);
  return torch::lazy::LazyGraphExecutor::Get()->DumpBackendComputation(lazy_tensors);
}

std::string GetLiveTensorsReport(size_t nodes_threshold,
                                 const std::string& device_str) {
  auto opt_device = GetOptionalDevice(device_str);
  auto tensors = torch::lazy::LazyGraphExecutor::Get()->GetLiveTensors(
      opt_device ? &opt_device.value() : nullptr);
  std::stringstream ss;
  for (auto& tensor : tensors) {
    torch::lazy::Value ir_value = tensor->CurrentIrValue();
    if (ir_value) {
      std::vector<torch::lazy::Node*> roots({ir_value.node.get()});
      auto post_order = torch::lazy::Util::ComputePostOrder(roots);
      if (post_order.size() > nodes_threshold) {
        ss << "Tensor: id=" << tensor->GetUniqueId()
           << ", shape=" << tensor->shape().Get()
           << ", device=" << tensor->GetDevice()
           << ", ir_nodes=" << post_order.size() << "\n";
        for (size_t i = post_order.size(); i > 0; --i) {
          if (!post_order[i - 1]->metadata().frame_info.empty()) {
            ss << post_order[i - 1]->metadata().frame_info;
            break;
          }
        }
        ss << torch::lazy::DumpUtil::PostOrderToText(post_order, roots);
        ss << "\n\n";
      }
    }
  }
  return ss.str();
}

std::ptrdiff_t GetTensorViewAliasId(const at::Tensor& tensor) {
  torch::lazy::LazyTensorPtr lazy_tensor = torch::lazy::TryGetLtcTensor(tensor);
  return lazy_tensor->GetViewAliasId();
}

std::ptrdiff_t GetTensorId(const at::Tensor& tensor) {
  torch::lazy::LazyTensorPtr lazy_tensor = torch::lazy::TryGetLtcTensor(tensor);
  return lazy_tensor->GetUniqueId();
}

std::vector<at::Tensor> GetLtcTensorsFromAten(
    const std::vector<at::Tensor>& aten_tensors,
    const std::vector<std::string>& devices) {
  auto data_handles =
      torch::lazy::CreateTensorsData(aten_tensors, GetLtcDevices(devices));

  std::vector<at::Tensor> lazy_tensors;
  lazy_tensors.reserve(data_handles.size());
  for (auto& data_handle : data_handles) {
    torch::lazy::LazyTensorPtr lazy_tensor = torch::lazy::LazyTensor::Create(std::move(data_handle));
    lazy_tensors.push_back(torch::lazy::CreateAtenFromLtcTensor(std::move(lazy_tensor)));
  }
  return lazy_tensors;
}

std::shared_ptr<torch::lazy::Value> CreateToken(const std::string& device_str) {
  // This should be using lazy_tensors::CreateToken() once we have added Token
  // support to the backend AllReduce(). Meanwhile we use a constant as token,
  // and we handle it accordingly in cross_replica_reduces.cpp. This needs to be
  // device data (hence coming in as computation parameter) as otherwise the
  // backend compiler passes might remove it, vanishing its sequencing effects.
  auto device = GetDeviceOrCurrent(device_str);
  torch::lazy::Value ir_value = torch::lazy::LazyGraphExecutor::Get()->GetDeviceDataIrValue(
      0.0, c10::ScalarType::Float, device);
  return std::make_shared<torch::lazy::Value>(std::move(ir_value));
}

py::object GetMetricData(const std::string& name) {
  torch::lazy::MetricData* data = torch::lazy::GetMetric(name);
  if (data == nullptr) {
    return py::none();
  }

  double accumulator = 0.0;
  size_t total_samples = 0;
  auto samples = data->Samples(&accumulator, &total_samples);
  auto py_samples = py::tuple(samples.size());
  for (size_t i = 0; i < samples.size(); ++i) {
    auto sample = py::tuple(2);
    sample[0] = 1.0e-9 * samples[i].timestamp_ns;
    sample[1] = samples[i].value;

    py_samples[i] = sample;
  }
  auto result = py::tuple(3);
  result[0] = total_samples;
  result[1] = accumulator;
  result[2] = py_samples;
  return result;
}

py::object GetRevisions() {
  auto py_dict = py::dict();
  py_dict["ltc"] = std::string(LTC_GITREV);
  py_dict["torch"] = std::string(TORCH_GITREV);
  return py_dict;
}

std::vector<at::Tensor> LtcCreateTensorList(const at::TensorList& tensors) {
  std::vector<at::Tensor> aten_ltc_tensors(tensors.size());
  std::vector<torch::lazy::LazyTensorPtr> ltc_tensors;
  // We need to separate out the defined tensors first, torch::lazy::GetLtcTensor() doesn't
  // work with undefined tensors.
  std::vector<bool> to_translate(tensors.size());
  for (size_t i = 0; i < tensors.size(); ++i) {
    const at::Tensor& tensor = tensors[i];
    if (tensor.defined()) {
      auto lazy_tensor = torch::lazy::TryGetLtcTensor(tensor);
      if (lazy_tensor) {
        to_translate[i] = true;
        ltc_tensors.push_back(lazy_tensor);
      } else {
        aten_ltc_tensors[i] = tensor;
      }
    }
  }
  auto defined_aten_ltc_tensors =
      torch::lazy::LazyGraphExecutor::Get()->GetTensors(&ltc_tensors);
  // Insert undefined tensors into the result, back into the original undefined
  // positions.
  for (size_t i = 0, defined_pos = 0; i < tensors.size(); ++i) {
    if (to_translate[i]) {
      aten_ltc_tensors[i] = std::move(defined_aten_ltc_tensors[defined_pos++]);
    }
  }
  return aten_ltc_tensors;
}

// py::dict GetMemoryInfo(const std::string& device_str) {
//   lazy_tensors::ComputationClient::MemoryInfo mem_info;
//   {
//     NoGilSection nogil;
//     auto device = GetDeviceOrCurrent(device_str);
//     mem_info = torch::lazy::getBackend()->GetMemoryInfo(
//         device.toString());
//   }
//   auto py_dict = py::dict();
//   py_dict["kb_free"] = mem_info.kb_free;
//   py_dict["kb_total"] = mem_info.kb_total;
//   return py_dict;
// }

using namespace torch::lazy;

void InitLtcModuleBindings(py::module m) {
  m.def("_prepare_to_exit", []() { PrepareToExit(); });
  m.def("_get_git_revs", []() { return GetRevisions(); });
  m.def("_get_ltc_tensors_dot",
        [](const std::vector<at::Tensor>& tensors) -> std::string {
          auto coverter = [](c10::ArrayRef<torch::lazy::Node*> nodes) {
            return torch::lazy::DumpUtil::ToDot(nodes);
          };
          return GetTensorsDump(tensors, coverter);
        });
  m.def("_get_ltc_tensors_text",
        [](const std::vector<at::Tensor>& tensors) -> std::string {
          auto coverter = [](c10::ArrayRef<torch::lazy::Node*> nodes) {
            return torch::lazy::DumpUtil::ToText(nodes);
          };
          return GetTensorsDump(tensors, coverter);
        });
  m.def("_get_ltc_tensors_backend",
        [](const std::vector<at::Tensor>& tensors) -> std::string {
          return GetTensorsBackendGraph(tensors);
        });
  m.def("_ltc_tensors_from_aten", [](const std::vector<at::Tensor>& tensors,
                                     const std::vector<std::string>& devices) {
    std::vector<at::Tensor> result;
    {
      NoGilSection nogil;
      std::vector<at::Tensor> lazy_tensors =
          GetLtcTensorsFromAten(tensors, devices);
      result.reserve(lazy_tensors.size());
      for (size_t i = 0; i < lazy_tensors.size(); ++i) {
        result.push_back(torch::autograd::make_variable(
            lazy_tensors[i], /*requires_grad=*/tensors.at(i).requires_grad()));
      }
    }
    return result;
  });
  m.def("_ltc_get_cpu_tensors", [](const std::vector<at::Tensor>& tensors) {
    std::vector<at::Tensor> result;
    {
      NoGilSection nogil;
      std::vector<at::Tensor> cpu_tensors = LtcCreateTensorList(tensors);
      result.reserve(cpu_tensors.size());
      for (size_t i = 0; i < cpu_tensors.size(); ++i) {
        result.push_back(torch::autograd::make_variable(
            cpu_tensors[i], /*requires_grad=*/tensors.at(i).requires_grad()));
      }
    }
    return result;
  });
  m.def("_ltc_get_tensor_view_alias_id",
        [](const at::Tensor& tensor) { return GetTensorViewAliasId(tensor); });
  m.def("_ltc_get_tensor_id",
        [](const at::Tensor& tensor) { return GetTensorId(tensor); });
  m.def("_ltc_get_devices",
        []() { return torch::lazy::getBackend()->GetBackendDevices(); });
  m.def("_ltc_get_all_devices",
        []() { return torch::lazy::getBackend()->GetBackendDevices(); });
  m.def("_ltc_real_devices", [](const std::vector<std::string>& devices) {
    std::vector<std::string> ltc_devices;
    {
      NoGilSection nogil;
      ltc_devices = GetLtcDeviceStrings(devices);
    }
    return ltc_devices;
  });
  m.def(
      "_ltc_set_replication_devices",
      [](const std::vector<std::string>& devices) {
        throw std::runtime_error("TODO(whc) design/implement distributed APIs");
        //   auto replication_devices =
        //       std::make_shared<std::vector<std::string>>(devices);
        //   torch::lazy::getBackend()->SetReplicationDevices(
        //       std::move(replication_devices));
      });
  m.def("_ltc_get_replication_devices", []() {
    throw std::runtime_error("TODO(whc) design/implement distributed APIs");
    // auto replication_devices =
    //     torch::lazy::getBackend()->GetReplicationDevices();
    // return replication_devices != nullptr ? *replication_devices
    //                                       : std::vector<std::string>();
  });
  m.def("_ltc_get_replication_devices_count", []() {
    throw std::runtime_error("TODO(whc) design/implement distributed APIs");
    // auto replication_devices =
    //     torch::lazy::getBackend()->GetReplicationDevices();
    // return replication_devices != nullptr ? replication_devices->size() : 0;
  });

  py::class_<torch::lazy::Value, std::shared_ptr<torch::lazy::Value>>(
      m, "IrValue");
  m.def("_ltc_create_token",
        [](const std::string& device) { return CreateToken(device); });
  m.def(
      "_ltc_all_reduce_inplace",
      [](const std::string& reduce_type, const std::vector<at::Tensor>& tensors,
         const std::shared_ptr<torch::lazy::Value>& token, double scale,
         const py::list& groups) {
        std::vector<std::vector<int64_t>> replica_groups =
            CreateReduceGroups(groups);
        std::shared_ptr<torch::lazy::Value> new_token;
        {
          NoGilSection nogil;
          new_token = AllReduceInPlace(reduce_type, tensors, token, scale,
                                       replica_groups);
        }
        return new_token;
      });
  m.def("_ltc_all_reduce",
        [](const std::string& reduce_type, const at::Tensor& input,
           const std::shared_ptr<torch::lazy::Value>& token, double scale,
           const py::list& groups) {
          std::vector<std::vector<int64_t>> replica_groups =
              CreateReduceGroups(groups);
          at::Tensor result;
          std::shared_ptr<torch::lazy::Value> new_token;
          {
            NoGilSection nogil;
            std::tie(result, new_token) =
                AllReduce(reduce_type, input, token, scale, replica_groups);
          }
          auto result_tuple = py::tuple(2);
          result_tuple[0] = torch::autograd::make_variable(
              result, /*requires_grad=*/input.requires_grad());
          result_tuple[1] = new_token;
          return result_tuple;
        });
  m.def("_ltc_all_to_all", [](const at::Tensor& input,
                              const std::shared_ptr<torch::lazy::Value>& token,
                              int64_t split_dimension, int64_t concat_dimension,
                              int64_t split_count, const py::list& groups) {
    std::vector<std::vector<int64_t>> replica_groups =
        CreateReduceGroups(groups);
    at::Tensor result;
    std::shared_ptr<torch::lazy::Value> new_token;
    {
      NoGilSection nogil;
      std::tie(result, new_token) =
          AllToAll(input, token, split_dimension, concat_dimension, split_count,
                   replica_groups);
    }
    auto result_tuple = py::tuple(2);
    result_tuple[0] = torch::autograd::make_variable(
        result, /*requires_grad=*/input.requires_grad());
    result_tuple[1] = new_token;
    return result_tuple;
  });
  m.def("_ltc_collective_permute",
        [](const at::Tensor& input,
           const std::shared_ptr<torch::lazy::Value>& token,
           const py::list& pairs) {
          std::vector<std::pair<int64_t, int64_t>> source_target_pairs =
              CreateSourceTargetPairs(pairs);
          at::Tensor result;
          std::shared_ptr<torch::lazy::Value> new_token;
          {
            NoGilSection nogil;
            std::tie(result, new_token) =
                CollectivePermute(input, token, source_target_pairs);
          }
          auto result_tuple = py::tuple(2);
          result_tuple[0] = torch::autograd::make_variable(
              result, /*requires_grad=*/input.requires_grad());
          result_tuple[1] = new_token;
          return result_tuple;
        });
  m.def("_ltc_set_default_device", [](const std::string& device) {
    // TODO: Replace this API with _ltc_set_default_device_type.
    // The reasons why deprecating this API are that:
    // i) It makes sense to set default device type to CPU, GPU or TPU,
    // but not lazy given that would just use whatever default hardware type.
    // ii) Setting ordinal like lazy:0 doesn't make any sense now as distributed
    // training/multi-device support is still under development.
    LOG(ERROR) << "Setting the default device is deprecated. Use "
                  "_ltc_set_default_device_type to set the default "
                  "device type instead.";
    return;
  });
  m.def("_ltc_get_default_device", []() {
    // TODO: Call the backend API to get the default ordinal as well. For xla, the
    // default is lazy:1.
    // It's always lazy:X given distributed training/multi-device is not supported yet.
    return "lazy:0";
  });
  m.def(
      "_ltc_set_rng_seed",
      [](uint64_t seed, const std::string& device) {
        SetRngSeed(seed, device);
      },
      py::arg("seed") = 101, py::arg("device") = "");
  m.def(
      "_ltc_get_rng_seed",
      [](const std::string& device) { return GetRngSeed(device); },
      py::arg("device") = "");
  m.def(
      "_ltc_sync_multi",
      [](const std::vector<at::Tensor>& tensors,
         const std::vector<std::string>& devices, bool wait,
         bool sync_ltc_data) {
        NoGilSection nogil;
        SyncTensors(tensors, devices, wait, sync_ltc_data);
      },
      py::arg("tensors"), py::arg("devices"), py::arg("wait") = true,
      py::arg("sync_ltc_data") = true);
  m.def(
      "_ltc_sync_live_tensors",
      [](const std::string& device, const std::vector<std::string>& devices,
         bool wait) {
        NoGilSection nogil;
        SyncLiveTensors(device, devices, wait);
      },
      py::arg("device") = "", py::arg("devices"), py::arg("wait") = true);
  m.def(
      "_ltc_step_marker",
      [](const std::string& device, const std::vector<std::string>& devices,
         bool wait) {
        NoGilSection nogil;
        StepMarker(device, devices, wait);
      },
      py::arg("device") = "", py::arg("devices"), py::arg("wait") = true);
  m.def(
      "_ltc_wait_device_ops",
      [](const std::vector<std::string>& devices) {
        NoGilSection nogil;
        // TODO: Add support of non-empty devices.
        if (!devices.empty()) {
          LOG(ERROR) << "Non-empty devices are not supported.";
        }
        torch::lazy::LazyGraphExecutor::Get()->WaitDeviceOps({});
      },
      py::arg("devices"));
  m.def("_ltc_reset_metrics",
        []() { torch::lazy::MetricsArena::Get()->Reset(); });
  m.def("_ltc_counter_names", []() { return torch::lazy::GetCounterNames(); });
  m.def("_ltc_counter_value", [](const std::string& name) -> py::object {
    torch::lazy::CounterData* data = torch::lazy::GetCounter(name);
    return data != nullptr ? py::cast<int64_t>(data->Value()) : py::none();
  });
  m.def("_ltc_metric_names", []() { return torch::lazy::GetMetricNames(); });
  m.def("_ltc_metric_data", [](const std::string& name) -> py::object {
    return GetMetricData(name);
  });
  m.def("_ltc_metrics_report",
        []() { return lazy_tensors::metrics_reader::CreateMetricReport(); });
  m.def(
      "_ltc_tensors_report",
      [](size_t nodes_threshold, const std::string& device) {
        return GetLiveTensorsReport(nodes_threshold, device);
      },
      py::arg("nodes_threshold") = 100, py::arg("device") = "");
  // m.def("_ltc_memory_info", [](const std::string& device) -> py::object {
  //   return GetMemoryInfo(device);
  // });
  m.def("_ltc_init_ts_backend", []() { compiler::InitTorchScriptBackend(); });
  m.def("_ltc_set_noop_execution_mode", [](bool enable_noop) {
    torch::lazy::LazyGraphExecutor::Get()->SetNoOpExecutionMode(enable_noop);
  });
  m.def("_ltc_enable_thread_pool", []() {
    FLAGS_torch_lazy_use_thread_pool = true;
  });
  /*
   * Return tensor ids and tensors for DeviceData nodes.
   * TODO(shunting) revisit this API for XLA
   */
  m.def("_get_ltc_tensors_ts_device_data_node",
        [](const std::vector<at::Tensor>& tensors) {
          std::vector<Node*> roots;
          for (auto& tensor : tensors) {
            auto xtensor = TryGetLtcTensor(tensor);
            roots.push_back(xtensor->GetIrValue().node.get());
          }
          auto post_order = Util::ComputePostOrder(roots);
          std::vector<int64_t> tensor_ids;
          std::vector<at::IValue> ivalues;
          for (auto nodeptr : post_order) {
            if (nodeptr->op() == *torch::lazy::ltc_device_data) {
              const auto* device_data_node = torch::lazy::NodeCast<torch::lazy::DeviceData>(nodeptr, *torch::lazy::ltc_device_data);
              auto infoptr = device_data_node->data()->info();
              auto deviceDataInfoPtr = (torch::lazy::LazyGraphExecutor::DeviceDataInfo*) infoptr;
              tensor_ids.push_back(deviceDataInfoPtr->tensor_id);

              auto* tsDataPtr = (torch_lazy_tensors::compiler::TSData*) device_data_node->data().get();
              /*
               * If the TSData contains a tensor, then the tensor id will uniquely identify the tensor.
               * We use that tensor id to find the tensor in other places: e.g. in the python forward method parameters.
               *
               * If the TSData contains a scalar, the tensor id itself is not important. We reuse the scalar value in
               * future calls.
               */
              if (tsDataPtr->HasValue()) {
                ivalues.push_back(tsDataPtr->data());
              } else {
                CHECK(tsDataPtr->scalar.has_value());
                ivalues.push_back(tsDataPtr->scalar.value());
              }
            }
          }
          return std::make_pair(tensor_ids, ivalues);
        });
  m.def("_get_graph_hash", [](const std::vector<at::Tensor>& tensors) {
    std::vector<LazyTensorPtr> xtensors;
    for (auto& tensor : tensors) {
      xtensors.push_back(TryGetLtcTensor(tensor));
    }
    auto hash = LazyGraphExecutor::Get()->GetGraphHash(xtensors);
    std::string bin((const char*) &hash, sizeof(hash));
    return py::bytes(bin);
  });
  // TODO(shunting) revisit this part for XLA
  m.def("_run_cached_graph", [](const std::string& hash_str, const std::vector<at::IValue>& graph_inputs) {
    TORCH_CHECK(hash_str.size() == sizeof(hash_t));
    hash_t hash = *(hash_t*) (hash_str.c_str());
    auto cachedComputation = LazyGraphExecutor::Get()->GetComputationCache()->Get(hash);
    TORCH_CHECK(cachedComputation, "Failed to get computation by hash. Maybe the entry get kicked out of the LRU cache"); // TODO implement a fallback mechanism, or make sure those entries never get kicked out
    auto computationPtr = (torch::lazy::TSComputation*) cachedComputation->computation.get();

    std::vector<torch::jit::IValue> stack;
    for (auto arg : graph_inputs) {
      stack.emplace_back(arg);
    }
    computationPtr->graph_executor().run(stack);
    std::vector<at::Tensor> result;
    for (torch::jit::IValue elem : stack) {
      result.push_back(elem.toTensor());
    }
    return result;
  });
}  // namespace

}  // namespace

void InitLtcBindings(py::module m) { InitLtcModuleBindings(m); }

}  // namespace torch_lazy_tensors

PYBIND11_MODULE(_LAZYC, m) {
  try {
    torch::utils::cuda_lazy_init();
  } catch (const python_error&) {
    // Do nothing, CUDA not available.
  }
  torch_lazy_tensors::InitLtcBindings(m);
}
