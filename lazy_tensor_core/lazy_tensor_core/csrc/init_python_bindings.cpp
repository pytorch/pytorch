#include <c10/core/Device.h>
#include <c10/util/Optional.h>

#include <cstring>
#include <sstream>
#include <string>
#include <thread>
#include <vector>

#include "lazy_tensor_core/csrc/aten_ltc_bridge.h"
#include "lazy_tensor_core/csrc/compiler/backend_impl_interface.h"
#include "lazy_tensor_core/csrc/device.h"
#include "lazy_tensor_core/csrc/helpers.h"
#include "lazy_tensor_core/csrc/ir_dump_util.h"
#include "lazy_tensor_core/csrc/ir_util.h"
#include "lazy_tensor_core/csrc/python_util.h"
#include "lazy_tensor_core/csrc/tensor_impl.h"
#include "lazy_tensor_core/csrc/tensor_util.h"
#include "lazy_tensor_core/csrc/torch_util.h"
#include "lazy_tensor_core/csrc/ts_backend/backend_impl.h"
#include "lazy_tensor_core/csrc/version.h"
#include "lazy_tensors/computation_client/computation_client.h"
#include "lazy_tensors/computation_client/ltc_util.h"
#include "lazy_tensors/computation_client/metrics.h"
#include "lazy_tensors/computation_client/metrics_analysis.h"
#include "lazy_tensors/computation_client/metrics_reader.h"
#include "lazy_tensors/computation_client/multi_wait.h"
#include "lazy_tensors/computation_client/sys_util.h"
#include "lazy_tensors/computation_client/thread_pool.h"
#include "lazy_tensors/computation_client/util.h"
#include "torch/csrc/autograd/utils/wrap_outputs.h"
#include "torch/csrc/autograd/variable.h"
#include "torch/csrc/jit/python/pybind.h"
#include "torch/csrc/utils/cuda_lazy_init.h"

namespace torch_lazy_tensors {
namespace {

struct NoGilSection {
  NoGilSection() : state(PyEval_SaveThread()) {}
  ~NoGilSection() { PyEval_RestoreThread(state); }
  PyThreadState* state = nullptr;
};

c10::optional<Device> GetOptionalDevice(const std::string& device_str) {
  if (device_str.empty()) {
    return c10::nullopt;
  }
  return bridge::AtenDeviceToLtcDevice(c10::Device(device_str));
}

Device GetDeviceOrCurrent(const std::string& device_str) {
  if (device_str.empty()) {
    return GetCurrentDevice();
  }
  return bridge::AtenDeviceToLtcDevice(c10::Device(device_str));
}

void PrepareToExit() {
  lazy_tensors::ComputationClient* client =
      lazy_tensors::ComputationClient::GetIfInitialized();
  if (client != nullptr) {
    LazyTensor::WaitDeviceOps({});
    client->PrepareToExit();
  }
}

std::string GetTensorsDump(
    const std::vector<at::Tensor>& tensors,
    const std::function<std::string(lazy_tensors::Span<const ir::Node* const>)>&
        coverter) {
  std::vector<const ir::Node*> nodes;
  std::vector<ir::Value> values;
  for (auto& tensor : tensors) {
    LazyTensor xtensor = bridge::GetLtcTensor(tensor);
    values.push_back(xtensor.GetIrValue());
    nodes.push_back(values.back().node.get());
  }
  return coverter(nodes);
}

std::string SetCurrentThreadDevice(const std::string& device_str) {
  c10::Device prev_device = bridge::SetCurrentDevice(c10::Device(device_str));
  std::stringstream ss;
  ss << prev_device;
  return ss.str();
}

std::string GetCurrentThreadDevice() {
  return bridge::GetCurrentAtenDevice().str();
}

std::vector<std::string> GetLtcDevices(
    const std::vector<std::string>& devices) {
  std::vector<std::string> ltc_devices;
  ltc_devices.reserve(devices.size());
  for (auto& device_str : devices) {
    Device device = bridge::AtenDeviceToLtcDevice(c10::Device(device_str));
    ltc_devices.emplace_back(device.ToString());
  }
  return ltc_devices;
}

std::vector<LazyTensor> GetLtcTensors(const std::vector<at::Tensor>& tensors,
                                      bool want_all) {
  std::vector<LazyTensor> xtensors;
  xtensors.reserve(tensors.size());
  if (want_all) {
    for (auto& tensor : tensors) {
      xtensors.push_back(bridge::GetLtcTensor(tensor));
    }
  } else {
    for (auto& tensor : tensors) {
      auto xtensor = bridge::TryGetLtcTensor(tensor);
      if (xtensor) {
        xtensors.push_back(*xtensor);
      }
    }
  }
  return xtensors;
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
  LTC_ERROR() << "Unknown AllReduce type: " << reduce_type;
}

std::vector<std::vector<lazy_tensors::int64>> CreateReduceGroups(
    const py::list& groups) {
  std::vector<std::vector<lazy_tensors::int64>> replica_groups;
  for (auto& group : groups) {
    replica_groups.emplace_back();
    for (auto& replica_id : group.cast<py::list>()) {
      replica_groups.back().push_back(replica_id.cast<lazy_tensors::int64>());
    }
  }
  return replica_groups;
}

std::vector<std::pair<lazy_tensors::int64, lazy_tensors::int64>>
CreateSourceTargetPairs(const py::list& pairs) {
  std::vector<std::pair<lazy_tensors::int64, lazy_tensors::int64>>
      source_target_pairs;
  for (auto& pair : pairs) {
    const auto& pylist_pair = pair.cast<py::list>();
    LTC_CHECK_EQ(len(pylist_pair), 2);
    source_target_pairs.push_back({pylist_pair[0].cast<lazy_tensors::int64>(),
                                   pylist_pair[1].cast<lazy_tensors::int64>()});
  }
  return source_target_pairs;
}

std::shared_ptr<ir::Value> AllReduceInPlace(
    const std::string& reduce_type, const std::vector<at::Tensor>& tensors,
    const std::shared_ptr<ir::Value>& token, double scale,
    const std::vector<std::vector<lazy_tensors::int64>>& replica_groups) {
  std::vector<LazyTensor> xtensors = GetLtcTensors(tensors, /*want_all=*/true);
  return std::make_shared<ir::Value>(LazyTensor::all_reduce(
      &xtensors, *token, GetReduceType(reduce_type), scale, replica_groups));
}

std::pair<at::Tensor, std::shared_ptr<ir::Value>> AllReduce(
    const std::string& reduce_type, const at::Tensor& input,
    const std::shared_ptr<ir::Value>& token, double scale,
    const std::vector<std::vector<lazy_tensors::int64>>& replica_groups) {
  LazyTensor result;
  ir::Value new_token;
  std::tie(result, new_token) =
      LazyTensor::all_reduce(bridge::GetLtcTensor(input), *token,
                             GetReduceType(reduce_type), scale, replica_groups);
  return std::pair<at::Tensor, std::shared_ptr<ir::Value>>(
      bridge::AtenFromLtcTensor(std::move(result)),
      std::make_shared<ir::Value>(new_token));
}

std::pair<at::Tensor, std::shared_ptr<ir::Value>> AllToAll(
    const at::Tensor& input, const std::shared_ptr<ir::Value>& token,
    lazy_tensors::int64 split_dimension, lazy_tensors::int64 concat_dimension,
    lazy_tensors::int64 split_count,
    const std::vector<std::vector<lazy_tensors::int64>>& replica_groups) {
  LazyTensor result;
  ir::Value new_token;
  std::tie(result, new_token) = LazyTensor::all_to_all(
      bridge::GetLtcTensor(input), *token, split_dimension, concat_dimension,
      split_count, replica_groups);
  return std::pair<at::Tensor, std::shared_ptr<ir::Value>>(
      bridge::AtenFromLtcTensor(std::move(result)),
      std::make_shared<ir::Value>(new_token));
}

std::pair<at::Tensor, std::shared_ptr<ir::Value>> CollectivePermute(
    const at::Tensor& input, const std::shared_ptr<ir::Value>& token,
    const std::vector<std::pair<lazy_tensors::int64, lazy_tensors::int64>>&
        source_target_pairs) {
  LazyTensor result;
  ir::Value new_token;
  std::tie(result, new_token) = LazyTensor::collective_permute(
      bridge::GetLtcTensor(input), *token, source_target_pairs);
  return std::pair<at::Tensor, std::shared_ptr<ir::Value>>(
      bridge::AtenFromLtcTensor(std::move(result)),
      std::make_shared<ir::Value>(new_token));
}

void SyncTensors(const std::vector<at::Tensor>& tensors,
                 const std::vector<std::string>& devices, bool wait,
                 bool sync_ltc_data) {
  std::vector<LazyTensor> xtensors = GetLtcTensors(tensors, /*want_all=*/false);
  LazyTensor::SyncTensorsGraph(&xtensors, devices, wait, sync_ltc_data);
}

void SyncLiveTensors(const std::string& device_str,
                     const std::vector<std::string>& devices, bool wait) {
  auto opt_device = GetOptionalDevice(device_str);
  LazyTensor::SyncLiveTensorsGraph(opt_device ? &opt_device.value() : nullptr,
                                   devices, wait);
}

void StepMarker(const std::string& device_str,
                const std::vector<std::string>& devices, bool wait) {
  Device device = GetDeviceOrCurrent(device_str);
  LazyTensor::SyncLiveTensorsGraph(&device, devices, wait);
  LazyTensor::MarkStep(device);
  bool debug_mode = lazy_tensors::sys_util::GetEnvBool("PT_LTC_DEBUG", false);
  if (TF_PREDICT_FALSE(debug_mode)) {
    std::string report = lazy_tensors::metrics::CreatePerformanceReport();
    if (!report.empty()) {
      std::string fout =
          lazy_tensors::sys_util::GetEnvString("PT_LTC_DEBUG_FILE", "");
      if (TF_PREDICT_FALSE(!fout.empty())) {
        std::ofstream out_file(fout, std::ios_base::app);
        out_file << report;
      } else {
        std::cout << report;
      }
    }
  }
}

void SetRngSeed(lazy_tensors::uint64 seed, const std::string& device_str) {
  Device device = GetDeviceOrCurrent(device_str);
  LazyTensor::SetRngSeed(device, seed);
}

lazy_tensors::uint64 GetRngSeed(const std::string& device_str) {
  return LazyTensor::GetRunningSeed(GetDeviceOrCurrent(device_str));
}

std::string GetTensorsBackendGraph(const std::vector<at::Tensor>& tensors) {
  std::vector<LazyTensor> xtensors = GetLtcTensors(tensors, /*want_all=*/false);
  return LazyTensor::DumpBackendComputation(xtensors);
}

std::string GetLiveTensorsReport(size_t nodes_threshold,
                                 const std::string& device_str) {
  auto opt_device = GetOptionalDevice(device_str);
  auto tensors =
      LazyTensor::GetLiveTensors(opt_device ? &opt_device.value() : nullptr);
  std::stringstream ss;
  for (auto& tensor : tensors) {
    ir::Value ir_value = tensor.CurrentIrValue();
    if (ir_value) {
      std::vector<const ir::Node*> roots({ir_value.node.get()});
      auto post_order = ir::Util::ComputePostOrder(roots);
      if (post_order.size() > nodes_threshold) {
        ss << "Tensor: id=" << tensor.GetUniqueId()
           << ", shape=" << tensor.shape().get()
           << ", device=" << tensor.GetDevice()
           << ", ir_nodes=" << post_order.size() << "\n";
        for (size_t i = post_order.size(); i > 0; --i) {
          if (!post_order[i - 1]->metadata().frame_info.empty()) {
            ss << post_order[i - 1]->metadata().frame_info;
            break;
          }
        }
        ss << ir::DumpUtil::PostOrderToText(post_order, roots);
        ss << "\n\n";
      }
    }
  }
  return ss.str();
}

std::ptrdiff_t GetTensorViewAliasId(const at::Tensor& tensor) {
  LazyTensor xtensor = bridge::GetLtcTensor(tensor);
  return xtensor.GetViewAliasId();
}

std::ptrdiff_t GetTensorId(const at::Tensor& tensor) {
  LazyTensor xtensor = bridge::GetLtcTensor(tensor);
  return xtensor.GetUniqueId();
}

std::vector<at::Tensor> GetLtcTensorsFromAten(
    const std::vector<at::Tensor>& aten_tensors,
    const std::vector<std::string>& devices) {
  auto data_handles = CreateTensorsData(aten_tensors, GetLtcDevices(devices));

  std::vector<at::Tensor> lazy_tensors;
  lazy_tensors.reserve(data_handles.size());
  for (auto& data_handle : data_handles) {
    LazyTensor lazy_tensor = LazyTensor::Create(std::move(data_handle));
    lazy_tensors.push_back(bridge::AtenFromLtcTensor(std::move(lazy_tensor)));
  }
  return lazy_tensors;
}

std::shared_ptr<ir::Value> CreateToken(const std::string& device_str) {
  // This should be using lazy_tensors::CreateToken() once we have added Token
  // support to the backend AllReduce(). Meanwhile we use a constant as token,
  // and we handle it accordingly in cross_replica_reduces.cpp. This needs to be
  // device data (hence coming in as computation parameter) as otherwise the
  // backend compiler passes might remove it, vanishing its sequencing effects.
  Device device = GetDeviceOrCurrent(device_str);
  ir::Value ir_value = LazyTensor::GetDeviceDataIrValue(
      0.0, lazy_tensors::PrimitiveType::F32, device);
  return std::make_shared<ir::Value>(std::move(ir_value));
}

py::object GetMetricData(const std::string& name) {
  lazy_tensors::metrics::MetricData* data =
      lazy_tensors::metrics::GetMetric(name);
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

py::object LtcNms(const at::Tensor& boxes, const at::Tensor& scores,
                  const at::Tensor& score_threshold,
                  const at::Tensor& iou_threshold,
                  lazy_tensors::int64 output_size) {
  at::Tensor selected_indices;
  at::Tensor num_valid;
  {
    NoGilSection nogil;
    auto nms_result = LazyTensor::nms(
        bridge::GetLtcTensor(boxes), bridge::GetLtcTensor(scores),
        bridge::GetLtcTensor(score_threshold),
        bridge::GetLtcTensor(iou_threshold), output_size);
    selected_indices = bridge::AtenFromLtcTensor(std::move(nms_result.first));
    num_valid = bridge::AtenFromLtcTensor(std::move(nms_result.second));
  }
  auto result_tuple = py::tuple(2);
  result_tuple[0] =
      torch::autograd::make_variable(selected_indices, /*requires_grad=*/false);
  result_tuple[1] =
      torch::autograd::make_variable(num_valid, /*requires_grad=*/false);
  return result_tuple;
}

py::dict GetMemoryInfo(const std::string& device_str) {
  lazy_tensors::ComputationClient::MemoryInfo mem_info;
  {
    NoGilSection nogil;
    Device device = GetDeviceOrCurrent(device_str);
    mem_info = lazy_tensors::ComputationClient::Get()->GetMemoryInfo(
        device.ToString());
  }
  auto py_dict = py::dict();
  py_dict["kb_free"] = mem_info.kb_free;
  py_dict["kb_total"] = mem_info.kb_total;
  return py_dict;
}

void InitLtcModuleBindings(py::module m) {
  m.def("_prepare_to_exit", []() { PrepareToExit(); });
  m.def("_get_git_revs", []() { return GetRevisions(); });
  m.def("_ltc_nms", [](const at::Tensor& boxes, const at::Tensor& scores,
                       const at::Tensor& score_threshold,
                       const at::Tensor& iou_threshold,
                       lazy_tensors::int64 output_size) {
    return LtcNms(boxes, scores, score_threshold, iou_threshold, output_size);
  });
  m.def("_get_ltc_tensors_dot",
        [](const std::vector<at::Tensor>& tensors) -> std::string {
          auto coverter = [](lazy_tensors::Span<const ir::Node* const> nodes) {
            return ir::DumpUtil::ToDot(nodes);
          };
          return GetTensorsDump(tensors, coverter);
        });
  m.def("_get_ltc_tensors_text",
        [](const std::vector<at::Tensor>& tensors) -> std::string {
          auto coverter = [](lazy_tensors::Span<const ir::Node* const> nodes) {
            return ir::DumpUtil::ToText(nodes);
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
      std::vector<at::Tensor> cpu_tensors =
          bridge::LtcCreateTensorList(tensors);
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
  m.def("_ltc_get_devices", []() {
    return lazy_tensors::ComputationClient::Get()->GetLocalDevices();
  });
  m.def("_ltc_get_all_devices", []() {
    return lazy_tensors::ComputationClient::Get()->GetAllDevices();
  });
  m.def("_ltc_real_devices", [](const std::vector<std::string>& devices) {
    std::vector<std::string> ltc_devices;
    {
      NoGilSection nogil;
      ltc_devices = GetLtcDevices(devices);
    }
    return ltc_devices;
  });
  m.def("_ltc_set_replication_devices",
        [](const std::vector<std::string>& devices) {
          auto replication_devices =
              std::make_shared<std::vector<std::string>>(devices);
          lazy_tensors::ComputationClient::Get()->SetReplicationDevices(
              std::move(replication_devices));
        });
  m.def("_ltc_get_replication_devices", []() {
    auto replication_devices =
        lazy_tensors::ComputationClient::Get()->GetReplicationDevices();
    return replication_devices != nullptr ? *replication_devices
                                          : std::vector<std::string>();
  });
  m.def("_ltc_get_replication_devices_count", []() {
    auto replication_devices =
        lazy_tensors::ComputationClient::Get()->GetReplicationDevices();
    return replication_devices != nullptr ? replication_devices->size() : 0;
  });

  py::class_<ir::Value, std::shared_ptr<ir::Value>>(m, "IrValue");
  m.def("_ltc_create_token",
        [](const std::string& device) { return CreateToken(device); });
  m.def("_ltc_all_reduce_inplace", [](const std::string& reduce_type,
                                      const std::vector<at::Tensor>& tensors,
                                      const std::shared_ptr<ir::Value>& token,
                                      double scale, const py::list& groups) {
    std::vector<std::vector<lazy_tensors::int64>> replica_groups =
        CreateReduceGroups(groups);
    std::shared_ptr<ir::Value> new_token;
    {
      NoGilSection nogil;
      new_token =
          AllReduceInPlace(reduce_type, tensors, token, scale, replica_groups);
    }
    return new_token;
  });
  m.def("_ltc_all_reduce",
        [](const std::string& reduce_type, const at::Tensor& input,
           const std::shared_ptr<ir::Value>& token, double scale,
           const py::list& groups) {
          std::vector<std::vector<lazy_tensors::int64>> replica_groups =
              CreateReduceGroups(groups);
          at::Tensor result;
          std::shared_ptr<ir::Value> new_token;
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
  m.def("_ltc_all_to_all",
        [](const at::Tensor& input, const std::shared_ptr<ir::Value>& token,
           lazy_tensors::int64 split_dimension,
           lazy_tensors::int64 concat_dimension,
           lazy_tensors::int64 split_count, const py::list& groups) {
          std::vector<std::vector<lazy_tensors::int64>> replica_groups =
              CreateReduceGroups(groups);
          at::Tensor result;
          std::shared_ptr<ir::Value> new_token;
          {
            NoGilSection nogil;
            std::tie(result, new_token) =
                AllToAll(input, token, split_dimension, concat_dimension,
                         split_count, replica_groups);
          }
          auto result_tuple = py::tuple(2);
          result_tuple[0] = torch::autograd::make_variable(
              result, /*requires_grad=*/input.requires_grad());
          result_tuple[1] = new_token;
          return result_tuple;
        });
  m.def("_ltc_collective_permute",
        [](const at::Tensor& input, const std::shared_ptr<ir::Value>& token,
           const py::list& pairs) {
          std::vector<std::pair<lazy_tensors::int64, lazy_tensors::int64>>
              source_target_pairs = CreateSourceTargetPairs(pairs);
          at::Tensor result;
          std::shared_ptr<ir::Value> new_token;
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
    return SetCurrentThreadDevice(device);
  });
  m.def("_ltc_get_default_device", []() { return GetCurrentThreadDevice(); });
  m.def("_ltc_set_rng_seed",
        [](lazy_tensors::uint64 seed, const std::string& device) {
          SetRngSeed(seed, device);
        },
        py::arg("seed") = 101, py::arg("device") = "");
  m.def("_ltc_get_rng_seed",
        [](const std::string& device) { return GetRngSeed(device); },
        py::arg("device") = "");
  m.def("_ltc_sync_multi",
        [](const std::vector<at::Tensor>& tensors,
           const std::vector<std::string>& devices, bool wait,
           bool sync_ltc_data) {
          NoGilSection nogil;
          SyncTensors(tensors, devices, wait, sync_ltc_data);
        },
        py::arg("tensors"), py::arg("devices"), py::arg("wait") = true,
        py::arg("sync_ltc_data") = true);
  m.def("_ltc_sync_live_tensors",
        [](const std::string& device, const std::vector<std::string>& devices,
           bool wait) {
          NoGilSection nogil;
          SyncLiveTensors(device, devices, wait);
        },
        py::arg("device") = "", py::arg("devices"), py::arg("wait") = true);
  m.def("_ltc_step_marker",
        [](const std::string& device, const std::vector<std::string>& devices,
           bool wait) {
          NoGilSection nogil;
          StepMarker(device, devices, wait);
        },
        py::arg("device") = "", py::arg("devices"), py::arg("wait") = true);
  m.def("_ltc_wait_device_ops",
        [](const std::vector<std::string>& devices) {
          NoGilSection nogil;
          LazyTensor::WaitDeviceOps(devices);
        },
        py::arg("devices"));
  m.def("_ltc_counter_names",
        []() { return lazy_tensors::metrics::GetCounterNames(); });
  m.def("_ltc_counter_value", [](const std::string& name) -> py::object {
    lazy_tensors::metrics::CounterData* data =
        lazy_tensors::metrics::GetCounter(name);
    return data != nullptr ? py::cast<int64_t>(data->Value()) : py::none();
  });
  m.def("_ltc_metric_names",
        []() { return lazy_tensors::metrics::GetMetricNames(); });
  m.def("_ltc_metric_data", [](const std::string& name) -> py::object {
    return GetMetricData(name);
  });
  m.def("_ltc_metrics_report",
        []() { return lazy_tensors::metrics_reader::CreateMetricReport(); });
  m.def("_ltc_tensors_report",
        [](size_t nodes_threshold, const std::string& device) {
          return GetLiveTensorsReport(nodes_threshold, device);
        },
        py::arg("nodes_threshold") = 100, py::arg("device") = "");
  m.def("_ltc_memory_info", [](const std::string& device) -> py::object {
    return GetMemoryInfo(device);
  });
  m.def("_ltc_init_ts_backend", []() { compiler::InitTorchScriptBackend(); });
}

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
