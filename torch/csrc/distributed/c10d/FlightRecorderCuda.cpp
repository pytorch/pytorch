#ifdef USE_C10D_NCCL
#include <ATen/cuda/CUDAEvent.h>
#include <cuda_runtime.h>

#include <nlohmann/json.hpp>
#include <filesystem>
#include <fstream>
#include <mutex>
#include <vector>

#include <torch/csrc/distributed/c10d/FlightRecorderDetail.hpp>
#include <torch/csrc/distributed/c10d/ProcessGroupNCCL.hpp>
#include <torch/csrc/distributed/c10d/control_plane/Handlers.hpp>

namespace c10d {
control_plane::RegisterHandler dumpHandler{
    "dump_nccl_trace_pickle",
    [](const control_plane::Request& req, control_plane::Response& res) {
      const auto& params = req.params();
      size_t validParamCount = 0;

      // valid params
      const std::string includeCollectivesStr = "includecollectives";
      const std::string includeStackTracesStr = "includestacktraces";
      const std::string onlyActiveStr = "onlyactive";

      std::unordered_map<std::string, bool> processedParams = {
          {includeCollectivesStr, true},
          {includeStackTracesStr, true},
          {onlyActiveStr, false}};

      for (const auto& [paramName, paramValue] : params) {
        auto it = processedParams.find(paramName);
        if (it != processedParams.end()) {
          validParamCount++;
          if (paramValue == "true") {
            it->second = true;
          } else if (paramValue == "false") {
            it->second = false;
          } else {
            res.setStatus(400);
            res.setContent(
                "Invalid value for " + paramName +
                    " valid values are true or false",
                "text/plain");
            return;
          }
        }
      }
      if (validParamCount < params.size()) {
        res.setStatus(400);
        res.setContent(
            "Invalid parameters - unexpected param passed in", "text/plain");
        return;
      }
      res.setContent(
          dump_nccl_trace(
              processedParams[includeCollectivesStr],
              processedParams[includeStackTracesStr],
              processedParams[onlyActiveStr]),
          "application/octet-stream");
    }};

control_plane::RegisterHandler jsonDumpHandler{
    "dump_nccl_trace_json",
    [](const control_plane::Request& req, control_plane::Response& res) {
      const auto& params = req.params();
      size_t validParamCount = 0;

      // valid params
      const std::string includeCollectivesStr = "includecollectives";
      const std::string onlyActiveStr = "onlyactive";

      std::unordered_map<std::string, bool> processedParams = {
          {includeCollectivesStr, true}, {onlyActiveStr, false}};

      for (const auto& [paramName, paramValue] : params) {
        auto it = processedParams.find(paramName);
        if (it != processedParams.end()) {
          validParamCount++;
          if (paramValue == "true") {
            it->second = true;
          } else if (paramValue == "false") {
            it->second = false;
          } else {
            res.setStatus(400);
            res.setContent(
                "Invalid value for " + paramName +
                    " valid values are true or false",
                "text/plain");
            return;
          }
        }
      }
      if (validParamCount < params.size()) {
        res.setStatus(400);
        res.setContent(
            "Invalid parameters - unexpected param passed in", "text/plain");
        return;
      }
      res.setStatus(200);
      res.setContent(
          dump_nccl_trace_json(
              processedParams[includeCollectivesStr],
              processedParams[onlyActiveStr]),
          "application/json");
    }};

/* Helper used by work::getDuration() and nccl flight recorder */
template <>
float getDurationFromEvent<at::cuda::CUDAEvent>(
    at::cuda::CUDAEvent& ncclStartEvent,
    at::cuda::CUDAEvent& ncclEndEvent) {
  TORCH_CHECK(
      ncclEndEvent.query(),
      "getDuration can only be called after work is succeeded.")
  return ncclStartEvent.elapsed_time(ncclEndEvent);
}

template struct FlightRecorder<at::cuda::CUDAEvent>;
} // namespace c10d
#endif // USE_C10D_NCCL
