/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include "RocprofLogger.h"

#include <rocprofiler-sdk/context.h>
#include <rocprofiler-sdk/cxx/name_info.hpp>
#include <rocprofiler-sdk/fwd.h>
#include <rocprofiler-sdk/marker/api_id.h>
#include <rocprofiler-sdk/registration.h>
#include <rocprofiler-sdk/rocprofiler.h>

#include <time.h>
#include <unistd.h>
#include <chrono>
#include <cstring>
#include <mutex>

#include "ApproximateClock.h"
#include "Demangle.h"
#include "Logger.h"
#include "ThreadUtil.h"

using namespace libkineto;
using namespace std::chrono;
using namespace RocLogger;

namespace {

using kernel_symbol_data_t =
    rocprofiler_callback_tracing_code_object_kernel_symbol_register_data_t;
using kernel_symbol_map_t =
    std::unordered_map<rocprofiler_kernel_id_t, kernel_symbol_data_t>;
using kernel_name_map_t =
    std::unordered_map<rocprofiler_kernel_id_t, std::string>;
using rocprofiler::sdk::buffer_name_info;
using rocprofiler::sdk::callback_name_info;
using agent_info_map_t = std::unordered_map<uint64_t, rocprofiler_agent_v0_t>;

// extract copy args
struct copy_args {
  const char* dst{""};
  const char* src{""};
  size_t size{0};
  const char* copyKindStr{""};
  hipMemcpyKind copyKind{hipMemcpyDefault};
  hipStream_t stream{nullptr};
  rocprofiler_callback_tracing_kind_t kind;
  rocprofiler_tracing_operation_t operation;
};
auto extract_copy_args =
    []([[maybe_unused]] rocprofiler_callback_tracing_kind_t kind,
       [[maybe_unused]] rocprofiler_tracing_operation_t operation,
       [[maybe_unused]] uint32_t arg_num,
       const void* const arg_value_addr,
       [[maybe_unused]] int32_t indirection_count,
       [[maybe_unused]] const char* arg_type,
       const char* arg_name,
       const char* arg_value_str,
       [[maybe_unused]] int32_t dereference_count,
       void* cb_data) -> int {
  auto& args = *(static_cast<copy_args*>(cb_data));
  if (strcmp("dst", arg_name) == 0) {
    args.dst = arg_value_str;
  } else if (strcmp("src", arg_name) == 0) {
    args.src = arg_value_str;
  } else if (strcmp("sizeBytes", arg_name) == 0) {
    args.size = *(reinterpret_cast<const size_t*>(arg_value_addr));
  } else if (strcmp("kind", arg_name) == 0) {
    args.copyKindStr = arg_value_str;
    args.copyKind = *(reinterpret_cast<const hipMemcpyKind*>(arg_value_addr));
  } else if (strcmp("stream", arg_name) == 0) {
    args.stream = *(reinterpret_cast<const hipStream_t*>(arg_value_addr));
  }
  return 0;
};

// extract kernel args
struct kernel_args {
  // const char *stream;
  hipStream_t stream{nullptr};
  uint32_t privateSize{0};
  uint32_t groupSize{0};
  rocprofiler_dim3_t workgroupSize{0, 0, 0};
  rocprofiler_dim3_t gridSize{0, 0, 0};
  rocprofiler_callback_tracing_kind_t kind;
  rocprofiler_tracing_operation_t operation;
};
auto extract_kernel_args =
    []([[maybe_unused]] rocprofiler_callback_tracing_kind_t kind,
       [[maybe_unused]] rocprofiler_tracing_operation_t operation,
       [[maybe_unused]] uint32_t arg_num,
       const void* const arg_value_addr,
       [[maybe_unused]] int32_t indirection_count,
       [[maybe_unused]] const char* arg_type,
       const char* arg_name,
       [[maybe_unused]] const char* arg_value_str,
       [[maybe_unused]] int32_t dereference_count,
       void* cb_data) -> int {
  auto& args = *(static_cast<kernel_args*>(cb_data));
  if (strcmp("stream", arg_name) == 0)
    args.stream = *(reinterpret_cast<const hipStream_t*>(arg_value_addr));
  else if (strcmp("numBlocks", arg_name) == 0)
    args.workgroupSize =
        *(reinterpret_cast<const rocprofiler_dim3_t*>(arg_value_addr));
  else if (strcmp("dimBlocks", arg_name) == 0)
    args.gridSize =
        *(reinterpret_cast<const rocprofiler_dim3_t*>(arg_value_addr));
  else if (strcmp("sharedMemBytes", arg_name) == 0)
    args.groupSize = *(reinterpret_cast<const uint32_t*>(arg_value_addr));
  else if (strcmp("gridDimX", arg_name) == 0)
    args.workgroupSize.x = *(reinterpret_cast<const uint32_t*>(arg_value_addr));
  else if (strcmp("gridDimY", arg_name) == 0)
    args.workgroupSize.y = *(reinterpret_cast<const uint32_t*>(arg_value_addr));
  else if (strcmp("gridDimZ", arg_name) == 0)
    args.workgroupSize.z = *(reinterpret_cast<const uint32_t*>(arg_value_addr));
  else if (strcmp("blockDimX", arg_name) == 0)
    args.gridSize.x = *(reinterpret_cast<const uint32_t*>(arg_value_addr));
  else if (strcmp("blockDimY", arg_name) == 0)
    args.gridSize.y = *(reinterpret_cast<const uint32_t*>(arg_value_addr));
  else if (strcmp("blockDimZ", arg_name) == 0)
    args.gridSize.z = *(reinterpret_cast<const uint32_t*>(arg_value_addr));
  else if (strcmp("globalWorkSizeX", arg_name) == 0)
    args.workgroupSize.x = *(reinterpret_cast<const uint32_t*>(arg_value_addr));
  else if (strcmp("globalWorkSizeY", arg_name) == 0)
    args.workgroupSize.y = *(reinterpret_cast<const uint32_t*>(arg_value_addr));
  else if (strcmp("globalWorkSizeZ", arg_name) == 0)
    args.workgroupSize.z = *(reinterpret_cast<const uint32_t*>(arg_value_addr));
  else if (strcmp("localWorkSizeX", arg_name) == 0)
    args.gridSize.x = *(reinterpret_cast<const uint32_t*>(arg_value_addr));
  else if (strcmp("localWorkSizeY", arg_name) == 0)
    args.gridSize.y = *(reinterpret_cast<const uint32_t*>(arg_value_addr));
  else if (strcmp("localWorkSizeZ", arg_name) == 0)
    args.gridSize.z = *(reinterpret_cast<const uint32_t*>(arg_value_addr));
  return 0;
};

// extract malloc args
struct malloc_args {
  const char* ptr;
  size_t size;
};
auto extract_malloc_args =
    []([[maybe_unused]] rocprofiler_callback_tracing_kind_t kind,
       [[maybe_unused]] rocprofiler_tracing_operation_t operation,
       [[maybe_unused]] uint32_t arg_num,
       const void* const arg_value_addr,
       [[maybe_unused]] int32_t indirection_count,
       [[maybe_unused]] const char* arg_type,
       const char* arg_name,
       const char* arg_value_str,
       [[maybe_unused]] int32_t dereference_count,
       void* cb_data) -> int {
  auto& args = *(static_cast<malloc_args*>(cb_data));
  if (strcmp("ptr", arg_name) == 0) {
    args.ptr = arg_value_str;
  }
  if (strcmp("size", arg_name) == 0) {
    args.size = *(reinterpret_cast<const size_t*>(arg_value_addr));
  }
  return 0;
};

// copy api calls
bool isCopyApi(uint32_t id) {
  switch (id) {
    case ROCPROFILER_HIP_RUNTIME_API_ID_hipMemcpy:
    case ROCPROFILER_HIP_RUNTIME_API_ID_hipMemcpy2D:
    case ROCPROFILER_HIP_RUNTIME_API_ID_hipMemcpy2DAsync:
    case ROCPROFILER_HIP_RUNTIME_API_ID_hipMemcpy2DFromArray:
    case ROCPROFILER_HIP_RUNTIME_API_ID_hipMemcpy2DFromArrayAsync:
    case ROCPROFILER_HIP_RUNTIME_API_ID_hipMemcpy2DToArray:
    case ROCPROFILER_HIP_RUNTIME_API_ID_hipMemcpy2DToArrayAsync:
    case ROCPROFILER_HIP_RUNTIME_API_ID_hipMemcpy3D:
    case ROCPROFILER_HIP_RUNTIME_API_ID_hipMemcpy3DAsync:
    case ROCPROFILER_HIP_RUNTIME_API_ID_hipMemcpyAsync:
    case ROCPROFILER_HIP_RUNTIME_API_ID_hipMemcpyAtoH:
    case ROCPROFILER_HIP_RUNTIME_API_ID_hipMemcpyDtoD:
    case ROCPROFILER_HIP_RUNTIME_API_ID_hipMemcpyDtoDAsync:
    case ROCPROFILER_HIP_RUNTIME_API_ID_hipMemcpyDtoH:
    case ROCPROFILER_HIP_RUNTIME_API_ID_hipMemcpyDtoHAsync:
    case ROCPROFILER_HIP_RUNTIME_API_ID_hipMemcpyFromArray:
    case ROCPROFILER_HIP_RUNTIME_API_ID_hipMemcpyFromSymbol:
    case ROCPROFILER_HIP_RUNTIME_API_ID_hipMemcpyFromSymbolAsync:
    case ROCPROFILER_HIP_RUNTIME_API_ID_hipMemcpyHtoA:
    case ROCPROFILER_HIP_RUNTIME_API_ID_hipMemcpyHtoD:
    case ROCPROFILER_HIP_RUNTIME_API_ID_hipMemcpyHtoDAsync:
    case ROCPROFILER_HIP_RUNTIME_API_ID_hipMemcpyParam2D:
    case ROCPROFILER_HIP_RUNTIME_API_ID_hipMemcpyParam2DAsync:
    case ROCPROFILER_HIP_RUNTIME_API_ID_hipMemcpyPeer:
    case ROCPROFILER_HIP_RUNTIME_API_ID_hipMemcpyPeerAsync:
    case ROCPROFILER_HIP_RUNTIME_API_ID_hipMemcpyToArray:
    case ROCPROFILER_HIP_RUNTIME_API_ID_hipMemcpyToSymbol:
    case ROCPROFILER_HIP_RUNTIME_API_ID_hipMemcpyToSymbolAsync:
    case ROCPROFILER_HIP_RUNTIME_API_ID_hipMemcpyWithStream:
      return true;
      break;
    default:;
  }
  return false;
}

// kernel api calls
bool isKernelApi(uint32_t id) {
  switch (id) {
    case ROCPROFILER_HIP_RUNTIME_API_ID_hipExtLaunchKernel:
    case ROCPROFILER_HIP_RUNTIME_API_ID_hipExtLaunchMultiKernelMultiDevice:
    case ROCPROFILER_HIP_RUNTIME_API_ID_hipLaunchCooperativeKernel:
    case ROCPROFILER_HIP_RUNTIME_API_ID_hipLaunchCooperativeKernelMultiDevice:
    case ROCPROFILER_HIP_RUNTIME_API_ID_hipLaunchKernel:
    case ROCPROFILER_HIP_RUNTIME_API_ID_hipModuleLaunchCooperativeKernel:
    case ROCPROFILER_HIP_RUNTIME_API_ID_hipModuleLaunchCooperativeKernelMultiDevice:
    case ROCPROFILER_HIP_RUNTIME_API_ID_hipModuleLaunchKernel:
    case ROCPROFILER_HIP_RUNTIME_API_ID_hipExtModuleLaunchKernel:
    case ROCPROFILER_HIP_RUNTIME_API_ID_hipHccModuleLaunchKernel:
    case ROCPROFILER_HIP_RUNTIME_API_ID_hipLaunchCooperativeKernel_spt:
    case ROCPROFILER_HIP_RUNTIME_API_ID_hipLaunchKernel_spt:
      return true;
      break;
    default:;
  }
  return false;
}

// malloc api calls
bool isMallocApi(uint32_t id) {
  switch (id) {
    case ROCPROFILER_HIP_RUNTIME_API_ID_hipMalloc:
    case ROCPROFILER_HIP_RUNTIME_API_ID_hipFree:
      return true;
      break;
    default:;
  }
  return false;
}

class RocprofApiIdList : public ApiIdList {
 public:
  RocprofApiIdList(callback_name_info& names);
  uint32_t mapName(const std::string& apiName) override;
  std::vector<rocprofiler_tracing_operation_t> allEnabled();

 private:
  std::unordered_map<std::string, size_t> nameMap_;
};

struct GlobalContext {
  rocprofiler_client_id_t* clientId{nullptr};
  rocprofiler_tool_configure_result_t cfg = rocprofiler_tool_configure_result_t{
      sizeof(rocprofiler_tool_configure_result_t),
      &RocprofLogger::toolInit,
      &RocprofLogger::toolFinalize,
      nullptr};

  // Contexts
  rocprofiler_context_id_t utilityContext = {0};
  rocprofiler_context_id_t context = {0};

  // Buffers
  rocprofiler_buffer_id_t buffer = {};

  // Manage kernel names - #betterThanRoctracer
  kernel_symbol_map_t kernel_info = {};
  kernel_name_map_t kernel_names = {};
  std::mutex kernel_lock;

  // Manage buffer name - #betterThanRoctracer
  callback_name_info name_info = {};
  buffer_name_info buff_name_info = {};

  // Agent info
  // <rocprofiler_profile_config_id_t.handle, rocprofiler_agent_v0_t>
  agent_info_map_t agents = {};
};

GlobalContext& getGlobalContext() {
  static GlobalContext instance;
  return instance;
}

std::vector<rocprofiler_agent_v0_t> get_gpu_device_agents() {
  std::vector<rocprofiler_agent_v0_t> agents;

  // Callback used by rocprofiler_query_available_agents to return
  // agents on the device. This can include CPU agents as well. We
  // select GPU agents only (i.e. type == ROCPROFILER_AGENT_TYPE_GPU)
  rocprofiler_query_available_agents_cb_t iterate_cb =
      [](rocprofiler_agent_version_t agents_ver,
         const void** agents_arr,
         size_t num_agents,
         void* udata) {
        if (agents_ver != ROCPROFILER_AGENT_INFO_VERSION_0)
          throw std::runtime_error{"unexpected rocprofiler agent version"};
        auto* agents_v =
            static_cast<std::vector<rocprofiler_agent_v0_t>*>(udata);
        for (size_t i = 0; i < num_agents; ++i) {
          const auto* agent =
              static_cast<const rocprofiler_agent_v0_t*>(agents_arr[i]);
          // if(agent->type == ROCPROFILER_AGENT_TYPE_GPU)
          // agents_v->emplace_back(*agent);
          agents_v->emplace_back(*agent);
        }
        return ROCPROFILER_STATUS_SUCCESS;
      };

  // Query the agents, only a single callback is made that contains a vector
  // of all agents.
  rocprofiler_query_available_agents(
      ROCPROFILER_AGENT_INFO_VERSION_0,
      iterate_cb,
      sizeof(rocprofiler_agent_t),
      const_cast<void*>(static_cast<const void*>(&agents)));
  return agents;
}
} // namespace

//
// Static setup
//
extern "C" rocprofiler_tool_configure_result_t* rocprofiler_configure(
    [[maybe_unused]] uint32_t version,
    [[maybe_unused]] const char* runtime_version,
    [[maybe_unused]] uint32_t priority,
    rocprofiler_client_id_t* id) {
  auto& globalContext = getGlobalContext();

  id->name = "kineto";
  globalContext.clientId = id;

  // return pointer to configure data
  return &globalContext.cfg;
}

int RocprofLogger::toolInit(
    [[maybe_unused]] rocprofiler_client_finalize_t finialize_func,
    [[maybe_unused]] void* tool_data) {
  // Gather api names
  auto& globalContext = getGlobalContext();
  globalContext.name_info = rocprofiler::sdk::get_callback_tracing_names();
  globalContext.buff_name_info = rocprofiler::sdk::get_buffer_tracing_names();

  // Gather agent info
  auto agent_info = get_gpu_device_agents();
  for (auto agent : agent_info) {
    globalContext.agents[agent.id.handle] = agent;
  }

  //
  // Setup utility context to gather code object info
  //
  rocprofiler_create_context(&globalContext.utilityContext);
  auto code_object_ops = std::vector<rocprofiler_tracing_operation_t>{
      ROCPROFILER_CODE_OBJECT_DEVICE_KERNEL_SYMBOL_REGISTER};

  rocprofiler_configure_callback_tracing_service(
      globalContext.utilityContext,
      ROCPROFILER_CALLBACK_TRACING_CODE_OBJECT,
      code_object_ops.data(),
      code_object_ops.size(),
      RocprofLogger::code_object_callback,
      nullptr);
  {
    int isValid = 0;
    rocprofiler_context_is_valid(globalContext.utilityContext, &isValid);
    if (isValid == 0) {
      globalContext.utilityContext.handle = 0; // Can't destroy it, so leak it
      return -1;
    }
  }
  rocprofiler_start_context(globalContext.utilityContext);

  //
  // select some api calls to omit, in the most inconvenient way possible
  // #betterThanRoctracer
  RocprofApiIdList apiList(globalContext.name_info);
  apiList.setInvertMode(true); // Omit the specified api
  apiList.add("hipGetDevice");
  apiList.add("hipSetDevice");
  apiList.add("hipGetLastError");
  apiList.add("__hipPushCallConfiguration");
  apiList.add("__hipPopCallConfiguration");
  apiList.add("hipCtxSetCurrent");
  apiList.add("hipGetDevicePropertiesR0600");
  apiList.add("hipGetDeviceCount");
  apiList.add("hipDeviceGetAttribute");
  apiList.add("hipRuntimeGetVersion");
  apiList.add("hipPeekAtLastError");
  apiList.add("hipModuleGetFunction");

  // Get a vector of the enabled api calls
  auto apis = apiList.allEnabled();

  //
  // Setup main context to collect runtime and kernel info
  //
  rocprofiler_create_context(&globalContext.context);

  // Collect api info via callback
  rocprofiler_configure_callback_tracing_service(
      globalContext.context,
      ROCPROFILER_CALLBACK_TRACING_HIP_RUNTIME_API,
      apis.data(),
      apis.size(),
      api_callback,
      nullptr);

  // Collect async ops via buffers
  constexpr auto buffer_size_bytes = 0x40000;
  constexpr auto buffer_watermark_bytes = buffer_size_bytes / 2;

  rocprofiler_create_buffer(
      globalContext.context,
      buffer_size_bytes,
      buffer_watermark_bytes,
      ROCPROFILER_BUFFER_POLICY_LOSSLESS,
      RocprofLogger::buffer_callback,
      nullptr,
      &globalContext.buffer);

  rocprofiler_configure_buffer_tracing_service(
      globalContext.context,
      ROCPROFILER_BUFFER_TRACING_KERNEL_DISPATCH,
      nullptr,
      0,
      globalContext.buffer);

  rocprofiler_configure_buffer_tracing_service(
      globalContext.context,
      ROCPROFILER_BUFFER_TRACING_MEMORY_COPY,
      nullptr,
      0,
      globalContext.buffer);
  {
    int isValid = 0;
    rocprofiler_context_is_valid(globalContext.context, &isValid);
    if (isValid == 0) {
      globalContext.context.handle = 0; // Can't destroy it, so leak it
      return -1;
    }
  }
  rocprofiler_stop_context(globalContext.context);

  return 0;
}

void RocprofLogger::toolFinalize([[maybe_unused]] void* tool_data) {
  auto& globalContext = getGlobalContext();
  rocprofiler_stop_context(globalContext.utilityContext);
  globalContext.utilityContext.handle = 0;
  rocprofiler_stop_context(globalContext.context);
  globalContext.context.handle = 0;
}

class Flush {
 public:
  std::mutex mutex_;
  std::atomic<uint64_t> maxCorrelationId_;
  uint64_t maxCompletedCorrelationId_{0};
  void reportCorrelation(const uint64_t& cid) {
    uint64_t prev = maxCorrelationId_;
    while (prev < cid && !maxCorrelationId_.compare_exchange_weak(prev, cid)) {
    }
  }
};

RocprofLogger& RocprofLogger::singleton() {
  static RocprofLogger instance;
  return instance;
}

RocprofLogger::RocprofLogger() {}

RocprofLogger::~RocprofLogger() {
  stopLogging();
  endTracing();
}

namespace {
thread_local std::deque<uint64_t>
    t_externalIds[RocLogger::CorrelationDomain::size];
}

void RocprofLogger::pushCorrelationID(uint64_t id, CorrelationDomain type) {
  if (!singleton().externalCorrelationEnabled_) {
    return;
  }
  t_externalIds[type].push_back(id);
}

void RocprofLogger::popCorrelationID(CorrelationDomain type) {
  if (!singleton().externalCorrelationEnabled_) {
    return;
  }
  if (!t_externalIds[type].empty()) {
    t_externalIds[type].pop_back();
  } else {
    LOG(ERROR)
        << "Attempt to popCorrelationID from an empty external Ids stack";
  }
}

void RocprofLogger::clearLogs() {
  // CuptiActivityProfiler clears this before the output Loggers use the data
  // for (auto &row : rows_)
  //  delete row;
  rows_.clear();
  for (int i = 0; i < CorrelationDomain::size; ++i) {
    externalCorrelations_[i].clear();
  }
}

void RocprofLogger::insert_row_to_buffer(rocprofBase* row) {
  RocprofLogger* dis = &singleton();
  std::lock_guard<std::mutex> lock(dis->rowsMutex_);
  if (dis->rows_.size() >= dis->maxBufferSize_) {
    LOG_FIRST_N(WARNING, 10)
        << "Exceeded max GPU buffer count (" << dis->rows_.size() << " > "
        << dis->maxBufferSize_ << ") - terminating tracing";
    return;
  }
  dis->rows_.push_back(row);
}

void RocprofLogger::code_object_callback(
    rocprofiler_callback_tracing_record_t record,
    [[maybe_unused]] rocprofiler_user_data_t* user_data,
    [[maybe_unused]] void* callback_data) {
  if (record.kind == ROCPROFILER_CALLBACK_TRACING_CODE_OBJECT &&
      record.operation == ROCPROFILER_CODE_OBJECT_LOAD) {
    if (record.phase == ROCPROFILER_CALLBACK_PHASE_UNLOAD) {
      // flush the buffer to ensure that any lookups for the client kernel names
      // for the code object are completed NOTE: not using buffer ATM
    }
  } else if (
      record.kind == ROCPROFILER_CALLBACK_TRACING_CODE_OBJECT &&
      record.operation ==
          ROCPROFILER_CODE_OBJECT_DEVICE_KERNEL_SYMBOL_REGISTER) {
    auto* data = static_cast<kernel_symbol_data_t*>(record.payload);
    if (record.phase == ROCPROFILER_CALLBACK_PHASE_LOAD) {
      auto& globalContext = getGlobalContext();
      std::lock_guard<std::mutex> lock(globalContext.kernel_lock);
      globalContext.kernel_info.emplace(data->kernel_id, *data);
      globalContext.kernel_names.emplace(
          data->kernel_id, demangle(data->kernel_name));
    } else if (record.phase == ROCPROFILER_CALLBACK_PHASE_UNLOAD) {
      // FIXME: clear these?  At minimum need kernel names at shutdown, async
      // completion
      // globalContext.kernel_info.erase(data->kernel_id);
      // globalContext.kernel_names.erase(data->kernel_id);
    }
  }
}

void RocprofLogger::api_callback(
    rocprofiler_callback_tracing_record_t record,
    [[maybe_unused]] rocprofiler_user_data_t* user_data,
    [[maybe_unused]] void* callback_data) {
  thread_local std::unordered_map<uint64_t, uint64_t> timestamps;

  if (record.kind == ROCPROFILER_CALLBACK_TRACING_HIP_RUNTIME_API) {
    if (record.phase == ROCPROFILER_CALLBACK_PHASE_ENTER) {
      timestamps[record.correlation_id.internal] = getApproximateTime();
    } // ROCPROFILER_CALLBACK_PHASE_ENTER
    else { // ROCPROFILER_CALLBACK_PHASE_EXIT
      uint64_t startTime = timestamps[record.correlation_id.internal];
      timestamps.erase(record.correlation_id.internal);
      uint64_t endTime = getApproximateTime();

      // Kernel Launch Records
      if (isKernelApi(record.operation)) {
        kernel_args args;
        rocprofiler_iterate_callback_tracing_kind_operation_args(
            record,
            extract_kernel_args,
            1 /*max_deref*/
            ,
            &args);

        rocprofKernelRow* row = new rocprofKernelRow(
            record.correlation_id.internal,
            record.kind,
            record.operation,
            processId(),
            systemThreadId(),
            startTime,
            endTime,
            nullptr,
            nullptr,
            args.workgroupSize.x,
            args.workgroupSize.y,
            args.workgroupSize.z,
            args.gridSize.x,
            args.gridSize.y,
            args.gridSize.z,
            args.groupSize,
            args.stream);
        insert_row_to_buffer(row);

      }
      // Copy Records
      else if (isCopyApi(record.operation)) {
        copy_args args;
        rocprofiler_iterate_callback_tracing_kind_operation_args(
            record,
            extract_copy_args,
            1 /*max_deref*/
            ,
            &args);

        rocprofCopyRow* row = new rocprofCopyRow(
            record.correlation_id.internal,
            record.kind,
            record.operation,
            processId(),
            systemThreadId(),
            startTime,
            endTime,
            args.src,
            args.dst,
            args.size,
            args.copyKind,
            args.stream);
        insert_row_to_buffer(row);
      }
      // Malloc Records
      else if (isMallocApi(record.operation)) {
        malloc_args args;
        args.size = 0;
        rocprofiler_iterate_callback_tracing_kind_operation_args(
            record,
            extract_malloc_args,
            1 /*max_deref*/
            ,
            &args);
        rocprofMallocRow* row = new rocprofMallocRow(
            record.correlation_id.internal,
            record.kind,
            record.operation,
            processId(),
            systemThreadId(),
            startTime,
            endTime,
            args.ptr,
            args.size);
        insert_row_to_buffer(row);
      }
      // Default Records
      else {
        rocprofRow* row = new rocprofRow(
            record.correlation_id.internal,
            record.kind,
            record.operation,
            processId(),
            systemThreadId(),
            startTime,
            endTime);
        insert_row_to_buffer(row);
      }
      // External correlation
      static RocprofLogger* dis = &singleton();
      for (int it = RocLogger::CorrelationDomain::begin;
           it < RocLogger::CorrelationDomain::end;
           ++it) {
        if (t_externalIds[it].size() > 0) {
          std::lock_guard<std::mutex> lock(dis->externalCorrelationsMutex_);
          dis->externalCorrelations_[it].emplace_back(
              record.correlation_id.internal, t_externalIds[it].back());
        }
      }
    } // ROCPROFILER_CALLBACK_PHASE_EXIT
  } // ROCPROFILER_CALLBACK_TRACING_HIP_RUNTIME_API
}

void RocprofLogger::buffer_callback(
    [[maybe_unused]] rocprofiler_context_id_t context,
    [[maybe_unused]] rocprofiler_buffer_id_t buffer_id,
    rocprofiler_record_header_t** headers,
    size_t num_headers,
    [[maybe_unused]] void* user_data,
    [[maybe_unused]] uint64_t drop_count) {
  for (size_t i = 0; i < num_headers; ++i) {
    auto* header = headers[i];

    if (header->category == ROCPROFILER_BUFFER_CATEGORY_TRACING) {
      if (header->kind == ROCPROFILER_BUFFER_TRACING_KERNEL_DISPATCH) {
        auto& record =
            *(static_cast<rocprofiler_buffer_tracing_kernel_dispatch_record_t*>(
                header->payload));
        auto& dispatch = record.dispatch_info;

        // Safe access to agents map with default value
        auto& globalContext = getGlobalContext();
        auto agent_it = globalContext.agents.find(dispatch.agent_id.handle);
        int device_id = (agent_it != globalContext.agents.end())
            ? agent_it->second.logical_node_type_id
            : -1;

        // Safe access to kernel_names map with default value
        auto kernel_it = globalContext.kernel_names.find(dispatch.kernel_id);
        std::string kernel_name =
            (kernel_it != globalContext.kernel_names.end())
            ? kernel_it->second
            : "<unknown kernel>";

        rocprofAsyncRow* row = new rocprofAsyncRow(
            record.correlation_id.internal,
            record.kind,
            record.operation,
            record.operation, // shared op - No longer a thing.  Placeholder
            device_id,
            dispatch.queue_id.handle,
            record.start_timestamp,
            record.end_timestamp,
            kernel_name);
        insert_row_to_buffer(row);
      } else if (header->kind == ROCPROFILER_BUFFER_TRACING_MEMORY_COPY) {
        auto& record =
            *(static_cast<rocprofiler_buffer_tracing_memory_copy_record_t*>(
                header->payload));

        // Safe access to agents map with default value
        auto& globalContext = getGlobalContext();
        auto agent_it = globalContext.agents.find(record.dst_agent_id.handle);
        int device_id = (agent_it != globalContext.agents.end())
            ? agent_it->second.logical_node_type_id
            : -1;

        rocprofAsyncRow* row = new rocprofAsyncRow(
            record.correlation_id.internal,
            record.kind,
            record.operation,
            record.operation, // shared op - No longer a thing.  Placeholder
            device_id,
            0,
            record.start_timestamp,
            record.end_timestamp,
            "");
        insert_row_to_buffer(row);
      }
    }
  }
}

std::string RocprofLogger::opString(
    rocprofiler_callback_tracing_kind_t kind,
    rocprofiler_tracing_operation_t op) {
  auto& globalContext = getGlobalContext();
  auto& ops = globalContext.name_info[kind].operations;
  if (op >= 0 && static_cast<size_t>(op) < ops.size()) {
    return std::string(ops[op]);
  }
  return "<unknown operation:" + std::to_string(op) + ">";
}

std::string RocprofLogger::opString(
    rocprofiler_buffer_tracing_kind_t kind,
    rocprofiler_tracing_operation_t op) {
  auto& globalContext = getGlobalContext();
  auto& ops = globalContext.buff_name_info[kind].operations;
  if (op >= 0 && static_cast<size_t>(op) < ops.size()) {
    return std::string(ops[op]);
  }
  return "<unknown operation:" + std::to_string(op) + ">";
}

void RocprofLogger::setMaxEvents(uint32_t maxBufferSize) {
  RocprofLogger* dis = &singleton();
  std::lock_guard<std::mutex> lock(dis->rowsMutex_);
  maxBufferSize_ = maxBufferSize;
}

void RocprofLogger::ensureRegistered() {
  int status = 0;
  rocprofiler_is_initialized(&status);
  VLOG(0) << "rocprofiler_is_initialized returned " << status;
  if (status == 0) {
    VLOG(0) << "Forcing rocprofiler-sdk tool registration";
    auto result = rocprofiler_force_configure(&rocprofiler_configure);
    if (result == ROCPROFILER_STATUS_SUCCESS) {
      VLOG(0) << "rocprofiler-sdk tool registration completed successfully";
      singleton().registered_ = true;
    } else {
      LOG(WARNING) << "rocprofiler_force_configure failed with status "
                   << result;
    }
  } else if (status == 1) {
    singleton().registered_ = true;
  }
}

void RocprofLogger::startLogging() {
  if (!registered_) {
    ensureRegistered();
  }

  externalCorrelationEnabled_ = true;
  logging_ = true;
  auto& globalContext = getGlobalContext();
  rocprofiler_start_context(globalContext.context);
}

void RocprofLogger::stopLogging() {
  if (logging_ == false)
    return;
  logging_ = false;

  // Flush buffers
  auto& globalContext = getGlobalContext();
  rocprofiler_flush_buffer(globalContext.buffer);
  rocprofiler_stop_context(globalContext.context);
}

void RocprofLogger::endTracing() {
  // This should be handled in RocprofLogger::toolFinalize
}

//
// ApiIdList
//   Jump through some extra hoops
//
//
RocprofApiIdList::RocprofApiIdList(callback_name_info& names) : nameMap_() {
  auto& hipapis =
      names[ROCPROFILER_CALLBACK_TRACING_HIP_RUNTIME_API].operations;

  for (size_t i = 0; i < hipapis.size(); ++i) {
    nameMap_.emplace(hipapis[i], i);
  }
}

uint32_t RocprofApiIdList::mapName(const std::string& apiName) {
  auto it = nameMap_.find(apiName);
  if (it != nameMap_.end()) {
    return it->second;
  }
  return 0;
}

std::vector<rocprofiler_tracing_operation_t> RocprofApiIdList::allEnabled() {
  std::vector<rocprofiler_tracing_operation_t> oplist;
  for (auto& it : nameMap_) {
    if (contains(it.second))
      oplist.push_back(it.second);
  }
  return oplist;
}
