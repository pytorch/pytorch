/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include "RocprofLogger.h"

#include <rocprofiler-sdk/context.h>
#include <rocprofiler-sdk/cxx/hash.hpp>
#include <rocprofiler-sdk/cxx/name_info.hpp>
#include <rocprofiler-sdk/cxx/operators.hpp>
#include <rocprofiler-sdk/fwd.h>
#include <rocprofiler-sdk/marker/api_id.h>
#include <rocprofiler-sdk/registration.h>
#include <rocprofiler-sdk/rocprofiler.h>

#include <unistd.h>
#include <array>
#include <chrono>
#include <cstddef>
#include <cstring>
#include <ctime>
#include <mutex>
#include <new>
#include <thread>
#include <type_traits>
#include <unordered_map>
#include <vector>

#include <fmt/format.h>

#include "ApproximateClock.h"
#include "Demangle.h"
#include "DeviceUtil.h"
#include "Logger.h"
#include "ThreadUtil.h"
#include "ThrowUtil.h"

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
using agent_info_map_t =
    std::unordered_map<rocprofiler_agent_id_t, rocprofiler_agent_v0_t>;

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

// memset args
struct memset_args {
  size_t sizeBytes{0};
  size_t width{0};
  size_t height{0};
  size_t count{0};
};
auto extract_memset_args =
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
  auto& args = *(static_cast<memset_args*>(cb_data));
  if (strcmp("sizeBytes", arg_name) == 0)
    args.sizeBytes = *(reinterpret_cast<const size_t*>(arg_value_addr));
  else if (strcmp("width", arg_name) == 0)
    args.width = *(reinterpret_cast<const size_t*>(arg_value_addr));
  else if (strcmp("height", arg_name) == 0)
    args.height = *(reinterpret_cast<const size_t*>(arg_value_addr));
  else if (strcmp("count", arg_name) == 0)
    args.count = *(reinterpret_cast<const size_t*>(arg_value_addr));
  return 0;
};

// Total bytes filled by a HIP memset API call, or 0 if unknown.
size_t memsetTotalBytes(uint32_t operation, const memset_args& args) {
  if (args.sizeBytes > 0)
    return args.sizeBytes;
  if (args.width > 0 && args.height > 0)
    return args.width * args.height;
  // hipMemsetD8/D16/D32 fill 'count' elements of known width.
  switch (operation) {
    case ROCPROFILER_HIP_RUNTIME_API_ID_hipMemsetD8:
    case ROCPROFILER_HIP_RUNTIME_API_ID_hipMemsetD8Async:
      return args.count;
    case ROCPROFILER_HIP_RUNTIME_API_ID_hipMemsetD16:
    case ROCPROFILER_HIP_RUNTIME_API_ID_hipMemsetD16Async:
      return args.count * 2;
    case ROCPROFILER_HIP_RUNTIME_API_ID_hipMemsetD32:
    case ROCPROFILER_HIP_RUNTIME_API_ID_hipMemsetD32Async:
      return args.count * 4;
    default:
      break;
  }
  return 0;
}

// memset api calls
bool isMemsetApi(uint32_t id) {
  switch (id) {
    case ROCPROFILER_HIP_RUNTIME_API_ID_hipMemset:
    case ROCPROFILER_HIP_RUNTIME_API_ID_hipMemsetAsync:
    case ROCPROFILER_HIP_RUNTIME_API_ID_hipMemset2D:
    case ROCPROFILER_HIP_RUNTIME_API_ID_hipMemset2DAsync:
    case ROCPROFILER_HIP_RUNTIME_API_ID_hipMemset3D:
    case ROCPROFILER_HIP_RUNTIME_API_ID_hipMemset3DAsync:
    case ROCPROFILER_HIP_RUNTIME_API_ID_hipMemsetD8:
    case ROCPROFILER_HIP_RUNTIME_API_ID_hipMemsetD8Async:
    case ROCPROFILER_HIP_RUNTIME_API_ID_hipMemsetD16:
    case ROCPROFILER_HIP_RUNTIME_API_ID_hipMemsetD16Async:
    case ROCPROFILER_HIP_RUNTIME_API_ID_hipMemsetD32:
    case ROCPROFILER_HIP_RUNTIME_API_ID_hipMemsetD32Async:
    case ROCPROFILER_HIP_RUNTIME_API_ID_hipMemset_spt:
    case ROCPROFILER_HIP_RUNTIME_API_ID_hipMemsetAsync_spt:
    case ROCPROFILER_HIP_RUNTIME_API_ID_hipMemset2D_spt:
    case ROCPROFILER_HIP_RUNTIME_API_ID_hipMemset2DAsync_spt:
    case ROCPROFILER_HIP_RUNTIME_API_ID_hipMemset3D_spt:
    case ROCPROFILER_HIP_RUNTIME_API_ID_hipMemset3DAsync_spt:
#if HIP_RUNTIME_API_TABLE_STEP_VERSION >= 14
    case ROCPROFILER_HIP_RUNTIME_API_ID_hipMemsetD2D8:
    case ROCPROFILER_HIP_RUNTIME_API_ID_hipMemsetD2D8Async:
    case ROCPROFILER_HIP_RUNTIME_API_ID_hipMemsetD2D16:
    case ROCPROFILER_HIP_RUNTIME_API_ID_hipMemsetD2D16Async:
    case ROCPROFILER_HIP_RUNTIME_API_ID_hipMemsetD2D32:
    case ROCPROFILER_HIP_RUNTIME_API_ID_hipMemsetD2D32Async:
#endif
      return true;
    default:
      break;
  }
  return false;
}

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

constexpr auto null_context_id = rocprofiler_context_id_t{0};
constexpr auto null_buffer_id = rocprofiler_buffer_id_t{0};

struct GlobalContext {
  rocprofiler_client_id_t* clientId{nullptr};
  rocprofiler_client_finalize_t finalizer = nullptr;

  rocprofiler_tool_configure_result_t cfg = rocprofiler_tool_configure_result_t{
      sizeof(rocprofiler_tool_configure_result_t),
      &RocprofLogger::toolInit,
      &RocprofLogger::toolFinalize,
      nullptr};

  // Contexts
  rocprofiler_context_id_t utilityContext = null_context_id;
  rocprofiler_context_id_t context = null_context_id;

  // Buffers
  rocprofiler_buffer_id_t buffer = null_buffer_id;

  // Manage kernel names - #betterThanRoctracer
  kernel_symbol_map_t kernel_info = {};
  kernel_name_map_t kernel_names = {};
  std::mutex kernel_lock;

  // Manage buffer name - #betterThanRoctracer
  callback_name_info name_info = {};
  buffer_name_info buff_name_info = {};

  // Agent info
  // <rocprofiler_agent_id_t, rocprofiler_agent_v0_t>
  agent_info_map_t agents = {};
};

GlobalContext& getGlobalContext() {
  static auto* instance = new GlobalContext();
  return *instance;
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
          KINETO_THROW(
              std::runtime_error, "unexpected rocprofiler agent version");
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
  // of all agents. On failure the vector stays empty and callers fall back
  // to device id -1, so a warning is sufficient.
  ROCPROF_CALL(rocprofiler_query_available_agents(
      ROCPROFILER_AGENT_INFO_VERSION_0,
      iterate_cb,
      sizeof(rocprofiler_agent_t),
      &agents));
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
    rocprofiler_client_finalize_t finalize_func,
    [[maybe_unused]] void* tool_data) {
  auto& globalContext = getGlobalContext();
  try {
    // Gather api names
    globalContext.name_info = rocprofiler::sdk::get_callback_tracing_names();
    globalContext.buff_name_info = rocprofiler::sdk::get_buffer_tracing_names();

    // Gather agent info
    auto agent_info = get_gpu_device_agents();
    for (auto agent : agent_info) {
      globalContext.agents[agent.id] = agent;
    }

    //
    // Setup utility context to gather code object info
    //
    // Failures below are only observable through their return status and
    // each one leaves the tool unable to trace, so abort init rather than
    // logging and continuing with a registered-but-deaf tool.
    ROCPROF_CALL_THROW(
        rocprofiler_create_context(&globalContext.utilityContext));
    auto code_object_ops = std::vector<rocprofiler_tracing_operation_t>{
        ROCPROFILER_CODE_OBJECT_DEVICE_KERNEL_SYMBOL_REGISTER};

    ROCPROF_CALL_THROW(rocprofiler_configure_callback_tracing_service(
        globalContext.utilityContext,
        ROCPROFILER_CALLBACK_TRACING_CODE_OBJECT,
        code_object_ops.data(),
        code_object_ops.size(),
        RocprofLogger::code_object_callback,
        nullptr));

    // rocprofiler_context_is_valid returns the validation status itself,
    // so a successful status implies isValid == 1.
    int isValid = 0;
    ROCPROF_CALL_THROW(
        rocprofiler_context_is_valid(globalContext.utilityContext, &isValid));
    ROCPROF_CALL_THROW(rocprofiler_start_context(globalContext.utilityContext));

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
    ROCPROF_CALL_THROW(rocprofiler_create_context(&globalContext.context));

    // Collect api info via callback
    ROCPROF_CALL_THROW(rocprofiler_configure_callback_tracing_service(
        globalContext.context,
        ROCPROFILER_CALLBACK_TRACING_HIP_RUNTIME_API,
        apis.data(),
        apis.size(),
        api_callback,
        nullptr));

    // Learn every kernel dispatch at enqueue (including each kernel of a graph
    // launch and runtime-internal blits), so stopLogging() knows which
    // dispatch records are still owed.
    auto dispatch_ops = std::vector<rocprofiler_tracing_operation_t>{
        ROCPROFILER_KERNEL_DISPATCH_ENQUEUE};
    ROCPROF_CALL_THROW(rocprofiler_configure_callback_tracing_service(
        globalContext.context,
        ROCPROFILER_CALLBACK_TRACING_KERNEL_DISPATCH,
        dispatch_ops.data(),
        dispatch_ops.size(),
        dispatch_callback,
        nullptr));

    // Collect async ops via buffers
    constexpr auto buffer_size_bytes = 0x40000;
    constexpr auto buffer_watermark_bytes = buffer_size_bytes / 2;

    ROCPROF_CALL_THROW(rocprofiler_create_buffer(
        globalContext.context,
        buffer_size_bytes,
        buffer_watermark_bytes,
        ROCPROFILER_BUFFER_POLICY_LOSSLESS,
        RocprofLogger::buffer_callback,
        nullptr,
        &globalContext.buffer));

    ROCPROF_CALL_THROW(rocprofiler_configure_buffer_tracing_service(
        globalContext.context,
        ROCPROFILER_BUFFER_TRACING_KERNEL_DISPATCH,
        nullptr,
        0,
        globalContext.buffer));

    ROCPROF_CALL_THROW(rocprofiler_configure_buffer_tracing_service(
        globalContext.context,
        ROCPROFILER_BUFFER_TRACING_MEMORY_COPY,
        nullptr,
        0,
        globalContext.buffer));

    ROCPROF_CALL_THROW(
        rocprofiler_context_is_valid(globalContext.context, &isValid));

    // set the finalize functor once we are certain that we have fully
    // initialized.
    globalContext.finalizer = finalize_func;
    return 0;
  } catch (const std::exception& e) {
    LOG(ERROR) << "RocprofLogger initialization failed: " << e.what();
    // The SDK discards this function's return value, so nothing upstream
    // stops the contexts started here. Stop the utility context before
    // dropping its handle; otherwise its code object callback keeps
    // running for the rest of the process.
    if (globalContext.utilityContext != null_context_id) {
      ROCPROF_CALL(rocprofiler_stop_context(globalContext.utilityContext));
    }
    globalContext.utilityContext = null_context_id;
    globalContext.context = null_context_id;
    globalContext.buffer = null_buffer_id;
    return -1;
  }
}

void RocprofLogger::toolFinalize([[maybe_unused]] void* tool_data) {
  auto& globalContext = getGlobalContext();
  // Nothing to stop here: rocprofiler-sdk stops all client contexts before
  // invoking this functor, and rocprofiler_stop_context on an already-stopped
  // context returns CONTEXT_NOT_FOUND. It does not flush client buffers,
  // however, so this is the final flush and not a redundant one: records
  // still below the buffer's threshold are delivered only by this call.
  // The status is unchecked because nothing is actionable this late in
  // teardown.
  rocprofiler_flush_buffer(globalContext.buffer);
  // Null the ids so later calls fail fast rather than touching handles
  // the SDK is about to destroy.
  globalContext.utilityContext = null_context_id;
  globalContext.context = null_context_id;
  globalContext.buffer = null_buffer_id;

  globalContext.finalizer = nullptr;
  globalContext.clientId = nullptr;
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
  // Intentionally leaky singleton
  static auto* instance = new RocprofLogger{};
  return *instance;
}

RocprofLogger::RocprofLogger() {}

namespace {
// A thread's thread_locals are destroyed when it exits; for the main thread
// that happens inside exit(), before static destructors that can still make
// traced HIP calls (rocBLAS/hipBLASLt unloading code objects while a session
// is open). ~ThreadState sets t_threadExiting before its members are
// destroyed, and the trivially destructible flag stays readable afterwards.
thread_local bool t_threadExiting = false;
struct ThreadState {
  std::deque<uint64_t> externalIds[RocLogger::CorrelationDomain::size];
  // This owner keeps the returned pointer alive while clearLogs() invalidates
  // and releases the registry's copy from another thread.
  std::shared_ptr<ThreadTraceBuffer> buffer;
  ~ThreadState() {
    t_threadExiting = true;
  }
};
thread_local ThreadState t_state;
} // namespace

void RocprofLogger::pushCorrelationID(uint64_t id, CorrelationDomain type) {
  if (t_threadExiting ||
      !singleton().externalCorrelationEnabled_.load(
          std::memory_order_relaxed)) {
    return;
  }
  t_state.externalIds[type].push_back(id);
}

void RocprofLogger::popCorrelationID(CorrelationDomain type) {
  if (t_threadExiting ||
      !singleton().externalCorrelationEnabled_.load(
          std::memory_order_relaxed)) {
    return;
  }
  if (!t_state.externalIds[type].empty()) {
    t_state.externalIds[type].pop_back();
  } else {
    LOG(ERROR)
        << "Attempt to popCorrelationID from an empty external Ids stack";
  }
}

void RocprofLogger::clearLogs() {
  {
    std::lock_guard<std::mutex> lock(threadBuffersMutex_);
    generation_.fetch_add(1, std::memory_order_release);
    for (auto& buffer : threadBuffers_) {
      std::lock_guard<std::mutex> bufferLock(buffer->mutex);
      buffer->rows.clear();
      for (auto& correlations : buffer->externalCorrelations) {
        correlations.clear();
      }
    }
    threadBuffers_.clear();
    totalRows_.store(0, std::memory_order_relaxed);
  }

  {
    std::lock_guard<std::mutex> lock(rowsMutex_);
    rows_.clear();
  }
  {
    std::lock_guard<std::mutex> lock(memsetMutex_);
    memsetOps_.clear();
  }
  {
    std::lock_guard<std::mutex> lock(externalCorrelationsMutex_);
    for (auto& correlations : externalCorrelations_) {
      correlations.clear();
    }
  }
}

ThreadTraceBuffer* RocprofLogger::getThreadBuffer() {
  static RocprofLogger* dis = &singleton();
  auto& buffer = t_state.buffer;

  if (buffer == nullptr ||
      buffer->generation != dis->generation_.load(std::memory_order_acquire)) {
    std::lock_guard<std::mutex> lock(dis->threadBuffersMutex_);
    const auto generation = dis->generation_.load(std::memory_order_relaxed);
    if (buffer == nullptr || buffer->generation != generation) {
      buffer = std::make_shared<ThreadTraceBuffer>(generation);
      dis->threadBuffers_.push_back(buffer);
    }
  }
  return buffer.get();
}

void RocprofLogger::mergeThreadBuffers() {
  std::lock_guard<std::mutex> threadBuffersLock(threadBuffersMutex_);
  if (threadBuffers_.empty()) {
    return;
  }
  generation_.fetch_add(1, std::memory_order_release);
  std::lock_guard<std::mutex> rowsLock(rowsMutex_);
  std::lock_guard<std::mutex> correlationsLock(externalCorrelationsMutex_);
  uint64_t mergedRows = 0;
  for (auto& buffer : threadBuffers_) {
    std::lock_guard<std::mutex> bufferLock(buffer->mutex);
    mergedRows += buffer->rows.size();
    rows_.insert(rows_.end(), buffer->rows.begin(), buffer->rows.end());
    buffer->rows.clear();
    for (int i = 0; i < CorrelationDomain::size; ++i) {
      auto& correlations = buffer->externalCorrelations[i];
      externalCorrelations_[i].insert(
          externalCorrelations_[i].end(),
          correlations.begin(),
          correlations.end());
      correlations.clear();
    }
  }
  totalRows_.fetch_sub(mergedRows, std::memory_order_relaxed);
  threadBuffers_.clear();
}

void RocprofLogger::insert_row_to_buffer(rocprofBase* row) {
  RocprofLogger* dis = &singleton();

  // perThreadBuffers disabled, or this thread's buffer is already destroyed
  // (buffer_callback still delivers records from the final flush in exit())
  if (!dis->perThreadBuffers_.load(std::memory_order_relaxed) ||
      t_threadExiting) {
    std::lock_guard<std::mutex> lock(dis->rowsMutex_);
    if (dis->rows_.size() >= dis->maxBufferSize_) {
      LOG_FIRST_N(WARNING, 10)
          << "Exceeded max GPU buffer count (" << dis->rows_.size() << " > "
          << dis->maxBufferSize_ << ") - terminating tracing";
      return;
    }
    dis->rows_.push_back(row);
    return;
  }

  // perThreadBuffers enabled
  while (true) {
    auto* buffer = getThreadBuffer();
    std::lock_guard<std::mutex> lock(buffer->mutex);
    if (buffer->generation !=
        dis->generation_.load(std::memory_order_acquire)) {
      continue;
    }

    uint64_t rowCount = dis->totalRows_.load(std::memory_order_relaxed);
    while (true) {
      if (rowCount >= dis->maxBufferSize_) {
        LOG_FIRST_N(WARNING, 10)
            << "Exceeded max GPU buffer count (" << rowCount
            << " >= " << dis->maxBufferSize_ << ") - terminating tracing";
        return;
      }
      if (dis->totalRows_.compare_exchange_weak(
              rowCount, rowCount + 1, std::memory_order_relaxed)) {
        break;
      }
    }
    buffer->rows.push_back(row);
    return;
  }
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
      std::lock_guard<std::mutex> lock{globalContext.kernel_lock};
      globalContext.kernel_info.emplace(data->kernel_id, *data);
      globalContext.kernel_names.emplace(
          data->kernel_id, demangle(data->kernel_name));
    }
  }
}

void RocprofLogger::api_callback(
    rocprofiler_callback_tracing_record_t record,
    rocprofiler_user_data_t* user_data,
    [[maybe_unused]] void* callback_data) {
  // Calls made after this thread's thread_locals are gone (see
  // ThreadState) happen during thread or process teardown; drop them.
  if (t_threadExiting) {
    return;
  }
  // The SDK keeps user_data from the ENTER to the EXIT phase of one call, so
  // the start time needs no per-thread map.
  if (record.kind == ROCPROFILER_CALLBACK_TRACING_HIP_RUNTIME_API) {
    if (record.phase == ROCPROFILER_CALLBACK_PHASE_ENTER) {
      user_data->value = getApproximateTime();
    } // ROCPROFILER_CALLBACK_PHASE_ENTER
    else { // ROCPROFILER_CALLBACK_PHASE_EXIT
      uint64_t startTime = user_data->value;
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
      // Memset Records
      // Recorded as plain runtime rows (like other unhandled APIs), but the
      // correlation id and size are kept to classify the fill kernels that
      // the runtime dispatches to implement the memset.
      else if (isMemsetApi(record.operation)) {
        memset_args args;
        rocprofiler_iterate_callback_tracing_kind_operation_args(
            record,
            extract_memset_args,
            1 /*max_deref*/
            ,
            &args);
        RocprofLogger& dis = singleton();
        std::lock_guard<std::mutex> lock(dis.memsetMutex_);
        dis.memsetOps_.emplace(
            record.correlation_id.internal,
            memsetTotalBytes(record.operation, args));
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
      // Store external correlations
      static RocprofLogger* dis = &singleton();

      // perThreadBuffers enabled
      if (dis->perThreadBuffers_.load(std::memory_order_relaxed)) {
        while (true) {
          auto* buffer = getThreadBuffer();
          std::lock_guard<std::mutex> lock(buffer->mutex);
          if (buffer->generation !=
              dis->generation_.load(std::memory_order_acquire)) {
            continue;
          }
          for (int it = RocLogger::CorrelationDomain::begin;
               it < RocLogger::CorrelationDomain::end;
               ++it) {
            if (t_state.externalIds[it].size() > 0) {
              buffer->externalCorrelations[it].emplace_back(
                  record.correlation_id.internal,
                  t_state.externalIds[it].back());
            }
          }
          break;
        }
      } else {
        // perThreadBuffers disabled
        for (int it = RocLogger::CorrelationDomain::begin;
             it < RocLogger::CorrelationDomain::end;
             ++it) {
          if (t_state.externalIds[it].size() > 0) {
            std::lock_guard<std::mutex> lock(dis->externalCorrelationsMutex_);
            dis->externalCorrelations_[it].emplace_back(
                record.correlation_id.internal, t_state.externalIds[it].back());
          }
        }
      }
    } // ROCPROFILER_CALLBACK_PHASE_EXIT
  } // ROCPROFILER_CALLBACK_TRACING_HIP_RUNTIME_API
}

void RocprofLogger::dispatch_callback(
    rocprofiler_callback_tracing_record_t record,
    [[maybe_unused]] rocprofiler_user_data_t* user_data,
    [[maybe_unused]] void* callback_data) {
  // Registered before the packet is submitted, so it always precedes the
  // dispatch's buffer record.
  if (record.kind != ROCPROFILER_CALLBACK_TRACING_KERNEL_DISPATCH ||
      record.operation != ROCPROFILER_KERNEL_DISPATCH_ENQUEUE ||
      record.phase != ROCPROFILER_CALLBACK_PHASE_ENTER) {
    return;
  }
  auto* data =
      static_cast<rocprofiler_callback_tracing_kernel_dispatch_data_t*>(
          record.payload);
  auto& dis = singleton();
  std::lock_guard<std::mutex> lock(dis.pendingDispatchesMutex_);
  dis.pendingDispatches_.insert(data->dispatch_info.dispatch_id);
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

        {
          auto& dis = singleton();
          std::lock_guard<std::mutex> lock(dis.pendingDispatchesMutex_);
          dis.pendingDispatches_.erase(dispatch.dispatch_id);
        }

        // Safe access to agents map with default value
        auto& globalContext = getGlobalContext();
        auto agent_it = globalContext.agents.find(dispatch.agent_id);
        int device_id = (agent_it != globalContext.agents.end())
            ? agent_it->second.logical_node_type_id
            : -1;

        // buffer callback for a given buffer always happens on same thread so
        // the only contention here is with code object callback on another
        // thread.
        auto kernel_name = std::string("<unknown kernel>");
        {
          std::lock_guard<std::mutex> lock{globalContext.kernel_lock};
          auto kernel_it = globalContext.kernel_names.find(dispatch.kernel_id);
          if (kernel_it != globalContext.kernel_names.end()) {
            kernel_name = kernel_it->second;
          }
        }

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
        // Launch dimensions from the dispatch record itself.  This is the
        // ground truth for what executed, and it covers kernels that were
        // never dispatched through a HIP launch API (graph replays, and
        // runtime-internal fill kernels).  The SDK reports grid_size as the
        // global work size (CUPTI reports the number of workgroups), so
        // divide by the workgroup size.
        const auto divdim = [](uint32_t global, uint32_t local) -> uint32_t {
          return local > 0 ? global / local : 0;
        };
        row->gridX = divdim(dispatch.grid_size.x, dispatch.workgroup_size.x);
        row->gridY = divdim(dispatch.grid_size.y, dispatch.workgroup_size.y);
        row->gridZ = divdim(dispatch.grid_size.z, dispatch.workgroup_size.z);
        row->workgroupX = dispatch.workgroup_size.x;
        row->workgroupY = dispatch.workgroup_size.y;
        row->workgroupZ = dispatch.workgroup_size.z;
        insert_row_to_buffer(row);
      } else if (header->kind == ROCPROFILER_BUFFER_TRACING_MEMORY_COPY) {
        auto& record =
            *(static_cast<rocprofiler_buffer_tracing_memory_copy_record_t*>(
                header->payload));

        // Safe access to agents map with default value
        auto& globalContext = getGlobalContext();
        auto agent_it = globalContext.agents.find(record.dst_agent_id);
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
  return fmt::format(
      "<unknown {} operation:{}>", globalContext.name_info[kind].name, op);
}

std::string RocprofLogger::opString(
    rocprofiler_buffer_tracing_kind_t kind,
    rocprofiler_tracing_operation_t op) {
  auto& globalContext = getGlobalContext();
  auto& ops = globalContext.buff_name_info[kind].operations;
  if (op >= 0 && static_cast<size_t>(op) < ops.size()) {
    return std::string(ops[op]);
  }
  return fmt::format(
      "<unknown {} operation:{}>", globalContext.buff_name_info[kind].name, op);
}

void RocprofLogger::setMaxEvents(uint32_t maxBufferSize) {
  RocprofLogger* dis = &singleton();
  std::lock_guard<std::mutex> lock(dis->rowsMutex_);
  maxBufferSize_ = maxBufferSize;
}

void RocprofLogger::setPerThreadBuffers(bool enabled) {
  perThreadBuffers_.store(enabled, std::memory_order_relaxed);
}

void RocprofLogger::setDispatchRecordWait(std::chrono::milliseconds wait) {
  dispatchRecordWait_ = wait;
}

void RocprofLogger::ensureRegistered() {
  int status = 0;
  rocprofiler_is_initialized(&status);
  VLOG(0) << "rocprofiler_is_initialized returned " << status;
  if (status == 0) {
    VLOG(0) << "Forcing rocprofiler-sdk tool registration";
    if (ROCPROF_CALL(rocprofiler_force_configure(&rocprofiler_configure)) ==
        ROCPROFILER_STATUS_SUCCESS) {
      VLOG(0) << "rocprofiler-sdk tool registration completed successfully";
    }
  }
}

void RocprofLogger::startLogging() {
  ensureRegistered();

  auto& globalContext = getGlobalContext();

  if (globalContext.context == null_context_id) {
    LOG(ERROR) << "Rocprofiler-SDK context is null. Cannot start logging.";
    return;
  }

  externalCorrelationEnabled_.store(true, std::memory_order_relaxed);
  {
    // Dispatches given up on by an earlier stop must not stall this session.
    std::lock_guard<std::mutex> lock(pendingDispatchesMutex_);
    pendingDispatches_.clear();
  }
  // start_context is idempotent (it returns SUCCESS for an already-active
  // context), so there is no need to query context state first.
  ROCPROF_CALL(rocprofiler_start_context(globalContext.context));
}

void RocprofLogger::stopLogging() {
  auto& globalContext = getGlobalContext();

  // A null context here is an expected state: CPU-only profiling in a ROCm
  // build, or after SDK finalization. Silent no-op; the actionable diagnostic
  // for a null context is in startLogging().
  if (globalContext.context == null_context_id) {
    return;
  }

  externalCorrelationEnabled_.store(false, std::memory_order_relaxed);
  // stop_context returns CONTEXT_NOT_FOUND when the context is not active,
  // which is an expected state (e.g. stopping twice) rather than an error.
  auto status = rocprofiler_stop_context(globalContext.context);
  if (status != ROCPROFILER_STATUS_SUCCESS &&
      status != ROCPROFILER_STATUS_ERROR_CONTEXT_NOT_FOUND) {
    LOG(WARNING) << fmt::format(
        "rocprofiler_stop_context failed: {} ({})",
        rocprofStatusString(status),
        static_cast<int>(status));
  }
  // Stopping does not wait for in-flight records: a kernel dispatch is bound
  // to the contexts active at enqueue, and its record is still written to
  // this buffer when the kernel completes after the stop.
  ROCPROF_CALL(rocprofiler_flush_buffer(globalContext.buffer));

  // The SDK writes a dispatch record from its own completion handler, which
  // can run after the application has already seen the kernel finish (e.g.
  // hipDeviceSynchronize returned). Such a record would miss the flush above
  // and be dropped as out of range in the next session, so re-flush until
  // every dispatch of this session has its record. The wait is bounded because
  // some dispatches never produce a usable one: a dispatch the SDK could not
  // time has no record, and a kernel still running at stop ends outside this
  // session's window, so its record is dropped even if the wait sees it.
  auto pendingCount = [this] {
    std::lock_guard<std::mutex> lock(pendingDispatchesMutex_);
    return pendingDispatches_.size();
  };
  const auto deadline = std::chrono::steady_clock::now() + dispatchRecordWait_;
  size_t pending = pendingCount();
  while (pending > 0 && std::chrono::steady_clock::now() < deadline) {
    std::this_thread::sleep_for(std::chrono::microseconds(50));
    ROCPROF_CALL(rocprofiler_flush_buffer(globalContext.buffer));
    pending = pendingCount();
  }
  if (pending > 0 && dispatchRecordWait_.count() > 0) {
    LOG_FIRST_N(WARNING, 10)
        << "Stopped waiting for " << pending
        << " kernel dispatch record(s) after " << dispatchRecordWait_.count()
        << " ms; their GPU events are missing from this trace. Kernels still "
        << "running when profiling stopped cannot be recorded; for kernels "
        << "that had finished, raise ROCPROF_DISPATCH_RECORD_WAIT_MSECS.";
  }
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
