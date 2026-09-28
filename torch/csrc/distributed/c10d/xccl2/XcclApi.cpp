#ifdef USE_C10D_XCCL

#include <torch/csrc/distributed/c10d/xccl2/Logging.hpp>
#include <torch/csrc/distributed/c10d/xccl2/XcclApi.hpp>

namespace c10d::xccl2 {

// DefaultXcclApi implementation
std::string_view DefaultXcclApi::getErrorString(onecclResult_t result) {
  std::lock_guard<std::mutex> lock(api_mutex_);
  const char* error_string = onecclGetErrorString(result);
  // Unlike ncclGetErrorString, oneCCL makes no documented guarantee that this
  // is non-null, and std::string_view(nullptr) is UB.
  return error_string ? std::string_view(error_string) : std::string_view{};
}

onecclResult_t DefaultXcclApi::setDevice(int device) {
  std::lock_guard<std::mutex> lock(api_mutex_);
  return onecclSetDevice(device);
}

onecclResult_t DefaultXcclApi::getUniqueId(onecclUniqueId* uniqueId) {
  std::lock_guard<std::mutex> lock(api_mutex_);
  return onecclGetUniqueId(uniqueId);
}

onecclResult_t DefaultXcclApi::commInitRankConfig(
    onecclComm_t* comm,
    int nranks,
    onecclUniqueId commId,
    int rank,
    onecclConfig_t* config) {
  std::lock_guard<std::mutex> lock(api_mutex_);
  return onecclCommInitRankConfig(comm, nranks, commId, rank, config);
}

onecclResult_t DefaultXcclApi::commDestroy(onecclComm_t comm) {
  std::lock_guard<std::mutex> lock(api_mutex_);
  return onecclCommDestroy(comm);
}

// onecclCommAbort and onecclCommGetAsyncError are declared
// CCL_C_NOT_IMPLEMENTED in <oneapi/ccl.h>, which expands to
// __attribute__((error(...))) -- i.e. referencing them is a *compile-time*
// error, not a runtime failure. So these cannot simply forward, and the
// backend must not rely on abort-based teardown or async error polling until
// oneCCL implements them. Uncomment the calls (and drop the [[maybe_unused]])
// once it does.
onecclResult_t DefaultXcclApi::commAbort([[maybe_unused]] onecclComm_t comm) {
  // return onecclCommAbort(comm);
  return onecclNotImplemented;
}

onecclResult_t DefaultXcclApi::commGetAsyncError(
    [[maybe_unused]] onecclComm_t comm,
    [[maybe_unused]] onecclResult_t* asyncError) {
  // return onecclCommGetAsyncError(comm, asyncError);
  return onecclNotImplemented;
}

onecclResult_t DefaultXcclApi::commSplit(
    onecclComm_t comm,
    int color,
    int key,
    onecclComm_t* newcomm,
    onecclConfig_t* config) {
  std::lock_guard<std::mutex> lock(api_mutex_);
  return onecclCommSplit(comm, color, key, newcomm, config);
}

onecclResult_t DefaultXcclApi::commRegister(
    onecclComm_t comm,
    void* buffer,
    size_t size,
    void** handle) {
  std::lock_guard<std::mutex> lock(api_mutex_);
  return onecclCommRegister(comm, buffer, size, handle);
}

onecclResult_t DefaultXcclApi::commDeregister(onecclComm_t comm, void* handle) {
  std::lock_guard<std::mutex> lock(api_mutex_);
  return onecclCommDeregister(comm, handle);
}

onecclResult_t DefaultXcclApi::send(
    const void* sendbuff,
    size_t count,
    onecclDataType_t datatype,
    int peer,
    onecclComm_t comm,
    c10::xpu::XPUStream stream) {
  std::lock_guard<std::mutex> lock(api_mutex_);
  return onecclSend(sendbuff, count, datatype, peer, comm, stream);
}

onecclResult_t DefaultXcclApi::recv(
    void* recvbuff,
    size_t count,
    onecclDataType_t datatype,
    int peer,
    onecclComm_t comm,
    c10::xpu::XPUStream stream) {
  std::lock_guard<std::mutex> lock(api_mutex_);
  return onecclRecv(recvbuff, count, datatype, peer, comm, stream);
}

onecclResult_t DefaultXcclApi::broadcast(
    const void* sendbuff,
    void* recvbuff,
    size_t count,
    onecclDataType_t datatype,
    int root,
    onecclComm_t comm,
    c10::xpu::XPUStream stream) {
  std::lock_guard<std::mutex> lock(api_mutex_);
  return onecclBroadcast(
      const_cast<void*>(sendbuff),
      recvbuff,
      count,
      datatype,
      root,
      comm,
      stream);
}

onecclResult_t DefaultXcclApi::bcast(
    void* buff,
    size_t count,
    onecclDataType_t datatype,
    int root,
    onecclComm_t comm,
    c10::xpu::XPUStream stream) {
  std::lock_guard<std::mutex> lock(api_mutex_);
  return onecclBroadcast(buff, buff, count, datatype, root, comm, stream);
}

onecclResult_t DefaultXcclApi::allReduce(
    const void* sendbuff,
    void* recvbuff,
    size_t count,
    onecclDataType_t datatype,
    onecclRedOp_t op,
    onecclComm_t comm,
    c10::xpu::XPUStream stream) {
  std::lock_guard<std::mutex> lock(api_mutex_);
  return onecclAllReduce(
      const_cast<void*>(sendbuff), recvbuff, count, datatype, op, comm, stream);
}

onecclResult_t DefaultXcclApi::reduce(
    const void* sendbuff,
    void* recvbuff,
    size_t count,
    onecclDataType_t datatype,
    onecclRedOp_t op,
    int root,
    onecclComm_t comm,
    c10::xpu::XPUStream stream) {
  std::lock_guard<std::mutex> lock(api_mutex_);
  return onecclReduce(
      const_cast<void*>(sendbuff),
      recvbuff,
      count,
      datatype,
      op,
      root,
      comm,
      stream);
}

onecclResult_t DefaultXcclApi::allGather(
    const void* sendbuff,
    void* recvbuff,
    size_t sendcount,
    onecclDataType_t datatype,
    onecclComm_t comm,
    c10::xpu::XPUStream stream) {
  std::lock_guard<std::mutex> lock(api_mutex_);
  return onecclAllGather(
      const_cast<void*>(sendbuff), recvbuff, sendcount, datatype, comm, stream);
}

onecclResult_t DefaultXcclApi::reduceScatter(
    const void* sendbuff,
    void* recvbuff,
    size_t recvcount,
    onecclDataType_t datatype,
    onecclRedOp_t op,
    onecclComm_t comm,
    c10::xpu::XPUStream stream) {
  std::lock_guard<std::mutex> lock(api_mutex_);
  return onecclReduceScatter(
      const_cast<void*>(sendbuff),
      recvbuff,
      recvcount,
      datatype,
      op,
      comm,
      stream);
}

onecclResult_t DefaultXcclApi::allToAll(
    const void* sendbuff,
    void* recvbuff,
    size_t count,
    onecclDataType_t datatype,
    onecclComm_t comm,
    c10::xpu::XPUStream stream) {
  std::lock_guard<std::mutex> lock(api_mutex_);
  return onecclAllToAll(
      const_cast<void*>(sendbuff), recvbuff, count, datatype, comm, stream);
}

onecclResult_t DefaultXcclApi::groupStart() {
  std::lock_guard<std::mutex> lock(api_mutex_);
  return onecclGroupStart();
}

onecclResult_t DefaultXcclApi::groupEnd() {
  std::lock_guard<std::mutex> lock(api_mutex_);
  return onecclGroupEnd();
}

onecclResult_t DefaultXcclApi::commUserRank(
    const onecclComm_t comm,
    int* userRank) {
  std::lock_guard<std::mutex> lock(api_mutex_);
  return onecclCommUserRank(comm, userRank);
}

onecclResult_t DefaultXcclApi::commCount(const onecclComm_t comm, int* count) {
  std::lock_guard<std::mutex> lock(api_mutex_);
  return onecclCommCount(comm, count);
}

onecclResult_t DefaultXcclApi::redOpCreatePreMulSum(
    onecclRedOp_t* op,
    void* scalar,
    onecclDataType_t datatype,
    onecclScalarResidence_t residence,
    onecclComm_t comm) {
  std::lock_guard<std::mutex> lock(api_mutex_);
  return onecclRedOpCreatePreMulSum(op, scalar, datatype, residence, comm);
}

onecclResult_t DefaultXcclApi::redOpDestroy(
    onecclRedOp_t op,
    onecclComm_t comm) {
  std::lock_guard<std::mutex> lock(api_mutex_);
  return onecclRedOpDestroy(op, comm);
}

onecclResult_t DefaultXcclApi::memAlloc(void** buff, size_t size) {
  std::lock_guard<std::mutex> lock(api_mutex_);
  return onecclMemAlloc(buff, size);
}

onecclResult_t DefaultXcclApi::memFree(void* buff) {
  std::lock_guard<std::mutex> lock(api_mutex_);
  return onecclMemFree(buff);
}

void DefaultXcclApi::setVersionInfo() {
  int version = -1;
  onecclResult_t res = onecclGetVersion(&version);

  if (res != onecclSuccess) {
    TC_LOG(ERROR) << "XCCL getVersion failed with error: "
                  << getErrorString(res);
    return;
  }

  version_info_.version = version;

  int major = -1;
  int minor = -1;
  int patch = -1;
  res = onecclExtractVersionComponents(version, &major, &minor, &patch);

  if (res != onecclSuccess) {
    TC_LOG(ERROR) << "XCCL extractVersionComponents failed with error: "
                  << getErrorString(res);
    TC_LOG(WARNING) << "XCCL Major/Minor/Patch info not available";
    return;
  }

  version_info_.major = major;
  version_info_.minor = minor;
  version_info_.patch = patch;
}

} // namespace c10d::xccl2

#endif // USE_C10D_XCCL
