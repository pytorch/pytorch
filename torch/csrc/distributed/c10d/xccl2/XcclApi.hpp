#pragma once

#ifdef USE_C10D_XCCL

#include <mutex>
#include <string_view>

#include <oneapi/ccl.h>

#include <c10/xpu/XPUStream.h>

namespace c10d::xccl2 {

/**
 * Abstract interface for oneCCL API operations.
 * This allows for dependency injection and testing by providing
 * a way to override oneCCL API calls.
 *
 * This binds the oneCCL *C* API (`onecclXxx`, <oneapi/ccl.h>) rather than the
 * C++ API (`ccl::`, <oneapi/ccl.hpp>) used by the older in-tree
 * `c10d::ProcessGroupXCCL`: the C API is the one that mirrors NCCL's shape, so
 * the backend built on top of it can stay structurally aligned with nccl2.
 *
 * Streams are passed as `c10::xpu::XPUStream`; the oneCCL C API takes an opaque
 * `void*` that it expects to be a `sycl::queue*`, which XPUStream implicitly
 * converts to.
 */
class XcclApi {
 private:
  struct VersionInfo {
    int version = -1;
    // Breakdown of version into major, minor, patch
    int major = -1;
    int minor = -1;
    int patch = -1;
  };

 protected:
  VersionInfo version_info_;

 public:
  virtual ~XcclApi() = default;

  // Error handling
  virtual std::string_view getErrorString(onecclResult_t result) = 0;

  // Device management
  [[nodiscard]] virtual onecclResult_t setDevice(int device) = 0;

  // Unique ID generation
  [[nodiscard]] virtual onecclResult_t getUniqueId(
      onecclUniqueId* uniqueId) = 0;

  // Communicator management
  [[nodiscard]] virtual onecclResult_t commInitRankConfig(
      onecclComm_t* comm,
      int nranks,
      onecclUniqueId commId,
      int rank,
      onecclConfig_t* config) = 0;

  [[nodiscard]] virtual onecclResult_t commDestroy(onecclComm_t comm) = 0;

  // Not implemented by oneCCL yet -- see the note in XcclApi.cpp. Kept in the
  // interface so callers (and the mock in tests) have a stable surface, and so
  // that wiring them up later is a one-line change.
  [[nodiscard]] virtual onecclResult_t commAbort(onecclComm_t comm) = 0;

  [[nodiscard]] virtual onecclResult_t commGetAsyncError(
      onecclComm_t comm,
      onecclResult_t* asyncError) = 0;

  [[nodiscard]] virtual onecclResult_t commSplit(
      onecclComm_t comm,
      int color,
      int key,
      onecclComm_t* newcomm,
      onecclConfig_t* config) = 0;

  // Memory registration
  [[nodiscard]] virtual onecclResult_t commRegister(
      onecclComm_t comm,
      void* buffer,
      size_t size,
      void** handle) = 0;

  [[nodiscard]] virtual onecclResult_t commDeregister(
      onecclComm_t comm,
      void* handle) = 0;

  // Point-to-point operations
  [[nodiscard]] virtual onecclResult_t send(
      const void* sendbuff,
      size_t count,
      onecclDataType_t datatype,
      int peer,
      onecclComm_t comm,
      c10::xpu::XPUStream stream) = 0;

  [[nodiscard]] virtual onecclResult_t recv(
      void* recvbuff,
      size_t count,
      onecclDataType_t datatype,
      int peer,
      onecclComm_t comm,
      c10::xpu::XPUStream stream) = 0;

  // Collective operations
  [[nodiscard]] virtual onecclResult_t broadcast(
      const void* sendbuff,
      void* recvbuff,
      size_t count,
      onecclDataType_t datatype,
      int root,
      onecclComm_t comm,
      c10::xpu::XPUStream stream) = 0;

  [[nodiscard]] virtual onecclResult_t bcast(
      void* buff,
      size_t count,
      onecclDataType_t datatype,
      int root,
      onecclComm_t comm,
      c10::xpu::XPUStream stream) = 0;

  [[nodiscard]] virtual onecclResult_t allReduce(
      const void* sendbuff,
      void* recvbuff,
      size_t count,
      onecclDataType_t datatype,
      onecclRedOp_t op,
      onecclComm_t comm,
      c10::xpu::XPUStream stream) = 0;

  [[nodiscard]] virtual onecclResult_t reduce(
      const void* sendbuff,
      void* recvbuff,
      size_t count,
      onecclDataType_t datatype,
      onecclRedOp_t op,
      int root,
      onecclComm_t comm,
      c10::xpu::XPUStream stream) = 0;

  [[nodiscard]] virtual onecclResult_t allGather(
      const void* sendbuff,
      void* recvbuff,
      size_t sendcount,
      onecclDataType_t datatype,
      onecclComm_t comm,
      c10::xpu::XPUStream stream) = 0;

  [[nodiscard]] virtual onecclResult_t reduceScatter(
      const void* sendbuff,
      void* recvbuff,
      size_t recvcount,
      onecclDataType_t datatype,
      onecclRedOp_t op,
      onecclComm_t comm,
      c10::xpu::XPUStream stream) = 0;

  [[nodiscard]] virtual onecclResult_t allToAll(
      const void* sendbuff,
      void* recvbuff,
      size_t count,
      onecclDataType_t datatype,
      onecclComm_t comm,
      c10::xpu::XPUStream stream) = 0;

  // Group operations
  [[nodiscard]] virtual onecclResult_t groupStart() = 0;
  [[nodiscard]] virtual onecclResult_t groupEnd() = 0;

  // Communicator introspection
  [[nodiscard]] virtual onecclResult_t commUserRank(
      const onecclComm_t comm,
      int* userRank) = 0;
  [[nodiscard]] virtual onecclResult_t commCount(
      const onecclComm_t comm,
      int* count) = 0;

  // Custom reduction operations
  [[nodiscard]] virtual onecclResult_t redOpCreatePreMulSum(
      onecclRedOp_t* op,
      void* scalar,
      onecclDataType_t datatype,
      onecclScalarResidence_t residence,
      onecclComm_t comm) = 0;
  [[nodiscard]] virtual onecclResult_t redOpDestroy(
      onecclRedOp_t op,
      onecclComm_t comm) = 0;

  // Allocator
  [[nodiscard]] virtual onecclResult_t memAlloc(void** buff, size_t size) = 0;
  [[nodiscard]] virtual onecclResult_t memFree(void* buff) = 0;

  virtual void setVersionInfo() = 0;

  [[nodiscard]] int getVersion() const {
    return version_info_.version;
  }

  [[nodiscard]] int getMajorVersion() const {
    return version_info_.major;
  }

  [[nodiscard]] int getMinorVersion() const {
    return version_info_.minor;
  }

  [[nodiscard]] int getPatchVersion() const {
    return version_info_.patch;
  }
};

/**
 * Default implementation that calls the underlying oneCCL APIs directly.
 */
class DefaultXcclApi : public XcclApi {
 public:
  ~DefaultXcclApi() override = default;

  // Error handling
  std::string_view getErrorString(onecclResult_t result) override;

  // Device management
  [[nodiscard]] onecclResult_t setDevice(int device) override;

  // Unique ID generation
  [[nodiscard]] onecclResult_t getUniqueId(onecclUniqueId* uniqueId) override;

  // Communicator management
  [[nodiscard]] onecclResult_t commInitRankConfig(
      onecclComm_t* comm,
      int nranks,
      onecclUniqueId commId,
      int rank,
      onecclConfig_t* config) override;

  [[nodiscard]] onecclResult_t commDestroy(onecclComm_t comm) override;

  [[nodiscard]] onecclResult_t commAbort(onecclComm_t comm) override;

  [[nodiscard]] onecclResult_t commGetAsyncError(
      onecclComm_t comm,
      onecclResult_t* asyncError) override;

  [[nodiscard]] onecclResult_t commSplit(
      onecclComm_t comm,
      int color,
      int key,
      onecclComm_t* newcomm,
      onecclConfig_t* config) override;

  // Memory registration
  [[nodiscard]] onecclResult_t commRegister(
      onecclComm_t comm,
      void* buffer,
      size_t size,
      void** handle) override;

  [[nodiscard]] onecclResult_t commDeregister(onecclComm_t comm, void* handle)
      override;

  // Point-to-point operations
  [[nodiscard]] onecclResult_t send(
      const void* sendbuff,
      size_t count,
      onecclDataType_t datatype,
      int peer,
      onecclComm_t comm,
      c10::xpu::XPUStream stream) override;

  [[nodiscard]] onecclResult_t recv(
      void* recvbuff,
      size_t count,
      onecclDataType_t datatype,
      int peer,
      onecclComm_t comm,
      c10::xpu::XPUStream stream) override;

  // Collective operations
  [[nodiscard]] onecclResult_t broadcast(
      const void* sendbuff,
      void* recvbuff,
      size_t count,
      onecclDataType_t datatype,
      int root,
      onecclComm_t comm,
      c10::xpu::XPUStream stream) override;

  [[nodiscard]] onecclResult_t bcast(
      void* buff,
      size_t count,
      onecclDataType_t datatype,
      int root,
      onecclComm_t comm,
      c10::xpu::XPUStream stream) override;

  [[nodiscard]] onecclResult_t allReduce(
      const void* sendbuff,
      void* recvbuff,
      size_t count,
      onecclDataType_t datatype,
      onecclRedOp_t op,
      onecclComm_t comm,
      c10::xpu::XPUStream stream) override;

  [[nodiscard]] onecclResult_t reduce(
      const void* sendbuff,
      void* recvbuff,
      size_t count,
      onecclDataType_t datatype,
      onecclRedOp_t op,
      int root,
      onecclComm_t comm,
      c10::xpu::XPUStream stream) override;

  [[nodiscard]] onecclResult_t allGather(
      const void* sendbuff,
      void* recvbuff,
      size_t sendcount,
      onecclDataType_t datatype,
      onecclComm_t comm,
      c10::xpu::XPUStream stream) override;

  [[nodiscard]] onecclResult_t reduceScatter(
      const void* sendbuff,
      void* recvbuff,
      size_t recvcount,
      onecclDataType_t datatype,
      onecclRedOp_t op,
      onecclComm_t comm,
      c10::xpu::XPUStream stream) override;

  [[nodiscard]] onecclResult_t allToAll(
      const void* sendbuff,
      void* recvbuff,
      size_t count,
      onecclDataType_t datatype,
      onecclComm_t comm,
      c10::xpu::XPUStream stream) override;

  // Group operations
  [[nodiscard]] onecclResult_t groupStart() override;
  [[nodiscard]] onecclResult_t groupEnd() override;

  // Communicator introspection
  [[nodiscard]] onecclResult_t commUserRank(
      const onecclComm_t comm,
      int* userRank) override;
  [[nodiscard]] onecclResult_t commCount(const onecclComm_t comm, int* count)
      override;

  // Custom reduction operations
  [[nodiscard]] onecclResult_t redOpCreatePreMulSum(
      onecclRedOp_t* op,
      void* scalar,
      onecclDataType_t datatype,
      onecclScalarResidence_t residence,
      onecclComm_t comm) override;
  [[nodiscard]] onecclResult_t redOpDestroy(onecclRedOp_t op, onecclComm_t comm)
      override;

  // Allocator
  [[nodiscard]] onecclResult_t memAlloc(void** buff, size_t size) override;
  [[nodiscard]] onecclResult_t memFree(void* buff) override;

  void setVersionInfo() override;

 private:
  mutable std::mutex api_mutex_;
};

} // namespace c10d::xccl2

#endif // USE_C10D_XCCL
