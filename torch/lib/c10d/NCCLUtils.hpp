#pragma once

#include <memory>

#include <nccl.h>

#define C10D_NCCL_CHECK(cmd)                                              \
  do {                                                                    \
    ncclResult_t error = cmd;                                             \
    if (error != ncclSuccess) {                                           \
      std::string err = "NCCL error in: " + std::string(__FILE__) + ":" + \
          std::to_string(__LINE__) + ", " +                               \
          std::string(ncclGetErrorString(error));                         \
      throw std::runtime_error(err);                                      \
    }                                                                     \
  } while (0)

namespace c10d {

// RAII wrapper for NCCL communicator
class NCCLComm {
 public:
  explicit NCCLComm(ncclComm_t ncclComm) : ncclComm_(ncclComm) {}

  NCCLComm() : NCCLComm(nullptr) {}

  ~NCCLComm() noexcept(false) {
    if (ncclComm_) {
      C10D_NCCL_CHECK(ncclCommDestroy(ncclComm_));
    }
  }

  static std::shared_ptr<NCCLComm> create(
      int numRanks,
      int rank,
      ncclUniqueId commId) {
    auto comm = std::make_shared<NCCLComm>();
    C10D_NCCL_CHECK(
        ncclCommInitRank(&(comm->ncclComm_), numRanks, commId, rank));
    return comm;
  }

  // Must not be copyable
  NCCLComm(const NCCLComm&) = delete;
  NCCLComm& operator=(const NCCLComm&) = delete;

  // Move constructable
  NCCLComm(NCCLComm&& other) {
    std::swap(ncclComm_, other.ncclComm_);
  }
  // Move assignable
  NCCLComm& operator=(NCCLComm&& other) {
    std::swap(ncclComm_, other.ncclComm_);
    return *this;
  }

  ncclComm_t getNcclComm() {
    return ncclComm_;
  }

 protected:
  ncclComm_t ncclComm_;
};

} // namespace c10d
