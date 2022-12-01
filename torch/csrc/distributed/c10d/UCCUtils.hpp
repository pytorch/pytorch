#pragma once

#ifdef USE_C10D_UCC

#include <torch/csrc/distributed/c10d/ProcessGroup.hpp>
#include <torch/csrc/distributed/c10d/Store.hpp>
#include <ucc/api/ucc.h>

namespace c10d {

// Macro to generate the error message on a non-successful UCC return value.
#define TORCH_UCC_GET_ERROR_MSG(_err, _error_msg, _result) \
  do {                                                     \
      _err = c10::str(                                     \
          "[",                                             \
          std::string(__FILE__),                           \
          ":",                                             \
          std::to_string(__LINE__),                        \
          "] ",                                            \
          logger->getLogPrefix(),                          \
          _error_msg,                                      \
          ", error code ",                                 \
          _result,                                         \
          ": ",                                            \
          ucc_status_string(_result),                      \
          ", system error code ",                          \
          errno);                                          \
  } while (0)

// Macro to throw on a non-successful UCC return value.
#define TORCH_UCC_CHECK(_cmd, _error_msg)               \
  do {                                                  \
    ucc_status_t result = _cmd;                         \
    if (result != UCC_OK) {                             \
      std::string err;                                  \
      TORCH_UCC_GET_ERROR_MSG(err, _error_msg, result); \
      TORCH_CHECK(false, err);                          \
    }                                                   \
  } while (0)

// Macro and throw on a non-successful UCC return value and free its request.
#define TORCH_UCC_CHECK_REQUEST(_request, _cmd, _error_msg) \
  do {                                                      \
    ucc_status_t result = _cmd;                             \
    if (result != UCC_OK) {                                 \
      std::string err;                                      \
      TORCH_UCC_GET_ERROR_MSG(err, _error_msg, result);     \
      if (_request != nullptr) {                            \
        ucc_collective_finalize(_request);                  \
      }                                                     \
      TORCH_CHECK(false, err);                              \
    }                                                       \
  } while (0)

// Macros to print logs with unified format
#define TORCH_UCC_LOG_ERROR(_phase, _msg) \
  LOG(ERROR) << logger->getLogPrefix(_phase) << "[ERROR] " << _msg;
#define TORCH_UCC_LOG_INFO(_phase, _msg) \
  LOG(INFO) << logger->getLogPrefix(_phase) << "[INFO] " << _msg;
#define TORCH_UCC_LOG_DEBUG(_phase, _msg) \
  VLOG(1) << logger->getLogPrefix(_phase) << "[DEBUG] " << _msg;

enum torch_ucc_phase_t {
  TORCH_UCC_UNKNOWN = -1,
  TORCH_UCC_INIT,
  TORCH_UCC_HEALTH_CHECK,
  TORCH_UCC_READY,
  TORCH_UCC_COLL_POST,
  TORCH_UCC_COLL_PROGRESS,
  TORCH_UCC_FINALIZE,
};

const std::map<torch_ucc_phase_t, std::string> ucc_phase_map = {
    {TORCH_UCC_UNKNOWN, "UNKNOWN"},
    {TORCH_UCC_INIT, "INIT"},
    {TORCH_UCC_HEALTH_CHECK, "HEALTH_CHECK"},
    {TORCH_UCC_READY, "READY"},
    {TORCH_UCC_COLL_POST, "COLL_POST"},
    {TORCH_UCC_COLL_PROGRESS, "COLL_PROGRESS"},
    {TORCH_UCC_FINALIZE, "FINALIZE"},
};

class CommTraceLogger;

class TORCH_API ProcessGroupUCCLogger : public torch::CustomClassHolder {
 public:
  ProcessGroupUCCLogger();
  ProcessGroupUCCLogger(std::string log_prefix, torch_ucc_phase_t phase);

  std::string getLogPrefix(torch_ucc_phase_t phase = TORCH_UCC_UNKNOWN);
  void setLogPrefix(std::string log_prefix);
  inline void setPhase(torch_ucc_phase_t phase) {
    local_phase = phase;
  }

  void initCommsTracer();
  void flushComms(int rank, int world_size);
  std::shared_ptr<CommTraceLogger> trace_generator = nullptr;

 protected:
  std::string log_prefix;
  torch_ucc_phase_t local_phase = TORCH_UCC_UNKNOWN;
  bool initialized_CommTraceLogger = false;
};

struct torch_ucc_oob_coll_info_t {
  c10::intrusive_ptr<Store> store;
  uint32_t comm_id;
  int rank;
  int size;
  void* rbuf;
  size_t msglen;
  std::string getKey(std::string key) {
    return std::to_string(comm_id) + key;
  }
};

class CommBase {
 public:
  CommBase(const c10::intrusive_ptr<ProcessGroupUCCLogger>& logger_)
      : logger(logger_) {}
  virtual void progress() = 0;
  virtual void free_request(ucc_coll_req_h request) = 0;
  virtual ~CommBase() {}
  c10::intrusive_ptr<ProcessGroupUCCLogger> logger;
};
class CommUCC : public CommBase {
 public:
  ucc_lib_h lib{nullptr};
  ucc_context_h context{nullptr};

 public:
  void progress() override;
  CommUCC(
      std::shared_ptr<torch_ucc_oob_coll_info_t> oob,
      const c10::intrusive_ptr<ProcessGroupUCCLogger>& logger);
  void free_request(ucc_coll_req_h request) override;
  ~CommUCC();
};

ucc_status_t oob_allgather(
    void* sbuf,
    void* rbuf,
    size_t msglen,
    void* coll_info,
    void** req);

ucc_status_t oob_allgather_test(void* req);

ucc_status_t oob_allgather_free(void* req);

// trim: remove spaces before and after the string view
// implementation borrowed from https://stackoverflow.com/a/17976541
inline c10::string_view trim(c10::string_view s) {
  auto wsfront = std::find_if_not(
      s.begin(), s.end(), [](int c) { return std::isspace(c); });
  auto wsback = std::find_if_not(s.rbegin(), s.rend(), [](int c) {
                  return std::isspace(c);
                }).base();
  return (
      wsback <= wsfront ? "" : s.substr(wsfront - s.begin(), wsback - wsfront));
}

inline std::string tolower(c10::string_view s) {
  std::string result;
  result.reserve(s.size());
  for (auto c : s) {
    result.push_back(std::tolower(c));
  }
  return result;
}

inline std::vector<std::string> parse_list(std::string list) {
  std::vector<std::string> result;
  list = tolower(trim(list));
  while (!list.empty()) {
    const auto end_pos = list.find_first_of(',');
    const auto token = trim(list.substr(0, end_pos));
    result.push_back(std::string(token));
    list = (end_pos != c10::string_view::npos) ? list.substr(end_pos + 1) : "";
  }
  return result;
}

} // namespace c10d

#endif // USE_C10D_UCC
