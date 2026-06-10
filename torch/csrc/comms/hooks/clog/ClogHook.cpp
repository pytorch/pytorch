// Copyright (c) Meta Platforms, Inc. and affiliates.

#include <torch/csrc/comms/TorchComm.hpp>
#include <torch/csrc/comms/hooks/clog/ClogHook.hpp>
#include <torch/csrc/comms/hooks/common/OpNameHelper.hpp>

#include <algorithm>
#include <cassert>
#include <chrono>
#include <stdexcept>
#include <variant>

#include <fmt/core.h>

#if __has_include(<cuda_runtime_api.h>) && __has_include(<ATen/cuda/CUDAContext.h>)
#include <ATen/cuda/CUDAContext.h>
#define CLOG_HAS_CUDA 1
#endif

namespace torch::comms {

namespace {
constexpr int kLogVersion = 1;

bool getAsyncOp(const PreHookArgs& args) {
  return std::visit(
      [](const auto& a) -> bool {
        using T = std::decay_t<decltype(a)>;
        if constexpr (
            std::is_same_v<T, SplitPreHookArgs> ||
            std::is_same_v<T, NewWindowPreHookArgs> ||
            std::is_same_v<T, FinalizePreHookArgs>) {
          return false;
        } else {
          return a.async_op;
        }
      },
      args);
}
} // namespace

ClogHook::ClogHook(
    const std::string& output,
    const std::vector<std::string>& events,
    const std::vector<std::string>& verbose) {
  // Parse events
  for (const auto& ev : events) {
    if (ev == "ALL" || ev == "LIFECYCLE") {
      log_lifecycle_ = true;
    }
  }

  // Parse verbose options
  for (const auto& v : verbose) {
    if (v == "buffers") {
      log_buffers_ = true;
    }
  }

  log_file_.open(output);

  // Write version header with base timestamp
  base_ts_ = now();
  log_file_.writeLine(
      fmt::format("V|{}|base_timestamp={:.3f}", kLogVersion, base_ts_));
}

ClogHook::~ClogHook() = default;

// -- Registration --

void ClogHook::registerWithComm(std::shared_ptr<TorchComm> comm) {
  log_file_.writeLine(buildNewCommSignature(
      comm->getCommName(), comm->getRank(), comm->getSize()));
  registerHooks(comm);
}

void ClogHook::registerHooks(std::shared_ptr<TorchComm> comm) {
  for (const auto& reg : registrations_) {
    if (reg.comm.lock() == comm) {
      throw std::runtime_error(
          "ClogHook: already registered with comm " +
          std::string(comm->getCommName()));
    }
  }

  std::string comm_name(comm->getCommName());
  auto self = shared_from_this();

  int device_index = comm->getDevice().index();
  auto pre_hook_handle = comm->registerPreHook(
      [self, comm_name, device_index](size_t op_id, const PreHookArgs& args) {
        self->onPreHook(comm_name, device_index, op_id, args);
      });

  auto post_hook_handle = comm->registerPostHook(
      [self, comm_name](size_t op_id, const PostHookArgs& args) {
        self->onPostHook(comm_name, op_id, args);
      });

  auto graph_replay_hook_handle =
      comm->registerGraphReplayHook([self, comm_name](
                                        uint64_t graph_id,
                                        uint64_t replay_id,
                                        void* stream,
                                        size_t collective_index,
                                        std::string_view event) {
        self->onGraphReplayEvent(
            comm_name, graph_id, replay_id, stream, collective_index, event);
      });

  registrations_.push_back(CommRegistration{.comm = comm});
  active_comm_count_.fetch_add(1, std::memory_order_relaxed);
}

// -- Formatting helpers (static, identical to original Clog) --

double ClogHook::now() {
  auto tp = std::chrono::system_clock::now();
  auto duration = tp.time_since_epoch();
  auto millis = std::chrono::duration_cast<std::chrono::milliseconds>(duration);
  return static_cast<double>(millis.count()) / 1000.0;
}

// graph_id is globally unique per the CUDA spec:
// https://docs.nvidia.com/cuda/archive/12.4.1/cuda-runtime-api/group__CUDART__STREAM.html#group__CUDART__STREAM_1g150be2211d73d782bc34c497ddb06f2f
ClogHook::GraphCaptureInfo ClogHook::getGraphCaptureInfo(int device_index) {
#ifdef CLOG_HAS_CUDA
  if (device_index < 0) {
    return {};
  }
  cudaStream_t stream = at::cuda::getCurrentCUDAStream(device_index).stream();
  cudaStreamCaptureStatus status;
  unsigned long long graph_id = 0;
  cudaError_t err = cudaStreamGetCaptureInfo(stream, &status, &graph_id);
  if (err == cudaSuccess && status == cudaStreamCaptureStatusActive) {
    return {stream, static_cast<uint64_t>(graph_id)};
  }
#endif
  return {};
}

// -- Fake stream management --

void* ClogHook::getFakeStream(const std::string& comm_name) {
  auto id = next_fake_stream_.fetch_add(1, std::memory_order_relaxed);
  auto* fake =
      reinterpret_cast<void*>(~id); // NOLINT(performance-no-int-to-ptr)
  return comm_fake_streams_.findOrInsert(comm_name, fake);
}

// -- Stream resolution for replay hooks --

void* ClogHook::resolveReplayStream(
    uint64_t graph_id,
    const std::string& comm_name,
    void* stream) {
  auto stream_map_opt = graph_collectives_.find(graph_id);
  if (!stream_map_opt) {
    return stream;
  }
  auto sit = stream_map_opt->find(stream);
  if (sit != stream_map_opt->end()) {
    return stream;
  }
  // Unknown stream — must be a comm-internal stream; use fake stream
  auto fake_opt = comm_fake_streams_.find(comm_name);
  if (fake_opt) {
    return *fake_opt;
  }
  return stream;
}

// -- I/O --

void ClogHook::logEvent(uint64_t corr_id, std::string_view event) {
  double delta = now() - base_ts_;
  log_file_.writeLine(fmt::format("C{}|{}|+{:.3f}", corr_id, event, delta));
}

void ClogHook::logGraphEvent(
    uint64_t graph_id,
    uint64_t corr_id,
    std::string_view event) {
  double delta = now() - base_ts_;
  log_file_.writeLine(
      fmt::format("G{}|C{}|{}|+{:.3f}", graph_id, corr_id, event, delta));
}

// -- Core logging --

WorkId ClogHook::logCollective(
    std::string_view comm_name,
    std::string sig_body,
    bool async_op,
    void* stream,
    uint64_t graph_id) {
  auto sig_key = std::string(sig_body);

  auto new_corr_id = next_corr_id_.fetch_add(1, std::memory_order_relaxed);
  auto corr_id = sig_map_.findOrInsert(sig_key, new_corr_id);

  // Track correlation IDs per graph per stream for replay hook lookups
  if (graph_id != kNoGraphCapture) {
    void* stream_key =
        async_op ? getFakeStream(std::string(comm_name)) : stream;
    graph_collectives_.insertOrModify(graph_id, [&](auto& stream_map) {
      stream_map[stream_key].push_back(
          GraphCollective{std::string(comm_name), corr_id});
    });
  }

  if (corr_id == new_corr_id) {
    log_file_.writeLine(fmt::format("C{}|sig|{}", corr_id, sig_key));
  }

  uint64_t work_id = next_work_id_.fetch_add(1, std::memory_order_relaxed);
  if (log_lifecycle_) {
    work_corr_map_.insert(
        work_id, WorkInfo{corr_id, graph_id, std::string(comm_name)});
    work_events_map_.insert(
        work_id,
        async_op ? std::vector<std::string>{"S", "E", "W"}
                 : std::vector<std::string>{"S", "E"});
  }

  auto q_event = fmt::format("Q|work_id={}", work_id);
  if (graph_id != kNoGraphCapture) {
    logGraphEvent(graph_id, corr_id, q_event);
  } else {
    logEvent(corr_id, q_event);
  }

  return work_id;
}

// -- Pre-hook --

void ClogHook::onPreHook(
    const std::string& comm_name,
    int device_index,
    size_t op_id,
    const PreHookArgs& args) {
  if (auto* split = std::get_if<SplitPreHookArgs>(&args)) {
    log_file_.writeLine(buildSplitLine(comm_name, *split));
    return;
  }

  if (std::get_if<NewWindowPreHookArgs>(&args) ||
      std::get_if<FinalizePreHookArgs>(&args)) {
    return;
  }

  auto sig = buildSignature(comm_name, args, log_buffers_);
  assert(!sig.empty());

  bool async_op = getAsyncOp(args);
  auto [stream, graph_id] = getGraphCaptureInfo(device_index);
  auto work_id =
      logCollective(comm_name, std::move(sig), async_op, stream, graph_id);

  if (work_id != kWorkIdInvalid) {
    op_to_work_.insert(op_id, work_id);
  }
}

// -- Post-hook --

void ClogHook::onPostHook(
    const std::string& comm_name,
    size_t op_id,
    const PostHookArgs& args) {
  // For split, register the new communicator.
  if (auto* split = std::get_if<SplitPostHookArgs>(&args)) {
    if (auto new_comm = split->new_comm.lock()) {
      registerHooks(new_comm);
    }
    return;
  }

  // TODO: add logging for window operations.
  if (std::get_if<NewWindowPostHookArgs>(&args)) {
    return;
  }

  if (std::get_if<FinalizePostHookArgs>(&args)) {
    auto remaining =
        active_comm_count_.fetch_sub(1, std::memory_order_acq_rel) - 1;
    if (remaining == 0) {
      work_events_map_.forEach(
          [&](uint64_t work_id, const auto& pending_events) {
            auto work_info = work_corr_map_.find(work_id);
            if (!work_info) {
              return;
            }
            std::string events_str;
            for (size_t i = 0; i < pending_events.size(); ++i) {
              if (i > 0) {
                events_str += ',';
              }
              events_str += pending_events[i];
            }
            log_file_.writeLine(fmt::format(
                "WARN|comm={}|leaked work_id={}|corr_id={}|pending_events={}",
                work_info->comm_name,
                work_id,
                work_info->corr_id,
                events_str));
          });
    }
    return;
  }

  // Retrieve the work_id assigned in the pre-hook.
  WorkId work_id = op_to_work_.findAndErase(op_id).value_or(kWorkIdInvalid);

  if (work_id == kWorkIdInvalid) {
    log_file_.writeLine(
        fmt::format("WARN|post-hook missing work_id for op_id {}", op_id));
    return;
  }

  // For collectives with a work object, register lifecycle hooks.
  std::visit(
      [this, work_id](const auto& a) {
        using T = std::decay_t<decltype(a)>;
        if constexpr (std::is_base_of_v<CollectivePostHookArgs, T>) {
          if (auto work = a.work.lock()) {
            if (!log_lifecycle_) {
              return;
            }
            work->registerWorkStartHook(
                [this, work_id]() { logLifecycleEvent(work_id, "S"); });
            work->registerWorkEndHook(
                [this, work_id]() { logLifecycleEvent(work_id, "E"); });
            work->registerWorkWaitPreHook(
                [this, work_id]() { logLifecycleEvent(work_id, "W"); });
          }
        }
      },
      args);
}

// -- Lifecycle events --

void ClogHook::logLifecycleEvent(WorkId work_id, std::string_view event) {
  if (work_id == kWorkIdInvalid) {
    return;
  }

  auto work_info_opt = work_corr_map_.find(work_id);
  if (!work_info_opt) {
    log_file_.writeLine(fmt::format(
        "WARN|lifecycle event {} for unknown work_id {}", event, work_id));
    return;
  }
  auto corr_id = work_info_opt->corr_id;
  auto graph_id = work_info_opt->graph_id;

  if (!work_events_map_.valueRemove(work_id, std::string(event))) {
    log_file_.writeLine(fmt::format(
        "WARN|unexpected lifecycle event {} for work_id {}", event, work_id));
    return;
  }

  // Clean up corr_id entry if no more events remain
  if (!work_events_map_.find(work_id)) {
    work_corr_map_.findAndErase(work_id);
  }

  if (graph_id != kNoGraphCapture) {
    logGraphEvent(graph_id, corr_id, event);
  } else {
    logEvent(corr_id, event);
  }
}

// -- Graph replay events --

void ClogHook::onGraphReplayEvent(
    const std::string& comm_name,
    uint64_t graph_id,
    uint64_t replay_id,
    void* stream,
    size_t collective_index,
    std::string_view event) {
  void* resolved = resolveReplayStream(graph_id, comm_name, stream);

  auto stream_map_opt = graph_collectives_.find(graph_id);
  if (!stream_map_opt) {
    log_file_.writeLine(fmt::format(
        "WARN|graph replay event for unknown graph_id {}", graph_id));
    return;
  }
  auto sit = stream_map_opt->find(resolved);
  if (sit == stream_map_opt->end() || collective_index >= sit->second.size()) {
    log_file_.writeLine(fmt::format(
        "WARN|graph replay event for unknown stream in graph_id {} index {}",
        graph_id,
        collective_index));
    return;
  }

  auto corr_id = sit->second[collective_index].corr_id;

  if (!log_lifecycle_) {
    return;
  }

  double delta = now() - base_ts_;
  log_file_.writeLine(fmt::format(
      "G{}|R{}|C{}|{}|+{:.3f}", graph_id, replay_id, corr_id, event, delta));
}

} // namespace torch::comms
