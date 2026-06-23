// In-process NCCL profiler plugin (ncclProfiler_v6) that feeds the CUPTI
// monitor's metadata store. It is the metadata half of the CollTrace
// replacement: NCCL hands us each collective's descriptor (func/algo/proto/
// size/...), which we serialize to a JSON blob and stash keyed by the external-
// correlation id the Python comms wrapper pushed around the call (read via the
// host-side mirror). The monitor later joins that blob onto the collective's
// GPU timing by id (see ProfilerObserver._attach_metadata). We never call CUPTI
// here
// -- the id is produced in Python; this plugin only reads it and writes the
// blob.
//
// Loading: NCCL resolves the profiler plugin by dlsym'ing `ncclProfiler_v6`
// from the program image when NCCL_PROFILER_PLUGIN=STATIC_PLUGIN
// (dlopen(NULL)). This TU compiles into torch_cpu (loaded RTLD_GLOBAL), so the
// default-visibility export below is discoverable -- no separate plugin .so.
// The Python side sets the env var before init_process_group (see
// torch.profiler._cupti).

#include <nlohmann/json.hpp>

#include <torch/csrc/profiler/cupti/monitor_native.h>
#include <torch/csrc/profiler/cupti/nccl_profiler_v6.h>

namespace {

const char* or_empty(const char* s) {
  return s != nullptr ? s : "";
}

ncclResult_t torchProfilerInit(
    void** context,
    uint64_t /*commId*/,
    int* eActivationMask,
    const char* /*commName*/,
    int /*nNodes*/,
    int /*nranks*/,
    int /*rank*/,
    ncclDebugLogger_t /*logfn*/) {
  // Activate only host collective events -- the minimum for per-collective
  // metadata. ProxyOp/ProxyStep/KernelCh fire per channel/step and make NCCL do
  // extra instrumentation, so they stay off (see the overhead guardrails). P2p
  // metadata is a follow-up.
  if (eActivationMask != nullptr) {
    *eActivationMask = ncclProfileColl;
  }
  if (context != nullptr) {
    *context = nullptr;
  }
  return ncclSuccess;
}

ncclResult_t torchProfilerStartEvent(
    void* /*context*/,
    void** eHandle,
    ncclProfilerEventDescr_v6_t* eDescr) {
  if (eHandle != nullptr) {
    *eHandle = nullptr; // we track no per-event state
  }
  if (eDescr == nullptr || eDescr->type != ncclProfileColl) {
    return ncclSuccess;
  }
  // Skip building the blob when there's no annotation on this thread's stack
  // (the monitor isn't running / nothing was pushed around this collective).
  // The store keys by this same current id; the comms hook pushes it around the
  // call, and the collective enqueues synchronously on this thread inside that
  // window, so the top of the mirror is this collective's id -- pass it to
  // put_external rather than re-resolving.
  uint64_t external_id = torch::profiler::impl::cuptiMonitorCurrentExternalId();
  if (external_id == 0) {
    return ncclSuccess;
  }
  const auto& c = eDescr->coll;
  nlohmann::json j{
      {"func", or_empty(c.func)},
      {"datatype", or_empty(c.datatype)},
      {"algo", or_empty(c.algo)},
      {"proto", or_empty(c.proto)},
      {"count", c.count},
      {"root", c.root},
      {"n_channels", static_cast<unsigned>(c.nChannels)},
      {"n_warps", static_cast<unsigned>(c.nWarps)},
      {"seq", c.seqNumber},
      {"rank", eDescr->rank},
  };
  torch::profiler::impl::CuptiMetadataStore::get().put_external(
      std::move(j), external_id);
  return ncclSuccess;
}

ncclResult_t torchProfilerStopEvent(void* /*eHandle*/) {
  return ncclSuccess;
}

ncclResult_t torchProfilerRecordEventState(
    void* /*eHandle*/,
    ncclProfilerEventState_v6_t /*eState*/,
    ncclProfilerEventStateArgs_v6_t* /*eStateArgs*/) {
  return ncclSuccess;
}

ncclResult_t torchProfilerFinalize(void* /*context*/) {
  return ncclSuccess;
}

} // namespace

// NCCL dlsym's this symbol from the program image under STATIC_PLUGIN; it must
// be extern "C" with default visibility (torch_cpu is otherwise
// -fvisibility=hidden).
extern "C"
    __attribute__((visibility("default"))) ncclProfiler_v6_t ncclProfiler_v6 = {
        "torch-cupti-monitor",
        torchProfilerInit,
        torchProfilerStartEvent,
        torchProfilerStopEvent,
        torchProfilerRecordEventState,
        torchProfilerFinalize,
};
