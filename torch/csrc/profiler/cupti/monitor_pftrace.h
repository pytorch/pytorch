#pragma once

// Perfetto-native (.pftrace) encoder for the CUPTI monitor. The Python side
// (torch/profiler/_cupti/monitor_trace.py) shapes the columnar trace window
// into a track list + per-kind groups of flat arrays + annotation specs; the
// pybind glue in monitor_python.cpp turns those into the plain C++ inputs below
// and calls cuptiMonitorEncodePftrace, which emits a concatenated stream of
// perfetto TracePackets via protozero (the perfetto SDK amalgamation under
// third_party/perfetto). Each slice becomes a SLICE_BEGIN (carrying its
// debug_annotations + flow) and a SLICE_END. Returns the raw, uncompressed
// trace bytes; the caller gzips. perfetto.h is pulled in only by the .cpp, and
// this stays free of any Python/pybind dependency so it can live in torch_cpu.

#include <c10/macros/Macros.h>

#include <cstdint>
#include <string>
#include <vector>

namespace torch::profiler::impl {

// A trace track: a process track (is_process) carries pid + name; a thread
// track carries pid + tid + name under parent (its process track's uuid, 0 if
// none).
struct PftraceTrack {
  uint64_t uuid;
  uint64_t parent;
  bool is_process;
  int32_t pid;
  int32_t tid;
  std::string name;
};

// One int debug-annotation column: int_value = vals[i]. Skipped for slice i
// when skip_zero and vals[i]==0 (matches the chrome path patching ids on only
// when nonzero), or when present is non-null and present[i]==0 (used for the
// unmapped half of an enum -- see PftraceStrAnno).
struct PftraceIntAnno {
  std::string key;
  const int64_t* vals;
  bool skip_zero;
  const uint8_t* present; // nullable: emit only where present[i] != 0
};

// One string debug-annotation column: string_value = table[idx[i]], skipped for
// slice i when idx[i] < 0. Low-cardinality enum names (copy kind, sync kind,
// ...) are passed as a small table + an index column rather than per-slice
// strings; idx < 0 marks a value with no name (its raw int is emitted via a
// paired PftraceIntAnno instead).
struct PftraceStrAnno {
  std::string key;
  const int64_t* idx;
  std::vector<std::string> table;
};

// One array debug-annotation column (e.g. grid / block): array_values is the
// per-slice tuple [cols[0][i], cols[1][i], ...] of int values.
struct PftraceArrAnno {
  std::string key;
  std::vector<const int64_t*> cols;
};

// One JSON debug-annotation column (annotation / collective-descriptor blob).
// Slice i's blob is buffer[offsets[i], offsets[i + 1]); empty (offsets equal)
// means none. Parsed natively and spread like _annotation_to_args: an object's
// top-level keys each become an annotation, a list/scalar/parse-failure becomes
// a single "annotation".
struct PftraceJsonAnno {
  const int32_t* offsets; // length n + 1
  const char* buffer;
};

// One per-kind group of slices sharing an annotation schema. ts / end /
// track_uuid / name_iid are length-n parallel columns; flow (nullable) is a
// per- slice ac2g flow id (0 = no flow). name_iid indexes the global name
// table.
struct PftraceGroup {
  const int64_t* ts;
  const int64_t* end;
  const uint64_t* track_uuid;
  const uint64_t* name_iid;
  size_t n;
  std::vector<PftraceIntAnno> int_annos;
  std::vector<PftraceStrAnno> str_annos;
  std::vector<PftraceArrAnno> arr_annos;
  std::vector<PftraceJsonAnno> json_annos;
  const int64_t* flow; // nullable
};

// tracks -> TrackDescriptor packets; name_table -> one interned EventName table
// (iid == index + 1). Emits each group's SLICE_BEGINs (with debug_annotations +
// flow) then SLICE_ENDs. trace_processor reorders by timestamp.
TORCH_API std::string cuptiMonitorEncodePftrace(
    const std::vector<PftraceTrack>& tracks,
    const std::vector<std::string>& name_table,
    const std::vector<PftraceGroup>& groups);

} // namespace torch::profiler::impl
