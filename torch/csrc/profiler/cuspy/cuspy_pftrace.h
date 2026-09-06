#pragma once

// Perfetto-native (.pftrace) encoder for Cuspy. The Python side
// (torch/profiler/_cuspy/trace.py) shapes the columnar trace window
// into a track list + per-kind groups of flat arrays + annotation specs; the
// pybind glue in cuspy_python.cpp turns those into the plain C++ inputs below
// and calls cuspyEncodePftrace, which emits a concatenated stream of
// perfetto TracePackets via protozero (the perfetto SDK amalgamation under
// third_party/perfetto). Each slice becomes a SLICE_BEGIN (carrying its
// debug_annotations + flow) and a SLICE_END. Returns the gzip-compressed trace
// bytes (ready to write as .pftrace.gz); gzipping here keeps the uncompressed
// buffer (tens of MB) off the pybind boundary. perfetto.h is pulled in only by
// the .cpp, and this stays free of any Python/pybind dependency so it lives in
// torch_cpu.

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

// An interned gpu_specifications entry: a render stage (Kernel/Memcpy/Memset)
// or a hardware-queue lane (Channel #N). Stages and queues share one iid space,
// as in Perfetto's GpuRenderStageEvent format. category is the
// InternedGpuRenderStageSpecification::RenderStageCategory enum (0=OTHER,
// 1=GRAPHICS, 2=COMPUTE).
struct PftraceGpuSpec {
  uint64_t iid;
  std::string name;
  int32_t category;
};

// An interned graphics_contexts entry: a GPU context handle -> owning pid.
struct PftraceGfxContext {
  uint64_t iid;
  int32_t pid;
};

// One GpuRenderStageEvent extra_data column: emits name=key,
// value=to_string(vals[i]); skipped where skip_zero && vals[i]==0 (so
// kernel-only fields are absent on memcpy/memset).
struct PftraceRenderExtra {
  std::string key;
  const int64_t* vals;
  bool skip_zero;
};

// A constant string extra_data (name=key, value=value) emitted on every
// GpuRenderStageEvent. For trace-wide string args the viewer's GPU Compute
// panel reads off the slice (arch / process_id / process_name) -- extra_data
// values are strings, which the numeric PftraceRenderExtra above can't carry.
struct PftraceConstExtra {
  std::string key;
  std::string value;
};

// One ComputeKernelLaunch.args column. The arg name is interned
// (InternedComputeArgName, field 1001 on InternedData) and referenced by
// name_iid -- this is what the viewer's "Launch Statistics" panel reads (see
// the trace_processor ParseExtraComputeArg name_iid -> InternedComputeArgName
// join).
struct PftraceComputeArg {
  uint64_t name_iid;
  const int64_t* vals;
  bool skip_zero;
};

// An interned entry (iid + name) for the compute-kernel / arg-name tables, raw-
// emitted onto InternedData at field 1000 (InternedComputeKernel) and 1001
// (InternedComputeArgName) respectively -- the v56.1 amalgamation defines the
// message types but not the InternedData accessors, so they go in by field id.
struct PftraceComputeName {
  uint64_t iid;
  std::string name;
};

// GpuRenderStageEvent columns: one event per element, each with its own
// duration (no SLICE_BEGIN/END pairing), so overlaps are fine and durations are
// exact. gpu_id + hw_queue_iid select the lane; stage_iid tags the op kind;
// context indexes graphics_contexts. n == 0 means no render stages.
//
// Compute kernels (grid_x != null && grid_x[i] > 0) get a structured
// ComputeKernelLaunch (grid_size/workgroup_size Dim3 + launch_args) plus a
// kernel_iid -> InternedComputeKernel (kernel name) -- this is what drives the
// viewer's GPU Compute "Launch Statistics" panel and names the lane slice.
// extra carries generic extra_data for the non-kernel kinds (memcpy/memset
// bytes). compute_kernels / compute_arg_names are the interned tables
// referenced by kernel_iid and launch_args[].name_iid.
struct PftraceRenderStages {
  const int64_t* ts;
  const int64_t* dur;
  const uint64_t* event_id;
  const int64_t* gpu_id;
  const uint64_t* hw_queue_iid;
  const uint64_t* stage_iid;
  const uint64_t* context;
  size_t n;
  const int64_t* grid_x; // nullable; > 0 marks a compute-kernel launch
  const int64_t* grid_y;
  const int64_t* grid_z;
  const int64_t* block_x;
  const int64_t* block_y;
  const int64_t* block_z;
  const uint64_t* kernel_iid; // nullable; -> InternedComputeKernel (0 = none)
  const uint64_t* name_iid; // nullable; -> EventName, the timeline slice label
  // nullable CSR (event_wait_offsets length n+1, event_wait_ids flat): render
  // stage i depends on the event_ids in event_wait_ids[event_wait_offsets[i],
  // event_wait_offsets[i+1]) -> GpuRenderStageEvent.event_wait_ids (field 18),
  // which the viewer draws as node->node flow arrows (the graph dependency
  // DAG).
  const int32_t* event_wait_offsets;
  const uint64_t* event_wait_ids;
  std::vector<PftraceComputeArg> launch_args; // -> ComputeKernelLaunch.args
  std::vector<PftraceRenderExtra> extra; // -> GpuRenderStageEvent.extra_data
  std::vector<PftraceConstExtra> const_extra; // trace-wide string extra_data
  std::vector<PftraceComputeName> compute_kernels; // -> InternedData field 1000
  std::vector<PftraceComputeName>
      compute_arg_names; // -> InternedData field 1001
  // Per-row collective-descriptor JSON blob (row i is
  // meta_buffer[meta_offsets[i], meta_offsets[i+1]); empty = none), spread onto
  // the render stage as extra_data (mirrors the chrome path's
  // metadata-into-args, so consumers need no correlation-id join). meta_offsets
  // nullable (no metadata column).
  const int32_t* meta_offsets; // nullable, length n + 1
  const char* meta_buffer;
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
  // nullable CSR (gpu_corr_offsets length n+1, gpu_corr_ids flat): per-slice
  // GpuTrackEvent.gpu_correlation (field 3000), whose
  // render_stage_submission_event_ids (field 1) is slice i's list
  // gpu_corr_ids[gpu_corr_offsets[i], gpu_corr_offsets[i+1]) -- the
  // render-stage event_ids this host launch submitted (a graph launch links to
  // all its replay's nodes). Empty slice (offsets equal) = no link.
  const int32_t* gpu_corr_offsets;
  const int64_t* gpu_corr_ids;
  // nullable: per-slice interned EventCategory iid (index into category_table +
  // 1), emitted as TrackEvent.category_iids so the viewer can tell cpu_op /
  // cuda_runtime / cuda_driver / user_annotation slices apart. 0 = no category.
  const uint64_t* cat_iid;
};

// GPU counters from the sampled CUpti_ActivityEnvironment records (power /
// temperature / clocks). Emitted as Perfetto GpuCounterEvents so the viewer
// renders them under "GPU / Counters / <gpu>" -- a sibling of the render-stage
// "GPU / Hardware Queues" -- rather than as process-tree counter tracks. specs
// is the GpuCounterDescriptor (counter_id -> display name), emitted once; then
// one GpuCounterEvent sample per row: at ts[i], gpu_id[i]'s counter_id[i] takes
// value[i]. n == 0 means no GPU counters.
struct PftraceGpuCounter {
  std::vector<std::pair<uint32_t, std::string>> specs;
  const int32_t* gpu_id;
  const int64_t* ts;
  const int32_t* counter_id;
  const double* value;
  size_t n;
  // counter_ids placed in the COMPUTE GpuCounterGroup (group_id=6) on the
  // descriptor -- the GPU Compute panel only joins counters in this group to
  // kernels (e.g. gpc__cycles_elapsed -> Cycles).
  std::vector<uint32_t> compute_group;
  // counter_ids emitted as GpuCounter.int_value (rounded) rather than
  // double_value -- integer counts like gpc__cycles_elapsed.
  std::vector<uint32_t> int_value_ids;
};

// tracks -> TrackDescriptor packets; name_table -> one interned EventName table
// (iid == index + 1); category_table -> one interned EventCategory table (iid
// == index + 1), referenced per-slice by PftraceGroup::cat_iid. Emits each
// group's SLICE_BEGINs (with debug_annotations + flow) then SLICE_ENDs.
// trace_processor reorders by timestamp. gpu_specs + gfx_contexts are interned
// alongside the name table, and stages -> one GpuRenderStageEvent packet each
// (the native "GPU Render Stages" hardware-queue lanes; additive to the
// track_event slices, which keep the flows + args). counters -> counter tracks
// + their samples.
//
// base_ns is the trace's absolute time base (the JSON export's
// baseTimeNanoseconds): the packet timestamps are already absolute (base_ns +
// offset), and a ClockSnapshot anchors them to REALTIME at base_ns so the wall-
// clock base is recoverable -- the native counterpart of baseTimeNanoseconds.
//
// compression_level is the gzip level applied to the serialized trace (0-9;
// 1 = fast, the default). Invalid levels fall back to uncompressed bytes.
TORCH_API std::string cuspyEncodePftrace(
    int64_t base_ns,
    const std::vector<PftraceTrack>& tracks,
    const std::vector<std::string>& name_table,
    const std::vector<std::string>& category_table,
    const std::vector<PftraceGroup>& groups,
    const std::vector<PftraceGpuSpec>& gpu_specs,
    const std::vector<PftraceGfxContext>& gfx_contexts,
    const PftraceRenderStages& stages,
    const PftraceGpuCounter& counters,
    int compression_level);

} // namespace torch::profiler::impl
