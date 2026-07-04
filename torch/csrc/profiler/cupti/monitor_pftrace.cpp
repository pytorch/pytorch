#include <torch/csrc/profiler/cupti/monitor_pftrace.h>

#include <perfetto.h>

#include <nlohmann/json.hpp>

#include <zlib.h>

#include <array>
#include <cmath>
#include <map>
#include <set>

namespace torch::profiler::impl {

namespace {
namespace pbz = perfetto::protos::pbzero;

// gzip-compress (level 1, fast) with a gzip wrapper (windowBits 15|16) so the
// result is a ready-to-write .pftrace.gz. Done here, not in Python, so the
// uncompressed trace (tens of MB) never crosses the pybind boundary. Falls back
// to the uncompressed bytes if zlib init fails.
std::string gzipCompress(const std::string& data) {
  z_stream zs{};
  if (deflateInit2(&zs, 1, Z_DEFLATED, 15 | 16, 8, Z_DEFAULT_STRATEGY) !=
      Z_OK) {
    return data;
  }
  // deflate does not modify the input; the const_cast satisfies zlib's
  // non-const Bytef* API.
  // NOLINTNEXTLINE(cppcoreguidelines-pro-type-const-cast)
  zs.next_in = reinterpret_cast<Bytef*>(const_cast<char*>(data.data()));
  zs.avail_in = static_cast<uInt>(data.size());
  std::string out;
  out.reserve(data.size() / 3 + 64);
  std::array<char, 1 << 16> buf{};
  int ret = Z_OK;
  do {
    zs.next_out = reinterpret_cast<Bytef*>(buf.data());
    zs.avail_out = static_cast<uInt>(buf.size());
    ret = deflate(&zs, Z_FINISH);
    out.append(buf.data(), buf.size() - zs.avail_out);
  } while (ret == Z_OK);
  deflateEnd(&zs);
  return out;
}

// Emit a parsed JSON value into a DebugAnnotation (recursing for object/array),
// mirroring the chrome path's typed args.
void emitJsonValue(pbz::DebugAnnotation* a, const nlohmann::json& v) {
  if (v.is_object()) {
    for (auto it = v.begin(); it != v.end(); ++it) {
      auto* d = a->add_dict_entries();
      d->set_name(it.key());
      emitJsonValue(d, it.value());
    }
  } else if (v.is_array()) {
    for (const auto& e : v) {
      emitJsonValue(a->add_array_values(), e);
    }
  } else if (v.is_boolean()) {
    a->set_bool_value(v.get<bool>());
  } else if (v.is_number_float()) {
    a->set_double_value(v.get<double>());
  } else if (v.is_number()) {
    a->set_int_value(v.get<int64_t>());
  } else if (v.is_string()) {
    a->set_string_value(v.get<std::string>());
  } else { // null
    a->set_string_value("null");
  }
}

// Spread one JSON blob onto a TrackEvent like _annotation_to_args: object ->
// top-level keys each an annotation; list -> a single "annotation" string
// (json.dumps); scalar -> a single typed "annotation"; parse failure -> the raw
// blob as "annotation".
void emitJsonBlob(pbz::TrackEvent* te, const char* data, size_t len) {
  auto parsed = nlohmann::json::parse(data, data + len, nullptr, false);
  if (parsed.is_discarded()) {
    auto* a = te->add_debug_annotations();
    a->set_name("annotation");
    a->set_string_value(data, len);
  } else if (parsed.is_object()) {
    for (auto it = parsed.begin(); it != parsed.end(); ++it) {
      auto* a = te->add_debug_annotations();
      a->set_name(it.key());
      emitJsonValue(a, it.value());
    }
  } else if (parsed.is_array()) {
    auto* a = te->add_debug_annotations();
    a->set_name("annotation");
    a->set_string_value(parsed.dump());
  } else {
    auto* a = te->add_debug_annotations();
    a->set_name("annotation");
    emitJsonValue(a, parsed);
  }
}
} // namespace

std::string cuptiMonitorEncodePftrace(
    const std::vector<PftraceTrack>& tracks,
    const std::vector<std::string>& name_table,
    const std::vector<PftraceGroup>& groups,
    const std::vector<PftraceGpuSpec>& gpu_specs,
    const std::vector<PftraceGfxContext>& gfx_contexts,
    const PftraceRenderStages& stages,
    const PftraceGpuCounter& counters) {
  protozero::HeapBuffered<pbz::Trace> trace;

  // Per-gpu InternedGpuCounterDescriptor iid (its own interned-id space). The
  // descriptor is interned (in interned_data) and the samples reference it by
  // counter_descriptor_iid -- inline descriptors on the event don't link.
  std::map<int32_t, uint64_t> gpu_counter_iid;
  for (size_t i = 0; i < counters.n; ++i) {
    int32_t g = counters.gpu_id[i];
    if (gpu_counter_iid.find(g) == gpu_counter_iid.end()) {
      gpu_counter_iid[g] = gpu_counter_iid.size() + 1;
    }
  }

  for (const auto& tr : tracks) {
    auto* td = trace->add_packet()->set_track_descriptor();
    td->set_uuid(tr.uuid);
    if (tr.parent) {
      td->set_parent_uuid(tr.parent);
    }
    if (tr.is_process) {
      auto* p = td->set_process();
      p->set_pid(tr.pid);
      p->set_process_name(tr.name);
    } else {
      auto* th = td->set_thread();
      th->set_pid(tr.pid);
      th->set_tid(tr.tid);
      th->set_thread_name(tr.name);
    }
  }

  if (!name_table.empty() || !gpu_specs.empty() || !gfx_contexts.empty() ||
      !stages.compute_kernels.empty() || !stages.compute_arg_names.empty() ||
      !gpu_counter_iid.empty()) {
    auto* pkt = trace->add_packet();
    pkt->set_trusted_packet_sequence_id(1);
    pkt->set_sequence_flags(pbz::TracePacket::SEQ_INCREMENTAL_STATE_CLEARED);
    auto* interned = pkt->set_interned_data();
    // InternedGpuCounterDescriptor per gpu (iid + gpu_id + counter_descriptor
    // specs); the gpu_counter_events reference it by counter_descriptor_iid.
    for (const auto& [gpu, diid] : gpu_counter_iid) {
      auto* desc = interned->add_gpu_counter_descriptors();
      desc->set_iid(diid);
      desc->set_gpu_id(gpu);
      auto* cd = desc->set_counter_descriptor();
      for (const auto& spec : counters.specs) {
        auto* sp = cd->add_specs();
        sp->set_counter_id(spec.first);
        sp->set_name(spec.second);
      }
      // COMPUTE group: the GPU Compute panel joins only group_id=6 counters to
      // kernels (Cycles / SM Frequency from gpc__cycles_elapsed).
      if (!counters.compute_group.empty()) {
        auto* grp = cd->add_counter_groups();
        grp->set_group_id(pbz::GpuCounterDescriptor::COMPUTE);
        for (uint32_t cid : counters.compute_group) {
          grp->add_counter_ids(cid);
        }
      }
    }
    uint64_t iid = 1;
    for (const auto& nm : name_table) {
      auto* en = interned->add_event_names();
      en->set_iid(iid++);
      en->set_name(nm);
    }
    for (const auto& s : gpu_specs) {
      auto* sp = interned->add_gpu_specifications();
      sp->set_iid(s.iid);
      sp->set_name(s.name);
      sp->set_category(
          static_cast<
              pbz::InternedGpuRenderStageSpecification::RenderStageCategory>(
              s.category));
    }
    for (const auto& c : gfx_contexts) {
      auto* gc = interned->add_graphics_contexts();
      gc->set_iid(c.iid);
      gc->set_pid(c.pid);
    }
    // InternedComputeKernel (field 1000) + InternedComputeArgName (field 1001):
    // the amalgamation defines the message types but not the InternedData
    // accessors (they belong to the GpuInternedData view), so write them by
    // field id via protozero. The viewer joins GpuRenderStageEvent.kernel_iid
    // and ExtraComputeArg.name_iid against these.
    for (const auto& k : stages.compute_kernels) {
      auto* ck = interned->BeginNestedMessage<pbz::InternedComputeKernel>(1000);
      ck->set_iid(k.iid);
      // CUPTI gives an already-demangled name, and the viewer labels the lane
      // slice (and groups Launch Statistics) from demangled_name (kernel_iid ->
      // InternedComputeKernel.demangled_name); without it the slice falls back
      // to the stage name ("Kernel"). The reference traces set only
      // demangled_name (no separate mangled `name`), so we match -- emitting
      // both would just duplicate the same string as a redundant kernel_name.
      ck->set_demangled_name(k.name);
    }
    for (const auto& a : stages.compute_arg_names) {
      auto* an =
          interned->BeginNestedMessage<pbz::InternedComputeArgName>(1001);
      an->set_iid(a.iid);
      an->set_name(a.name);
    }
  }

  for (const auto& g : groups) {
    for (size_t i = 0; i < g.n; ++i) {
      auto* pkt = trace->add_packet();
      pkt->set_timestamp(static_cast<uint64_t>(g.ts[i]));
      pkt->set_trusted_packet_sequence_id(1);
      pkt->set_sequence_flags(pbz::TracePacket::SEQ_NEEDS_INCREMENTAL_STATE);
      auto* te = pkt->set_track_event();
      te->set_type(pbz::TrackEvent::TYPE_SLICE_BEGIN);
      te->set_track_uuid(g.track_uuid[i]);
      te->set_name_iid(g.name_iid[i]);
      if (g.flow && g.flow[i]) {
        te->add_flow_ids(static_cast<uint64_t>(g.flow[i]));
      }
      // GpuTrackEvent.gpu_correlation (field 3000): a GpuCorrelation message
      // whose render_stage_submission_event_ids (field 1) is this correlation;
      // the viewer links it to the render stage with event_id == this value.
      // The extension type is absent from the amalgamation, so emit by field
      // id.
      if (g.gpu_corr && g.gpu_corr[i]) {
        auto* gc = te->BeginNestedMessage<protozero::Message>(3000);
        gc->AppendVarInt(1, static_cast<uint64_t>(g.gpu_corr[i]));
      }
      for (const auto& a : g.int_annos) {
        if (a.skip_zero && a.vals[i] == 0) {
          continue;
        }
        if (a.present && !a.present[i]) {
          continue;
        }
        auto* da = te->add_debug_annotations();
        da->set_name(a.key);
        da->set_int_value(a.vals[i]);
      }
      for (const auto& a : g.str_annos) {
        if (a.idx[i] < 0) {
          continue;
        }
        auto* da = te->add_debug_annotations();
        da->set_name(a.key);
        da->set_string_value(a.table[static_cast<size_t>(a.idx[i])]);
      }
      for (const auto& a : g.arr_annos) {
        auto* da = te->add_debug_annotations();
        da->set_name(a.key);
        for (const auto* col : a.cols) {
          da->add_array_values()->set_int_value(col[i]);
        }
      }
      for (const auto& a : g.json_annos) {
        const int32_t off = a.offsets[i];
        const int32_t len = a.offsets[i + 1] - off;
        if (len > 0) {
          emitJsonBlob(te, a.buffer + off, static_cast<size_t>(len));
        }
      }
    }
    for (size_t i = 0; i < g.n; ++i) {
      auto* pkt = trace->add_packet();
      pkt->set_timestamp(static_cast<uint64_t>(g.end[i]));
      pkt->set_trusted_packet_sequence_id(1);
      auto* te = pkt->set_track_event();
      te->set_type(pbz::TrackEvent::TYPE_SLICE_END);
      te->set_track_uuid(g.track_uuid[i]);
    }
  }

  // GPU Render Stages: one packet per GPU op, on its (gpu_id, hw_queue) lane.
  for (size_t i = 0; i < stages.n; ++i) {
    auto* pkt = trace->add_packet();
    pkt->set_timestamp(static_cast<uint64_t>(stages.ts[i]));
    pkt->set_trusted_packet_sequence_id(1);
    pkt->set_sequence_flags(pbz::TracePacket::SEQ_NEEDS_INCREMENTAL_STATE);
    auto* gse = pkt->set_gpu_render_stage_event();
    gse->set_event_id(stages.event_id[i]);
    gse->set_duration(static_cast<uint64_t>(stages.dur[i]));
    gse->set_gpu_id(static_cast<int32_t>(stages.gpu_id[i]));
    gse->set_hw_queue_iid(stages.hw_queue_iid[i]);
    gse->set_stage_iid(stages.stage_iid[i]);
    gse->set_context(stages.context[i]);
    // The slice's timeline label: the viewer uses the event's name_iid
    // (EventName), falling back to the stage name ("Kernel") when absent.
    if (stages.name_iid && stages.name_iid[i]) {
      gse->set_name_iid(stages.name_iid[i]);
    }
    // Compute kernels: kernel_iid -> InternedComputeKernel (names the slice)
    // and a structured ComputeKernelLaunch (grid/workgroup Dim3 + args) -> the
    // viewer's GPU Compute "Launch Statistics" panel. Each arg names itself via
    // name_iid -> InternedComputeArgName (the join the viewer reads).
    if (stages.grid_x && stages.grid_x[i] > 0) {
      if (stages.kernel_iid && stages.kernel_iid[i]) {
        gse->set_kernel_iid(stages.kernel_iid[i]);
      }
      auto* launch = gse->set_launch();
      auto* grid = launch->set_grid_size();
      grid->set_x(static_cast<uint32_t>(stages.grid_x[i]));
      grid->set_y(static_cast<uint32_t>(stages.grid_y[i]));
      grid->set_z(static_cast<uint32_t>(stages.grid_z[i]));
      auto* wg = launch->set_workgroup_size();
      wg->set_x(static_cast<uint32_t>(stages.block_x[i]));
      wg->set_y(static_cast<uint32_t>(stages.block_y[i]));
      wg->set_z(static_cast<uint32_t>(stages.block_z[i]));
      for (const auto& a : stages.launch_args) {
        if (a.skip_zero && a.vals[i] == 0) {
          continue;
        }
        auto* arg = launch->add_args();
        arg->set_name_iid(a.name_iid);
        arg->set_uint_value(static_cast<uint64_t>(a.vals[i]));
      }
    }
    for (const auto& e : stages.extra) {
      if (e.skip_zero && e.vals[i] == 0) {
        continue;
      }
      auto* ed = gse->add_extra_data();
      ed->set_name(e.key);
      ed->set_value(std::to_string(e.vals[i]));
    }
    // Trace-wide string args (arch / process_id / process_name) the GPU Compute
    // panel reads per-slice; emitted on every event with a constant value.
    for (const auto& ce : stages.const_extra) {
      auto* ed = gse->add_extra_data();
      ed->set_name(ce.key);
      ed->set_value(ce.value);
    }
  }

  // GPU counters (power / temperature / clocks from sampled environment
  // records): one GpuCounterEvent per sample, referencing the per-gpu
  // InternedGpuCounterDescriptor (interned above) by counter_descriptor_iid.
  // The viewer renders these under "GPU / Counters / <gpu>", a sibling of the
  // render-stage "GPU / Hardware Queues".
  const std::set<uint32_t> int_ids(
      counters.int_value_ids.begin(), counters.int_value_ids.end());
  for (size_t i = 0; i < counters.n; ++i) {
    auto* pkt = trace->add_packet();
    pkt->set_timestamp(static_cast<uint64_t>(counters.ts[i]));
    pkt->set_trusted_packet_sequence_id(1);
    // Reference the interned descriptor (incremental state) emitted on the
    // SEQ_INCREMENTAL_STATE_CLEARED packet above.
    pkt->set_sequence_flags(pbz::TracePacket::SEQ_NEEDS_INCREMENTAL_STATE);
    auto* gce = pkt->set_gpu_counter_event();
    gce->set_counter_descriptor_iid(gpu_counter_iid[counters.gpu_id[i]]);
    auto* c = gce->add_counters();
    auto cid = static_cast<uint32_t>(counters.counter_id[i]);
    c->set_counter_id(cid);
    if (int_ids.count(cid)) {
      c->set_int_value(static_cast<int64_t>(std::llround(counters.value[i])));
    } else {
      c->set_double_value(counters.value[i]);
    }
  }

  return gzipCompress(trace.SerializeAsString());
}

} // namespace torch::profiler::impl
