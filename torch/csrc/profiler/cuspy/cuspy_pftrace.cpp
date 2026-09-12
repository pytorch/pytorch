#include <torch/csrc/profiler/cuspy/cuspy_pftrace.h>

// TORCH_CUPTI_PFTRACE is defined (see caffe2/CMakeLists.txt) only when the
// CUPTI stubs were generated at build time, which is what pulls in perfetto. In
// that case the real protozero encoder is compiled; otherwise this TU is a
// dependency-free stub (no perfetto) whose encoder raises -- Cuspy
// is unavailable at runtime without the stubs anyway.
#ifdef TORCH_CUPTI_PFTRACE

#include <perfetto.h>

#include <nlohmann/json.hpp>

#include <array>
#include <cmath>
#include <map>
#include <set>

// miniz is included last: its zlib-compatible macros (deflate, compress, ...)
// must not shadow identifiers in the perfetto/nlohmann/std headers above. We
// use only the mz_-prefixed API below, so the macros are harmless here.
#include <miniz.h>

namespace torch::profiler::impl {

namespace {
namespace pbz = perfetto::protos::pbzero;

// gzip-compress at the given level (0-9; 1 = fast, the default) so the result
// is a ready-to-write .pftrace.gz. Done here, not in Python, so the
// uncompressed trace (tens of MB) never crosses the pybind boundary. miniz (the
// in-tree deflate) has no gzip-header mode, so we deflate raw (windowBits -15)
// and wrap it in a minimal gzip frame: a 10-byte header + the raw stream + an
// 8-byte trailer (CRC32 then uncompressed size mod 2^32, both little-endian).
// Falls back to the uncompressed bytes if deflate init fails (e.g. an
// out-of-range level).
std::string gzipCompress(const std::string& data, int level) {
  mz_stream zs{};
  if (mz_deflateInit2(&zs, level, MZ_DEFLATED, -15, 8, MZ_DEFAULT_STRATEGY) !=
      MZ_OK) {
    return data;
  }
  // deflate does not modify the input; the const_cast satisfies miniz's
  // non-const next_in.
  // NOLINTNEXTLINE(cppcoreguidelines-pro-type-const-cast)
  zs.next_in = reinterpret_cast<unsigned char*>(const_cast<char*>(data.data()));
  zs.avail_in = static_cast<unsigned int>(data.size());
  std::string out;
  out.reserve(data.size() / 3 + 64);
  // gzip header: magic (1f 8b), deflate method (08), no flags, zero mtime, no
  // extra flags, unknown OS (ff).
  static constexpr std::array<char, 10> kHeader = {
      '\x1f', '\x8b', '\x08', 0, 0, 0, 0, 0, 0, '\xff'};
  out.append(kHeader.data(), kHeader.size());
  std::array<char, 1 << 16> buf{};
  int ret = MZ_OK;
  do {
    zs.next_out = reinterpret_cast<unsigned char*>(buf.data());
    zs.avail_out = static_cast<unsigned int>(buf.size());
    ret = mz_deflate(&zs, MZ_FINISH);
    out.append(buf.data(), buf.size() - zs.avail_out);
  } while (ret == MZ_OK);
  mz_deflateEnd(&zs);
  const auto crc = static_cast<uint32_t>(mz_crc32(
      MZ_CRC32_INIT,
      reinterpret_cast<const unsigned char*>(data.data()),
      data.size()));
  for (uint32_t trailer : {crc, static_cast<uint32_t>(data.size())}) {
    for (int i = 0; i < 4; ++i) {
      out.push_back(static_cast<char>((trailer >> (8 * i)) & 0xff));
    }
  }
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

// Spread one JSON blob onto a GpuRenderStageEvent as extra_data (name/value
// string pairs), the extra_data analogue of emitJsonBlob: object -> one entry
// per top-level key (value = the raw string for a JSON string, else the compact
// dump()); array / scalar / parse failure -> a single "metadata" entry.
// extra_data has no typed values, so everything is stringified.
void emitJsonExtraData(
    pbz::GpuRenderStageEvent* gse,
    const char* data,
    size_t len) {
  auto parsed = nlohmann::json::parse(data, data + len, nullptr, false);
  if (parsed.is_object()) {
    for (auto it = parsed.begin(); it != parsed.end(); ++it) {
      auto* ed = gse->add_extra_data();
      ed->set_name(it.key());
      const auto& v = it.value();
      ed->set_value(v.is_string() ? v.get<std::string>() : v.dump());
    }
  } else {
    auto* ed = gse->add_extra_data();
    ed->set_name("metadata");
    ed->set_value(
        parsed.is_discarded() ? std::string(data, len) : parsed.dump());
  }
}
} // namespace

std::string cuspyEncodePftrace(
    int64_t base_ns,
    const std::vector<PftraceTrack>& tracks,
    const std::vector<std::string>& name_table,
    const std::vector<std::string>& category_table,
    const std::vector<PftraceGroup>& groups,
    const std::vector<PftraceGpuSpec>& gpu_specs,
    const std::vector<PftraceGfxContext>& gfx_contexts,
    const PftraceRenderStages& stages,
    const PftraceGpuCounter& counters,
    int compression_level) {
  protozero::HeapBuffered<pbz::Trace> trace;

  // Time base: the packet timestamps are absolute (base_ns + offset,
  // epoch-frame ns, as base_ns is epoch-derived). Emit a ClockSnapshot tying
  // the default trace clock (BOOTTIME) to REALTIME at base_ns so
  // trace_processor can recover wall-clock time -- the native counterpart of
  // the JSON's baseTimeNanoseconds, which likewise anchors its relative
  // timestamps to an absolute base.
  {
    auto* pkt = trace->add_packet();
    pkt->set_timestamp(static_cast<uint64_t>(base_ns));
    pkt->set_trusted_packet_sequence_id(1);
    auto* cs = pkt->set_clock_snapshot();
    auto* boot = cs->add_clocks();
    boot->set_clock_id(pbz::BUILTIN_CLOCK_BOOTTIME);
    boot->set_timestamp(static_cast<uint64_t>(base_ns));
    auto* real = cs->add_clocks();
    real->set_clock_id(pbz::BUILTIN_CLOCK_REALTIME);
    real->set_timestamp(static_cast<uint64_t>(base_ns));
  }

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

  if (!name_table.empty() || !category_table.empty() || !gpu_specs.empty() ||
      !gfx_contexts.empty() || !stages.compute_kernels.empty() ||
      !stages.compute_arg_names.empty() || !gpu_counter_iid.empty()) {
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
    // EventCategory table (iid == index + 1): TrackEvent.category_iids
    // references these to tag each CPU slice's kind (cpu_op / cuda_runtime /
    // ...).
    uint64_t cat_iid = 1;
    for (const auto& ct : category_table) {
      auto* ec = interned->add_event_categories();
      ec->set_iid(cat_iid++);
      ec->set_name(ct);
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
      if (g.cat_iid && g.cat_iid[i]) {
        te->add_category_iids(g.cat_iid[i]);
      }
      if (g.flow && g.flow[i]) {
        te->add_flow_ids(static_cast<uint64_t>(g.flow[i]));
      }
      // GpuTrackEvent.gpu_correlation (field 3000): a GpuCorrelation message
      // whose render_stage_submission_event_ids (field 1) lists the
      // render-stage event_ids this slice launched; the viewer links it to
      // those render stages. The extension type is absent from the
      // amalgamation, so emit by field id.
      if (g.gpu_corr_offsets) {
        const int32_t off = g.gpu_corr_offsets[i];
        const int32_t end = g.gpu_corr_offsets[i + 1];
        if (end > off) {
          auto* gc = te->BeginNestedMessage<protozero::Message>(3000);
          for (int32_t j = off; j < end; ++j) {
            gc->AppendVarInt(1, static_cast<uint64_t>(g.gpu_corr_ids[j]));
          }
        }
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
    // event_wait_ids (field 18): the event_ids this stage depends on; the
    // viewer draws them as flow arrows (the CUDA-graph node->node dependency
    // DAG).
    if (stages.event_wait_offsets) {
      const int32_t off = stages.event_wait_offsets[i];
      const int32_t end = stages.event_wait_offsets[i + 1];
      for (int32_t j = off; j < end; ++j) {
        gse->add_event_wait_ids(stages.event_wait_ids[j]);
      }
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
    // Per-row collective-descriptor JSON, spread as extra_data (mirrors the
    // chrome path spreading metadata into the GPU op's args).
    if (stages.meta_offsets) {
      const int32_t off = stages.meta_offsets[i];
      const int32_t len = stages.meta_offsets[i + 1] - off;
      if (len > 0) {
        emitJsonExtraData(
            gse, stages.meta_buffer + off, static_cast<size_t>(len));
      }
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

  return gzipCompress(trace.SerializeAsString(), compression_level);
}

} // namespace torch::profiler::impl

#else // TORCH_CUPTI_PFTRACE

#include <c10/util/Exception.h>

namespace torch::profiler::impl {

std::string cuspyEncodePftrace(
    int64_t /*base_ns*/,
    const std::vector<PftraceTrack>& /*tracks*/,
    const std::vector<std::string>& /*name_table*/,
    const std::vector<std::string>& /*category_table*/,
    const std::vector<PftraceGroup>& /*groups*/,
    const std::vector<PftraceGpuSpec>& /*gpu_specs*/,
    const std::vector<PftraceGfxContext>& /*gfx_contexts*/,
    const PftraceRenderStages& /*stages*/,
    const PftraceGpuCounter& /*counters*/,
    int /*compression_level*/) {
  TORCH_CHECK(
      false,
      "PyTorch was built without native .pftrace support: the CUPTI stubs were "
      "not generated at build time. Export a Chrome trace instead.");
}

} // namespace torch::profiler::impl

#endif // TORCH_CUPTI_PFTRACE
