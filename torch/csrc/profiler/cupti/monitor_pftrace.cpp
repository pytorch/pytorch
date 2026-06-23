#include <torch/csrc/profiler/cupti/monitor_pftrace.h>

#include <perfetto.h>

#include <nlohmann/json.hpp>

namespace torch::profiler::impl {

namespace {
namespace pbz = perfetto::protos::pbzero;

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
    const std::vector<PftraceGroup>& groups) {
  protozero::HeapBuffered<pbz::Trace> trace;

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

  if (!name_table.empty()) {
    auto* pkt = trace->add_packet();
    pkt->set_trusted_packet_sequence_id(1);
    pkt->set_sequence_flags(pbz::TracePacket::SEQ_INCREMENTAL_STATE_CLEARED);
    auto* interned = pkt->set_interned_data();
    uint64_t iid = 1;
    for (const auto& nm : name_table) {
      auto* en = interned->add_event_names();
      en->set_iid(iid++);
      en->set_name(nm);
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

  return trace.SerializeAsString();
}

} // namespace torch::profiler::impl
