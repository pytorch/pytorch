#include <torch/csrc/profiler/cupti/monitor_python.h>

#include <torch/csrc/profiler/cupti/monitor_native.h>
#include <torch/csrc/profiler/cupti/monitor_pftrace.h>
#include <torch/csrc/utils/pybind.h>

#include <pybind11/numpy.h>

#include <c10/util/ApproximateClock.h>

#include <nlohmann/json.hpp>

#include <cstdint>
#include <optional>
#include <string>
#include <utility>
#include <vector>

namespace torch::profiler::impl {

namespace {
// CUPTI invokes this to stamp records with an approximate timestamp; its
// address is handed to CUPTI on the Python side via the binding below.
uint64_t cuptiApproximateTimeCallback() {
  return c10::getApproximateTime();
}
} // namespace

void initCuptiMonitorBindings(py::module& m) {
  // GIL-free CUPTI monitor bindings, grouped under the
  // torch._C._profiler._cupti_monitor submodule. The two callback addresses are
  // registered with CUPTI on the Python side via ctypes; everything else is
  // driven from the decode thread.
  auto cupti_monitor = m.def_submodule(
      "_cupti_monitor",
      "GIL-free CUPTI monitor buffer pool + v2 record-layout capture.");
  using torch::profiler::impl::CuptiMonitorBuffers;
  cupti_monitor.def("approximate_time_callback_address", []() {
    return reinterpret_cast<uintptr_t>(&cuptiApproximateTimeCallback);
  });
  cupti_monitor.def("configure_buffers", [](size_t buffer_size) {
    CuptiMonitorBuffers::get().configure(buffer_size);
  });
  // The subscriber-scoped (cuptiActivityRegisterCallbacks_v2) buffer callbacks;
  // registered with CUPTI on the Python side via ctypes.
  cupti_monitor.def("buffer_request_callback_address", []() -> uintptr_t {
    return reinterpret_cast<uintptr_t>(
        &torch::profiler::impl::cuptiMonitorBufferRequested);
  });
  cupti_monitor.def("buffer_complete_callback_address", []() -> uintptr_t {
    return reinterpret_cast<uintptr_t>(
        &torch::profiler::impl::cuptiMonitorBufferCompleted);
  });
  // Pop a completed buffer, returning (ptr, valid_size, ctx, stream, layouts)
  // where layouts is the v2 user-defined record layout CUPTI reported for THIS
  // buffer (pBufferCompleteInfo->ppRecordLayouts), as a list of
  // (kind, record_size, [(field_id, offset, size)]). Empty for v1 buffers (and
  // when libcupti did not populate ppRecordLayouts). Returns None on shutdown.
  cupti_monitor.def("get_completed", []() -> py::object {
    std::optional<torch::profiler::impl::CompletedCuptiBuffer> buf;
    {
      py::gil_scoped_release release;
      buf = CuptiMonitorBuffers::get().get_completed();
    }
    if (!buf.has_value()) {
      return py::none();
    }
    py::list layouts;
    for (const auto& layout : buf->layouts) {
      py::list fields;
      for (const auto& field : layout.fields) {
        fields.append(py::make_tuple(field.field_id, field.offset, field.size));
      }
      layouts.append(
          py::make_tuple(layout.kind, layout.record_size, std::move(fields)));
    }
    return py::make_tuple(
        reinterpret_cast<uintptr_t>(buf->ptr),
        buf->valid_size,
        buf->ctx,
        buf->stream,
        std::move(layouts));
  });
  cupti_monitor.def("return_buffer", [](uintptr_t ptr) {
    // NOLINTNEXTLINE(performance-no-int-to-ptr)
    CuptiMonitorBuffers::get().return_buffer(reinterpret_cast<uint8_t*>(ptr));
  });
  cupti_monitor.def("pending_buffers", []() {
    return CuptiMonitorBuffers::get().pending_count();
  });
  cupti_monitor.def("allocated_buffers", []() {
    return CuptiMonitorBuffers::get().allocated_count();
  });
  cupti_monitor.def(
      "shutdown_buffers", []() { CuptiMonitorBuffers::get().shutdown(); });
  cupti_monitor.def(
      "reset_buffers", []() { CuptiMonitorBuffers::get().reset(); });

  // Native decode worker (GIL-free): pulls completed buffers, iterates records
  // with cuptiActivityGetNextRecord_v2 (address passed from Python, which owns
  // the libcupti handle + subscriber), and accumulates per-(kind, field)
  // columns.
  using torch::profiler::impl::CuptiMonitorDecoder;
  cupti_monitor.def(
      "configure_decoder",
      [](uintptr_t subscriber,
         uintptr_t get_next_record_fn,
         uint32_t fence_kind,
         int fence_end_field) {
        CuptiMonitorDecoder::get().configure(
            subscriber, get_next_record_fn, fence_kind, fence_end_field);
      },
      py::arg("subscriber"),
      py::arg("get_next_record_fn"),
      py::arg("fence_kind") = 0,
      py::arg("fence_end_field") = -1);
  // Drop noisy runtime/driver records by cbid in the decoder. filters: {kind:
  // (keep_mode, [cbids])} -- keep_mode True keeps only those cbids (driver
  // allowlist), False drops them (runtime blocklist). cbid_field_id is the cbid
  // field in the record.
  cupti_monitor.def(
      "set_cbid_filter",
      [](int cbid_field_id, const py::dict& filters) {
        std::unordered_map<
            uint32_t,
            std::pair<bool, std::unordered_set<uint32_t>>>
            converted;
        converted.reserve(filters.size());
        for (auto item : filters) {
          auto kind = item.first.cast<uint32_t>();
          auto spec = item.second.cast<py::tuple>();
          bool keep_mode = spec[0].cast<bool>();
          std::unordered_set<uint32_t> cbids;
          for (auto c : spec[1].cast<py::iterable>()) {
            cbids.insert(c.cast<uint32_t>());
          }
          converted.emplace(kind, std::make_pair(keep_mode, std::move(cbids)));
        }
        CuptiMonitorDecoder::get().set_cbid_filter(
            cbid_field_id, std::move(converted));
      },
      py::arg("cbid_field_id"),
      py::arg("filters"));
  cupti_monitor.def(
      "start_decoder", []() { CuptiMonitorDecoder::get().start(); });
  cupti_monitor.def(
      "stop_decoder", []() { CuptiMonitorDecoder::get().stop(); });
  cupti_monitor.def("decoder_max_sync_ns", []() {
    return CuptiMonitorDecoder::get().max_sync_ns();
  });
  // Benchmark entry point: time the native per-buffer decode over a synthetic
  // buffer. record_layouts is [(kind, record_size, [(field_id, offset, size),
  // ...])]
  // -- the same shape as the captured layouts. Returns total seconds for
  // `iters` decodes, run with the GIL released (as the decode worker does).
  cupti_monitor.def(
      "bench_decode",
      [](uintptr_t buffer_addr,
         size_t valid_size,
         const py::list& record_layouts,
         size_t iters) {
        std::vector<torch::profiler::impl::CuptiRecordLayout> layouts;
        layouts.reserve(record_layouts.size());
        for (const auto& entry : record_layouts) {
          auto t = entry.cast<py::tuple>();
          torch::profiler::impl::CuptiRecordLayout layout;
          layout.kind = t[0].cast<uint32_t>();
          layout.record_size = t[1].cast<size_t>();
          auto fields = t[2].cast<py::list>();
          layout.fields.reserve(fields.size());
          for (const auto& f : fields) {
            auto ft = f.cast<py::tuple>();
            layout.fields.push_back(
                {ft[0].cast<int>(),
                 ft[1].cast<size_t>(),
                 ft[2].cast<size_t>()});
          }
          layouts.push_back(std::move(layout));
        }
        py::gil_scoped_release release;
        return torch::profiler::impl::cuptiMonitorBenchDecode(
            buffer_addr, valid_size, layouts, iters);
      });
  cupti_monitor.def("decoder_buffers_decoded", []() {
    return CuptiMonitorDecoder::get().buffers_decoded();
  });
  cupti_monitor.def("decoder_valid_bytes", []() {
    return CuptiMonitorDecoder::get().valid_bytes();
  });
  // Host-side mirror of CUPTI's per-thread external-correlation stack so the
  // current id (what Python pushed) can be read -- CUPTI has push/pop but no
  // peek. The monitor calls note_external_push/pop alongside the CUPTI
  // push/pop; current_external_id() returns the top (0 if none) for consumers
  // on the same thread (e.g. an NCCL profiler plugin keying collective metadata
  // to the id).
  cupti_monitor.def("note_external_push", [](uint64_t external_id) {
    torch::profiler::impl::cuptiMonitorNoteExternalPush(external_id);
  });
  cupti_monitor.def("note_external_pop", []() {
    return torch::profiler::impl::cuptiMonitorNoteExternalPop();
  });
  cupti_monitor.def("current_external_id", []() {
    return torch::profiler::impl::cuptiMonitorCurrentExternalId();
  });
  // Opaque per-annotation metadata store: an in-process NCCL profiler plugin
  // puts a blob (JSON via nlohmann, or any encoding) keyed by the external-
  // correlation id it pushed (see current_external_id); the Python side drains
  // it alongside the decoded records (see drain_decoded) and joins the blob
  // onto the kernel events by id, at the same point the external-correlation ->
  // annotation join happens. Schema-agnostic -- the store never parses the
  // blob. Keyed only by external id: graph_node_ids appear only retroactively
  // in the replayed kernel records, so node-id keying is a consumer-side cache
  // built at first replay, not something a metadata producer can write.
  // metadata_put_external takes a JSON string (parsed to an object, merged into
  // a collective's metadata) and an optional external_id (default 0 == the
  // most- recently-pushed id on this thread, i.e. the collective being issued
  // now). The C++ NCCL plugin is one producer (its descriptor under the current
  // id); this lets Python backends contribute extra schema fields
  // (process_group, sizes,
  // ...), either inside the collective's push window or by passing its id.
  using torch::profiler::impl::CuptiMetadataStore;
  cupti_monitor.def(
      "metadata_put_external",
      [](const std::string& blob, uint64_t external_id) {
        // The id default lives here: 0 -> the most-recently-pushed id on this
        // thread (the collective being issued now). Pass an explicit id to
        // target a specific collective from outside its push window.
        if (external_id == 0) {
          external_id = torch::profiler::impl::cuptiMonitorCurrentExternalId();
        }
        CuptiMetadataStore::get().put_external(
            nlohmann::json::parse(blob), external_id);
      },
      py::arg("blob"),
      py::arg("external_id") = 0);
  // Drain the accumulated decode output as a tuple (groups, external_metadata):
  //  - groups: a list of (kind, {field_id: (field_size, bytes)}). Each entry is
  //    one (kind, layout): a kind whose record layout changed mid-session
  //    appears as multiple entries, so every entry's columns are length-aligned
  //    (the columnar consumers require it). The bytes are the raw little-endian
  //    field values concatenated; Python views them as the field's dtype
  //    (np.frombuffer).
  //  - external_metadata: {external_id: blob} drained from the metadata store,
  //  in
  //    the SAME call so the blobs and the records they annotate come from one
  //    consistent snapshot / one GIL crossing. The join (kernel.correlation_id
  //    -> external_id via the EXTERNAL_CORRELATION columns in `groups` -> blob)
  //    happens Python-side, where all of the window's records are present at
  //    once.
  // Resets both accumulators.
  cupti_monitor.def("drain_decoded", []() -> py::tuple {
    auto groups = CuptiMonitorDecoder::get().drain();
    py::list out;
    for (auto& [kind, kind_cols] : groups) {
      py::dict fields;
      for (auto& [field_id, col] : kind_cols) {
        fields[py::int_(field_id)] = py::make_tuple(
            col.field_size,
            py::bytes(
                reinterpret_cast<const char*>(col.bytes.data()),
                col.bytes.size()));
      }
      out.append(py::make_tuple(py::int_(kind), std::move(fields)));
    }
    py::dict ext_meta;
    for (auto& [id, blob] : CuptiMetadataStore::get().drain_external()) {
      ext_meta[py::int_(id)] =
          py::str(blob.dump()); // merged object -> JSON text
    }
    return py::make_tuple(std::move(out), std::move(ext_meta));
  });

  // Encode the prepared columnar window into a Perfetto-native trace (raw
  // TracePacket stream, uncompressed) via protozero. See monitor_pftrace.h. The
  // Python side (monitor_trace.py) shapes the window and gzips the result.
  //  - tracks: list of (uuid, parent, is_process, pid, tid, name)
  //  - name_table: list of distinct slice names (iid == index + 1)
  //  - groups: list of per-kind groups, each a tuple
  //      (ts, end, track_uuid, name_iid,
  //       int_annos:  [(key, int64_col, skip_zero), ...],
  //       str_annos:  [(key, idx_int64_col, [str, ...]), ...],
  //       arr_annos:  [(key, [int64_col, ...]), ...],
  //       json_annos: [(int32_offsets_col, blob_bytes), ...],
  //       flow: int64_col | None)
  // All columns are numpy arrays; they are kept alive for the encode.
  cupti_monitor.def(
      "encode_pftrace",
      [](const py::list& tracks,
         const py::list& name_table,
         const py::list& groups,
         const py::object& render,
         const py::object& counters) -> py::bytes {
        using torch::profiler::impl::PftraceGfxContext;
        using torch::profiler::impl::PftraceGpuCounter;
        using torch::profiler::impl::PftraceGpuSpec;
        using torch::profiler::impl::PftraceGroup;
        using torch::profiler::impl::PftraceRenderStages;
        using torch::profiler::impl::PftraceTrack;

        constexpr auto kFlags = py::array::c_style | py::array::forcecast;
        using A64 = py::array_t<int64_t, kFlags>;
        using AU64 = py::array_t<uint64_t, kFlags>;
        using A32 = py::array_t<int32_t, kFlags>;
        std::vector<py::object> keepalive;
        auto i64 = [&](const py::handle& h) -> const int64_t* {
          A64 a = py::cast<A64>(h);
          const int64_t* p = a.data();
          keepalive.push_back(std::move(a));
          return p;
        };
        auto u64 = [&](const py::handle& h) -> const uint64_t* {
          AU64 a = py::cast<AU64>(h);
          const uint64_t* p = a.data();
          keepalive.push_back(std::move(a));
          return p;
        };
        auto i32 = [&](const py::handle& h) -> const int32_t* {
          A32 a = py::cast<A32>(h);
          const int32_t* p = a.data();
          keepalive.push_back(std::move(a));
          return p;
        };
        using AF64 = py::array_t<double, kFlags>;
        auto f64 = [&](const py::handle& h) -> const double* {
          AF64 a = py::cast<AF64>(h);
          const double* p = a.data();
          keepalive.push_back(std::move(a));
          return p;
        };
        using AU8 = py::array_t<uint8_t, kFlags>;
        auto u8 = [&](const py::handle& h) -> const uint8_t* {
          if (h.is_none()) {
            return nullptr;
          }
          AU8 a = py::cast<AU8>(h);
          const uint8_t* p = a.data();
          keepalive.push_back(std::move(a));
          return p;
        };

        std::vector<PftraceTrack> track_vec;
        track_vec.reserve(tracks.size());
        for (const auto& item : tracks) {
          auto t = item.cast<py::tuple>();
          track_vec.push_back(
              {t[0].cast<uint64_t>(),
               t[1].cast<uint64_t>(),
               t[2].cast<bool>(),
               t[3].cast<int32_t>(),
               t[4].cast<int32_t>(),
               t[5].cast<std::string>()});
        }
        std::vector<std::string> name_vec;
        name_vec.reserve(name_table.size());
        for (const auto& item : name_table) {
          name_vec.push_back(item.cast<std::string>());
        }

        std::vector<PftraceGroup> group_vec;
        group_vec.reserve(groups.size());
        for (const auto& gobj : groups) {
          auto g = gobj.cast<py::tuple>();
          PftraceGroup pg{};
          pg.ts = i64(g[0]);
          pg.end = i64(g[1]);
          pg.track_uuid = u64(g[2]);
          pg.name_iid = u64(g[3]);
          pg.n = static_cast<size_t>(py::len(g[0]));
          for (const auto& s : g[4].cast<py::list>()) {
            auto t = s.cast<py::tuple>();
            pg.int_annos.push_back(
                {t[0].cast<std::string>(),
                 i64(t[1]),
                 t[2].cast<bool>(),
                 u8(t[3])});
          }
          for (const auto& s : g[5].cast<py::list>()) {
            auto t = s.cast<py::tuple>();
            pg.str_annos.push_back(
                {t[0].cast<std::string>(),
                 i64(t[1]),
                 t[2].cast<std::vector<std::string>>()});
          }
          for (const auto& s : g[6].cast<py::list>()) {
            auto t = s.cast<py::tuple>();
            std::vector<const int64_t*> cols;
            for (const auto& col : t[1].cast<py::list>()) {
              cols.push_back(i64(col));
            }
            pg.arr_annos.push_back({t[0].cast<std::string>(), std::move(cols)});
          }
          for (const auto& s : g[7].cast<py::list>()) {
            auto t = s.cast<py::tuple>();
            const int32_t* offsets = i32(t[0]);
            auto blob = t[1].cast<py::bytes>();
            keepalive.push_back(blob);
            pg.json_annos.push_back({offsets, PyBytes_AS_STRING(blob.ptr())});
          }
          pg.flow = g[8].is_none() ? nullptr : i64(g[8]);
          pg.gpu_corr = py::len(g) > 9 && !g[9].is_none() ? i64(g[9]) : nullptr;
          group_vec.push_back(std::move(pg));
        }

        // render: None, or (gpu_specs, gfx_contexts, stage_cols, extra, launch,
        // tables):
        //   gpu_specs    = [(iid, name, category), ...]
        //   gfx_contexts = [(iid, pid), ...]
        //   stage_cols   = (ts, dur, event_id, gpu_id, hw_queue_iid, stage_iid,
        //                   context) | None
        //   extra        = [(key, int64_col, skip_zero), ...]  (-> extra_data)
        //   launch       = (gx, gy, gz, bx, by, bz, kernel_iid_col, args) |
        //   None
        //                   (args = [(name_iid, int64_col, skip_zero), ...])
        //   tables       = (compute_kernels, compute_arg_names); each
        //                   [(iid, name), ...]
        std::vector<PftraceGpuSpec> spec_vec;
        std::vector<PftraceGfxContext> ctx_vec;
        PftraceRenderStages stages{};
        auto extras_from = [&](const py::handle& h) {
          std::vector<PftraceRenderExtra> v;
          for (const auto& e : h.cast<py::list>()) {
            auto t = e.cast<py::tuple>();
            v.push_back(
                {t[0].cast<std::string>(), i64(t[1]), t[2].cast<bool>()});
          }
          return v;
        };
        auto names_from = [&](const py::handle& h) {
          std::vector<torch::profiler::impl::PftraceComputeName> v;
          for (const auto& e : h.cast<py::list>()) {
            auto t = e.cast<py::tuple>();
            v.push_back({t[0].cast<uint64_t>(), t[1].cast<std::string>()});
          }
          return v;
        };
        if (!render.is_none()) {
          auto r = render.cast<py::tuple>();
          for (const auto& s : r[0].cast<py::list>()) {
            auto t = s.cast<py::tuple>();
            spec_vec.push_back(
                {t[0].cast<uint64_t>(),
                 t[1].cast<std::string>(),
                 t[2].cast<int32_t>()});
          }
          for (const auto& s : r[1].cast<py::list>()) {
            auto t = s.cast<py::tuple>();
            ctx_vec.push_back({t[0].cast<uint64_t>(), t[1].cast<int32_t>()});
          }
          if (!r[2].is_none()) {
            auto sc = r[2].cast<py::tuple>();
            stages.ts = i64(sc[0]);
            stages.dur = i64(sc[1]);
            stages.event_id = u64(sc[2]);
            stages.gpu_id = i64(sc[3]);
            stages.hw_queue_iid = u64(sc[4]);
            stages.stage_iid = u64(sc[5]);
            stages.context = u64(sc[6]);
            stages.name_iid =
                py::len(sc) > 7 && !sc[7].is_none() ? u64(sc[7]) : nullptr;
            stages.n = static_cast<size_t>(py::len(sc[0]));
            stages.extra = extras_from(r[3]);
            // r[6] (optional): const_extra = [(key, value), ...] string args.
            if (py::len(r) > 6 && !r[6].is_none()) {
              for (const auto& e : r[6].cast<py::list>()) {
                auto t = e.cast<py::tuple>();
                stages.const_extra.push_back(
                    {t[0].cast<std::string>(), t[1].cast<std::string>()});
              }
            }
            if (!r[4].is_none()) {
              auto lc = r[4].cast<py::tuple>();
              stages.grid_x = i64(lc[0]);
              stages.grid_y = i64(lc[1]);
              stages.grid_z = i64(lc[2]);
              stages.block_x = i64(lc[3]);
              stages.block_y = i64(lc[4]);
              stages.block_z = i64(lc[5]);
              stages.kernel_iid = u64(lc[6]);
              for (const auto& e : lc[7].cast<py::list>()) {
                auto t = e.cast<py::tuple>();
                stages.launch_args.push_back(
                    {t[0].cast<uint64_t>(), i64(t[1]), t[2].cast<bool>()});
              }
            }
            if (py::len(r) > 5 && !r[5].is_none()) {
              auto tb = r[5].cast<py::tuple>();
              stages.compute_kernels = names_from(tb[0]);
              stages.compute_arg_names = names_from(tb[1]);
            }
          }
        }

        // counters: None, or (specs, gpu_id_col, ts_col, counter_id_col,
        // value_col[, compute_group_ids]) where specs = [(counter_id, name),
        // ...] -> GpuCounterEvents; compute_group_ids (optional) = counter_ids
        // placed in the COMPUTE group for the GPU Compute panel.
        PftraceGpuCounter gpu_counters{};
        if (!counters.is_none()) {
          auto c = counters.cast<py::tuple>();
          for (const auto& s : c[0].cast<py::list>()) {
            auto t = s.cast<py::tuple>();
            gpu_counters.specs.emplace_back(
                t[0].cast<uint32_t>(), t[1].cast<std::string>());
          }
          gpu_counters.gpu_id = i32(c[1]);
          gpu_counters.ts = i64(c[2]);
          gpu_counters.counter_id = i32(c[3]);
          gpu_counters.value = f64(c[4]);
          gpu_counters.n = static_cast<size_t>(py::len(c[2]));
          if (py::len(c) > 5 && !c[5].is_none()) {
            for (const auto& g : c[5].cast<py::list>()) {
              gpu_counters.compute_group.push_back(g.cast<uint32_t>());
            }
          }
          if (py::len(c) > 6 && !c[6].is_none()) {
            for (const auto& g : c[6].cast<py::list>()) {
              gpu_counters.int_value_ids.push_back(g.cast<uint32_t>());
            }
          }
        }

        std::string out;
        {
          py::gil_scoped_release release;
          out = torch::profiler::impl::cuptiMonitorEncodePftrace(
              track_vec,
              name_vec,
              group_vec,
              spec_vec,
              ctx_vec,
              stages,
              gpu_counters);
        }
        return py::bytes(out);
      },
      py::arg("tracks"),
      py::arg("name_table"),
      py::arg("groups"),
      py::arg("render") = py::none(),
      py::arg("counters") = py::none());
}

} // namespace torch::profiler::impl
