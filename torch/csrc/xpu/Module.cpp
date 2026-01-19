#include <ATen/ATen.h>
#include <ATen/xpu/XPUContext.h>
#include <ATen/xpu/XPUGeneratorImpl.h>
#include <c10/xpu/XPUCachingAllocator.h>
#include <c10/xpu/XPUFunctions.h>
#include <torch/csrc/Module.h>
#include <torch/csrc/THP.h>
#include <torch/csrc/profiler/python/combined_traceback.h>
#include <torch/csrc/utils/device_lazy_init.h>
#include <torch/csrc/utils/pycfunction_helpers.h>
#include <torch/csrc/utils/python_numbers.h>
#include <torch/csrc/utils/python_strings.h>
#include <torch/csrc/xpu/Module.h>
#include <torch/csrc/xpu/XPUPluggableAllocator.h>
#include <torch/csrc/xpu/memory_snapshot.h>

using namespace torch;

// XPU management methods

static PyObject* THXPModule_getArchFlags(PyObject* self, PyObject* noargs) {
  HANDLE_TH_ERRORS
#ifdef XPU_ARCH_FLAGS
  static const std::string flags = std::string(C10_STRINGIZE(XPU_ARCH_FLAGS));
  return THPUtils_packString(flags);
#else
  Py_RETURN_NONE;
#endif
  END_HANDLE_TH_ERRORS
}

static PyObject* THXPModule_isInBadFork_wrap(PyObject* self, PyObject* noargs) {
  HANDLE_TH_ERRORS
  return PyBool_FromLong(torch::utils::is_device_in_bad_fork(at::kXPU));
  END_HANDLE_TH_ERRORS
}

static PyObject* THXPModule_setDevice_wrap(PyObject* self, PyObject* arg) {
  HANDLE_TH_ERRORS
  TORCH_CHECK(THPUtils_checkLong(arg), "invalid argument to set_device");

  auto device_index = THPUtils_unpackDeviceIndex(arg);
  c10::xpu::set_device(device_index);

  Py_RETURN_NONE;
  END_HANDLE_TH_ERRORS
}

static PyObject* THXPModule_exchangeDevice_wrap(PyObject* self, PyObject* arg) {
  HANDLE_TH_ERRORS
  TORCH_CHECK(THPUtils_checkLong(arg), "invalid argument to exchange_device");

  auto device_index = THPUtils_unpackDeviceIndex(arg);
  if (device_index < 0) {
    return THPUtils_packInt32(-1);
  }

  torch::utils::device_lazy_init(at::kXPU);
  auto current_device = c10::xpu::exchange_device(device_index);

  return THPUtils_packDeviceIndex(current_device);
  END_HANDLE_TH_ERRORS
}

static PyObject* THXPModule_maybeExchangeDevice_wrap(
    PyObject* self,
    PyObject* arg) {
  HANDLE_TH_ERRORS
  TORCH_CHECK(
      THPUtils_checkLong(arg), "invalid argument to maybe_exchange_device");

  auto device_index = THPUtils_unpackDeviceIndex(arg);
  if (device_index < 0) {
    return THPUtils_packInt32(-1);
  }

  torch::utils::device_lazy_init(at::kXPU);
  auto current_device = c10::xpu::maybe_exchange_device(device_index);

  return THPUtils_packDeviceIndex(current_device);
  END_HANDLE_TH_ERRORS
}

static PyObject* THXPModule_getDevice_wrap(PyObject* self, PyObject* noargs) {
  HANDLE_TH_ERRORS

  auto device_index = c10::xpu::current_device();

  return THPUtils_packDeviceIndex(device_index);
  END_HANDLE_TH_ERRORS
}

static PyObject* THXPModule_getDeviceCount_wrap(
    PyObject* self,
    PyObject* noargs) {
  HANDLE_TH_ERRORS
  // Note: This is distinct from initExtension because a stub xpu implementation
  // has some working functions (e.g. device_count) but cannot fully initialize.
  torch::utils::register_fork_handler_for_device_init(at::kXPU);
  return THPUtils_packUInt64(at::xpu::device_count());
  END_HANDLE_TH_ERRORS
}

static PyObject* THXPModule_getCurrentStream_wrap(
    PyObject* self,
    PyObject* device_index) {
  HANDLE_TH_ERRORS
  TORCH_CHECK(
      THPUtils_checkLong(device_index), "invalid argument to current_stream");
  auto c10_device_index = THPUtils_unpackDeviceIndex(device_index);
  auto stream = at::xpu::getCurrentXPUStream(c10_device_index);
  PyObject* output_tuple = PyTuple_New(3);
  PyTuple_SetItem(
      output_tuple, 0, THPUtils_packInt64(static_cast<int64_t>(stream.id())));
  PyTuple_SetItem(
      output_tuple, 1, THPUtils_packDeviceIndex(stream.device_index()));
  PyTuple_SetItem(
      output_tuple,
      2,
      THPUtils_packInt64(static_cast<int64_t>(stream.device_type())));
  return output_tuple;
  END_HANDLE_TH_ERRORS
}

static PyObject* THXPModule_getCurrentStream_raw(
    PyObject* self,
    PyObject* device_index) {
  HANDLE_TH_ERRORS
  TORCH_CHECK(
      THPUtils_checkLong(device_index),
      "invalid argument to getCurrentRawStream");
  auto c10_device_index = THPUtils_unpackDeviceIndex(device_index);
  return PyLong_FromVoidPtr(
      &at::xpu::getCurrentXPUStream(c10_device_index).queue());
  END_HANDLE_TH_ERRORS
}

static PyObject* THXPModule_setStream_wrap(
    PyObject* self,
    PyObject* args,
    PyObject* kwargs) {
  HANDLE_TH_ERRORS
  int64_t stream_id = 0;
  int64_t device_index = 0;
  int64_t device_type = 0;

  // NOLINTNEXTLINE(modernize-avoid-c-arrays,cppcoreguidelines-avoid-c-arrays)
  constexpr const char* kwlist[] = {
      "stream_id", "device_index", "device_type", nullptr};
  if (!PyArg_ParseTupleAndKeywords(
          args,
          kwargs,
          "|LLL",
          // NOLINTNEXTLINE(cppcoreguidelines-pro-type-const-cast)
          const_cast<char**>(kwlist),
          &stream_id,
          &device_index,
          &device_type)) {
  }

  auto stream = at::xpu::XPUStream::unpack3(
      stream_id,
      static_cast<c10::DeviceIndex>(device_index),
      static_cast<c10::DeviceType>(device_type));

  auto device = c10::xpu::current_device();
  if (device != stream.device_index()) {
    c10::xpu::set_device(stream.device_index());
  }
  at::xpu::setCurrentXPUStream(stream);
  Py_RETURN_NONE;
  END_HANDLE_TH_ERRORS
}

static PyObject* THXPModule_xpuSynchronize(PyObject* self, PyObject* arg) {
  HANDLE_TH_ERRORS
  TORCH_CHECK(THPUtils_checkLong(arg), "invalid argument to synchronize");
  auto device_index = THPUtils_unpackDeviceIndex(arg);
  {
    pybind11::gil_scoped_release no_gil;
    // Only the SYCL queues we have reserved will be synchronized, see Note
    // [Synchronize Streams on Device].
    c10::xpu::syncStreamsOnDevice(device_index);
  }
  Py_RETURN_NONE;
  END_HANDLE_TH_ERRORS
}

static PyObject* THXPModule_emptyCache(PyObject* self, PyObject* noargs) {
  HANDLE_TH_ERRORS
  c10::xpu::XPUCachingAllocator::emptyCache();
  END_HANDLE_TH_ERRORS
  Py_RETURN_NONE;
}

static PyObject* THXPModule_memoryStats(PyObject* self, PyObject* arg) {
  HANDLE_TH_ERRORS
  TORCH_CHECK(THPUtils_checkLong(arg), "invalid argument to memory_stats");
  const auto device_index = THPUtils_unpackDeviceIndex(arg);

  using c10::CachingAllocator::Stat;
  using c10::CachingAllocator::StatArray;
  using c10::CachingAllocator::StatType;
  using c10::CachingDeviceAllocator::DeviceStats;

  const auto statToDict = [](const Stat& stat) {
    py::dict dict;

    dict["current"] = stat.current;
    dict["peak"] = stat.peak;
    dict["allocated"] = stat.allocated;
    dict["freed"] = stat.freed;
    return dict;
  };

  const auto statArrayToDict = [=](const StatArray& statArray) {
    const std::array<const char*, static_cast<size_t>(StatType::NUM_TYPES)>
        statTypeNames = {"all", "small_pool", "large_pool"};
    py::dict dict;
    for (const auto i : c10::irange(statTypeNames.size())) {
      dict[statTypeNames[i]] = statToDict(statArray[i]);
    }
    return dict;
  };

  const DeviceStats stats =
      c10::xpu::XPUCachingAllocator::getDeviceStats(device_index);

  py::dict result;
  result["allocation"] = statArrayToDict(stats.allocation);
  result["segment"] = statArrayToDict(stats.segment);
  result["active"] = statArrayToDict(stats.active);
  result["inactive_split"] = statArrayToDict(stats.inactive_split);
  result["allocated_bytes"] = statArrayToDict(stats.allocated_bytes);
  result["reserved_bytes"] = statArrayToDict(stats.reserved_bytes);
  result["active_bytes"] = statArrayToDict(stats.active_bytes);
  result["inactive_split_bytes"] = statArrayToDict(stats.inactive_split_bytes);
  result["requested_bytes"] = statArrayToDict(stats.requested_bytes);

  return result.release().ptr();
  END_HANDLE_TH_ERRORS
}

static PyObject* THXPModule_resetPeakMemoryStats(
    PyObject* self,
    PyObject* arg) {
  HANDLE_TH_ERRORS
  TORCH_CHECK(
      THPUtils_checkLong(arg), "invalid argument to reset_peak_memory_stats");
  const auto device_index = THPUtils_unpackDeviceIndex(arg);
  c10::xpu::XPUCachingAllocator::resetPeakStats(device_index);
  END_HANDLE_TH_ERRORS
  Py_RETURN_NONE;
}

static PyObject* THXPModule_resetAccumulatedMemoryStats(
    PyObject* self,
    PyObject* arg) {
  HANDLE_TH_ERRORS
  TORCH_CHECK(
      THPUtils_checkLong(arg),
      "invalid argument to reset_accumulated_memory_stats");
  const auto device_index = THPUtils_unpackDeviceIndex(arg);
  c10::xpu::XPUCachingAllocator::resetAccumulatedStats(device_index);
  END_HANDLE_TH_ERRORS
  Py_RETURN_NONE;
}

// XPU module initialization

static void registerXpuDeviceProperties(PyObject* module) {
  // Add _xpuDeviceProperties class to torch._C
  using namespace c10::xpu;
  auto get_device_type = [](const DeviceProp& prop) {
    std::ostringstream stream;
    using namespace sycl::info;
    switch (prop.device_type) {
      case device_type::cpu:
        stream << "cpu";
        break;
      case device_type::gpu:
        stream << "gpu";
        break;
      case device_type::accelerator:
        stream << "accelerator";
        break;
      case device_type::host:
        stream << "host";
        break;
      default:
        stream << "unknown device type:"
               << static_cast<typename std::underlying_type_t<device_type>>(
                      prop.device_type);
        break;
    }
    return stream.str();
  };
  auto gpu_subslice_count = [](const DeviceProp& prop) {
    return (prop.gpu_eu_count / prop.gpu_eu_count_per_subslice);
  };
#if SYCL_COMPILER_VERSION >= 20250000
  auto get_device_architecture = [](const DeviceProp& prop) {
    return static_cast<int64_t>(prop.architecture);
  };
#endif
  // Wrapper class for XPU UUID
  struct XPUuuid {
    XPUuuid(const std::array<unsigned char, 16>& uuid) : bytes(uuid) {}
    const std::array<unsigned char, 16>& bytes{};
  };
  auto m = py::handle(module).cast<py::module>();

  py::class_<XPUuuid>(m, "_XPUuuid")
      .def_property_readonly(
          "bytes",
          [](const XPUuuid& uuid) {
            return std::vector<uint8_t>(uuid.bytes.begin(), uuid.bytes.end());
          })
      .def("__str__", [](const XPUuuid& uuid) {
        return uuid_to_string(reinterpret_cast<const char*>(uuid.bytes.data()));
      });

#define DEFINE_READONLY_MEMBER(member) \
  def_readonly(#member, &DeviceProp::member)

#define THXP_FORALL_DEVICE_PROPERTIES(_)                         \
  py::class_<DeviceProp>(m, "_XpuDeviceProperties")              \
      ._(name)                                                   \
      ._(platform_name)                                          \
      ._(vendor)                                                 \
      ._(device_id)                                              \
      ._(driver_version)                                         \
      ._(version)                                                \
      ._(max_compute_units)                                      \
      ._(gpu_eu_count)                                           \
      ._(max_work_group_size)                                    \
      ._(max_num_sub_groups)                                     \
      ._(sub_group_sizes)                                        \
      ._(has_fp16)                                               \
      ._(has_fp64)                                               \
      ._(has_atomic64)                                           \
      ._(has_bfloat16_conversions)                               \
      ._(has_subgroup_matrix_multiply_accumulate)                \
      ._(has_subgroup_matrix_multiply_accumulate_tensor_float32) \
      ._(has_subgroup_2d_block_io)

  THXP_FORALL_DEVICE_PROPERTIES(DEFINE_READONLY_MEMBER)
      .def_readonly("total_memory", &DeviceProp::global_mem_size)
      .def_property_readonly("gpu_subslice_count", gpu_subslice_count)
#if SYCL_COMPILER_VERSION >= 20250000
      .def_property_readonly("architecture", get_device_architecture)
#endif
      .def_property_readonly("type", get_device_type)
      .def_property_readonly(
          "uuid",
          [](const DeviceProp& prop) -> XPUuuid { return XPUuuid(prop.uuid); })
      .def(
          "__repr__",
          [&get_device_type, &gpu_subslice_count](const DeviceProp& prop) {
            std::ostringstream stream;
            stream << "_XpuDeviceProperties(name='" << prop.name
                   << "', platform_name='" << prop.platform_name << "', type='"
                   << get_device_type(prop) << "', device_id=0x" << std::hex
                   << std::uppercase << prop.device_id << std::dec << ", uuid="
                   << uuid_to_string(
                          reinterpret_cast<const char*>(prop.uuid.data()))
                   << ", driver_version='" << prop.driver_version
                   << "', total_memory="
                   << prop.global_mem_size / (1024ull * 1024) << "MB"
                   << ", max_compute_units=" << prop.max_compute_units
                   << ", gpu_eu_count=" << prop.gpu_eu_count
                   << ", gpu_subslice_count=" << gpu_subslice_count(prop)
                   << ", max_work_group_size=" << prop.max_work_group_size
                   << ", max_num_sub_groups=" << prop.max_num_sub_groups
                   << ", sub_group_sizes=[" << prop.sub_group_sizes
                   << "], has_fp16=" << prop.has_fp16
                   << ", has_fp64=" << prop.has_fp64
                   << ", has_atomic64=" << prop.has_atomic64 << ')';
            return stream.str();
          });
}

static void registerXpuPluggableAllocator(PyObject* module) {
  auto m = py::handle(module).cast<py::module>();

  py::class_<
      c10::xpu::XPUCachingAllocator::XPUAllocator,
      std::shared_ptr<c10::xpu::XPUCachingAllocator::XPUAllocator>>(
      m, "_xpu_XPUAllocator");

  m.def("_xpu_getAllocator", []() {
    return py::cast(torch::xpu::XPUPluggableAllocator::getCurrentAllocator());
  });
  m.def(
      "_xpu_changeCurrentAllocator",
      [](std::shared_ptr<c10::xpu::XPUCachingAllocator::XPUAllocator>
             allocator) {
        torch::xpu::XPUPluggableAllocator::changeCurrentAllocator(allocator);
      });
  m.def("_xpu_customAllocator", [](uint64_t malloc_ptr, uint64_t free_ptr) {
    using MallocFuncType = void*(size_t, int, sycl::queue*);
    using FreeFuncType = void(void*, size_t, int, sycl::queue*);
    std::function<MallocFuncType> malloc_fn =
        reinterpret_cast<MallocFuncType*>(malloc_ptr);
    std::function<FreeFuncType> free_fn =
        reinterpret_cast<FreeFuncType*>(free_ptr);
    return torch::xpu::XPUPluggableAllocator::createCustomAllocator(
        malloc_fn, free_fn);
  });
}

static void bindGetDeviceProperties(PyObject* module) {
  // Add method to torch.xpu
  auto m = py::handle(module).cast<py::module>();
  m.def(
      "_get_device_properties",
      [](c10::DeviceIndex device) -> c10::xpu::DeviceProp* {
        return at::xpu::getDeviceProperties(device);
      },
      py::return_value_policy::reference);
}

static void initXpuMethodBindings(PyObject* module) {
  auto m = py::handle(module).cast<py::module>();
  m.def("_xpu_getMemoryInfo", [](c10::DeviceIndex device_index) {
    py::gil_scoped_release no_gil;
    return at::getDeviceAllocator(at::kXPU)->getMemoryInfo(device_index);
  });
  m.def(
      "_xpu_getStreamFromExternal",
      [](uintptr_t data_ptr, c10::DeviceIndex device_index) {
        sycl::queue* ext_queue =
            // NOLINTNEXTLINE(performance-no-int-to-ptr)
            reinterpret_cast<sycl::queue*>(reinterpret_cast<void*>(data_ptr));
        at::xpu::XPUStream stream =
            c10::xpu::getStreamFromExternal(ext_queue, device_index);
        return std::make_tuple(
            stream.id(), stream.device_index(), stream.device_type());
      });
  m.def(
      "_xpu_canDeviceAccessPeer",
      [](c10::DeviceIndex device, c10::DeviceIndex peer) {
        return at::xpu::canDeviceAccessPeer(device, peer);
      });
  m.def("_xpu_getMemoryFraction", [](c10::DeviceIndex device) {
    return c10::xpu::XPUCachingAllocator::getMemoryFraction(device);
  });
  m.def("_xpu_setMemoryFraction", [](double fraction, c10::DeviceIndex device) {
    c10::xpu::XPUCachingAllocator::setMemoryFraction(fraction, device);
  });
  m.def("_xpu_memorySnapshot", [](std::optional<c10::MempoolId_t> mempool_id) {
    using c10::xpu::XPUCachingAllocator::BlockInfo;
    using c10::xpu::XPUCachingAllocator::SegmentInfo;

    py::str device_s = "device";
    py::str address_s = "address";
    py::str total_size_s = "total_size";
    py::str allocated_size_s = "allocated_size";
    py::str active_size_s = "active_size";
    py::str requested_size_s = "requested_size";
    py::str stream_s = "stream";
    py::str segment_type_s = "segment_type";
    py::str segment_pool_id = "segment_pool_id";
    py::str large_s = "large";
    py::str small_s = "small";
    py::str size_s = "size";
    py::str state_s = "state";
    py::str active_allocated_s = "active_allocated";
    py::str active_pending_free_s = "active_pending_free";
    py::str inactive_s = "inactive";
    py::str addr_s = "addr";
    py::str blocks_s = "blocks";
    py::str is_expandable_s = "is_expandable";
    py::str frames_s = "frames";
    py::str time_us_s = "time_us";

    py::list empty_frames;
    std::vector<CapturedTraceback*> to_gather_frames;
    std::vector<py::dict> to_gather_dest;

    auto add_frame_key = [&](const py::dict& d,
                             const std::shared_ptr<c10::GatheredContext>& ctx) {
      if (ctx) {
        auto sc = getCapturedTracebackFromContext(ctx);
        to_gather_frames.emplace_back(sc);
        to_gather_dest.emplace_back(d);
      } else {
        d[frames_s] = empty_frames;
      }
    };

    const auto segmentInfoToDict = [&](const SegmentInfo& segmentInfo) {
      py::dict segmentDict;
      segmentDict[device_s] = segmentInfo.device;
      segmentDict[address_s] = segmentInfo.address;
      segmentDict[total_size_s] = segmentInfo.total_size;
      segmentDict[allocated_size_s] = segmentInfo.allocated_size;
      segmentDict[active_size_s] = segmentInfo.active_size;
      segmentDict[requested_size_s] = segmentInfo.requested_size;
      // To ensure Python objects can be easily pickled, we represent the stream
      // as an integer rather than as a Stream object.
      segmentDict[stream_s] = reinterpret_cast<uint64_t>(segmentInfo.queue);
      segmentDict[segment_type_s] = (segmentInfo.is_large ? large_s : small_s);
      segmentDict[segment_pool_id] = segmentInfo.owner_private_pool_id;
      segmentDict[is_expandable_s] = segmentInfo.is_expandable;
      add_frame_key(segmentDict, segmentInfo.context_when_allocated);

      auto address = segmentInfo.address;
      py::list blocks;
      for (const auto& blockInfo : segmentInfo.blocks) {
        py::dict blockDict;
        blockDict[address_s] = address;
        blockDict[size_s] = blockInfo.size;
        blockDict[requested_size_s] = blockInfo.requested_size;
        blockDict[state_s] =
            (blockInfo.allocated
                 ? active_allocated_s
                 : (blockInfo.active ? active_pending_free_s : inactive_s));
        add_frame_key(blockDict, blockInfo.context_when_allocated);
        blocks.append(blockDict);
        address += blockInfo.size;
      }
      segmentDict[blocks_s] = blocks;

      return segmentDict;
    };

    auto snapshot = c10::xpu::XPUCachingAllocator::snapshot(
        mempool_id.value_or(std::make_pair(0, 0)));

    py::list segments;

    for (const auto& segmentInfo : snapshot.segments) {
      segments.append(segmentInfoToDict(segmentInfo));
    }

    py::list traces;
    py::str action_s = "action";
    py::str alloc_s = "alloc";
    py::str free_requested_s = "free_requested";
    py::str free_completed_s = "free_completed";
    py::str segment_alloc_s = "segment_alloc";
    py::str segment_free_s = "segment_free";
    py::str segment_map_s = "segment_map";
    py::str segment_unmap_s = "segment_unmap";
    py::str snapshot_s = "snapshot";
    py::str oom_s = "oom";
    py::str device_free_s = "device_free";

    using namespace c10::xpu::XPUCachingAllocator;

    const std::unordered_map<TraceEntry::Action, py::str> action_str_map = {
        {TraceEntry::ALLOC, alloc_s},
        {TraceEntry::FREE_REQUESTED, free_requested_s},
        {TraceEntry::FREE_COMPLETED, free_completed_s},
        {TraceEntry::SEGMENT_ALLOC, segment_alloc_s},
        {TraceEntry::SEGMENT_FREE, segment_free_s},
        {TraceEntry::SEGMENT_MAP, segment_map_s},
        {TraceEntry::SEGMENT_UNMAP, segment_unmap_s},
        {TraceEntry::SNAPSHOT, snapshot_s},
        {TraceEntry::OOM, oom_s},
    };

    auto action_to_str = [&](TraceEntry::Action action) {
      auto it = action_str_map.find(action);
      TORCH_INTERNAL_ASSERT(
          it != action_str_map.end(), "Unknown action type in TraceEntry");
      return it->second;
    };

    for (const auto& traceInfo : snapshot.device_traces) {
      py::list trace;
      for (const auto& te : traceInfo) {
        py::dict trace_entry;
        if (te.context_) {
          auto sc = getCapturedTracebackFromContext(te.context_);
          to_gather_frames.emplace_back(sc);
          to_gather_dest.emplace_back(trace_entry);
        }
        trace_entry[action_s] = action_to_str(te.action_);
        trace_entry[TraceEntry::OOM == te.action_ ? device_free_s : addr_s] =
            te.addr_;
        trace_entry[size_s] = te.size_;
        trace_entry[stream_s] = reinterpret_cast<uint64_t>(te.queue_);
        trace_entry[time_us_s] = te.time_.t_;
        trace.append(trace_entry);
      }
      traces.append(trace);
    }

    py::dict allocator_settings;
    py::str last_allocator_settings_s = "PYTORCH_ALLOC_CONF";
    py::str expandable_segments_s = "expandable_segments";

    allocator_settings[last_allocator_settings_s] =
        snapshot.config_metadata.last_allocator_settings;
    allocator_settings[expandable_segments_s] =
        snapshot.config_metadata.expandable_segments;

    py::dict result;
    result["segments"] = segments;
    result["device_traces"] = traces;
    result["allocator_settings"] = allocator_settings;

    auto frames = py_symbolize(to_gather_frames);
    for (const auto i : c10::irange(frames.size())) {
      to_gather_dest.at(i)[frames_s] = frames.at(i);
    }

    return result;
  });
  m.def("_xpu_recordMemoryHistory", &torch::xpu::_record_memory_history);
}

// Callback for python part. Used for additional initialization of python
// classes
static PyObject* THXPModule_initExtension(PyObject* self, PyObject* noargs) {
  HANDLE_TH_ERRORS
  TORCH_INTERNAL_ASSERT(!torch::utils::is_device_in_bad_fork(at::kXPU));
  torch::utils::register_fork_handler_for_device_init(at::kXPU);
  at::globalContext().lazyInitDevice(c10::DeviceType::XPU);

  auto m = THPObjectPtr(PyImport_ImportModule("torch.xpu"));
  if (!m)
    throw python_error();

  auto set_module_attr = [&](const char* name, PyObject* v) {
    if (PyObject_SetAttrString(m, name, v) < 0) {
      throw python_error();
    }
  };

  auto num_gpus = c10::xpu::device_count();
  THPObjectPtr default_xpu_generators(
      PyTuple_New(static_cast<Py_ssize_t>(num_gpus)));
  for (const auto i : c10::irange(num_gpus)) {
    const auto& gen = at::xpu::detail::getDefaultXPUGenerator(i);
    auto* cast_gen = THPGenerator_initDefaultGenerator(gen);
    PyTuple_SetItem(default_xpu_generators.get(), i, cast_gen);
  }
  set_module_attr("default_generators", default_xpu_generators.get());
  bindGetDeviceProperties(m);

  Py_RETURN_NONE;
  END_HANDLE_TH_ERRORS
}

// NOLINTNEXTLINE(*-c-arrays*, *-global-variables)
static struct PyMethodDef _THXPModule_methods[] = {
    {"_xpu_init", THXPModule_initExtension, METH_NOARGS, nullptr},
    {"_xpu_setDevice", THXPModule_setDevice_wrap, METH_O, nullptr},
    {"_xpu_exchangeDevice", THXPModule_exchangeDevice_wrap, METH_O, nullptr},
    {"_xpu_maybeExchangeDevice",
     THXPModule_maybeExchangeDevice_wrap,
     METH_O,
     nullptr},
    {"_xpu_getDevice", THXPModule_getDevice_wrap, METH_NOARGS, nullptr},
    {"_xpu_getDeviceCount",
     THXPModule_getDeviceCount_wrap,
     METH_NOARGS,
     nullptr},
    {"_xpu_getArchFlags", THXPModule_getArchFlags, METH_NOARGS, nullptr},
    {"_xpu_isInBadFork", THXPModule_isInBadFork_wrap, METH_NOARGS, nullptr},
    {"_xpu_getCurrentStream",
     THXPModule_getCurrentStream_wrap,
     METH_O,
     nullptr},
    {"_xpu_getCurrentRawStream",
     THXPModule_getCurrentStream_raw,
     METH_O,
     nullptr},
    {"_xpu_setStream",
     castPyCFunctionWithKeywords(THXPModule_setStream_wrap),
     METH_VARARGS | METH_KEYWORDS,
     nullptr},
    {"_xpu_synchronize", THXPModule_xpuSynchronize, METH_O, nullptr},
    {"_xpu_emptyCache", THXPModule_emptyCache, METH_NOARGS, nullptr},
    {"_xpu_memoryStats", THXPModule_memoryStats, METH_O, nullptr},
    {"_xpu_resetAccumulatedMemoryStats",
     THXPModule_resetAccumulatedMemoryStats,
     METH_O,
     nullptr},
    {"_xpu_resetPeakMemoryStats",
     THXPModule_resetPeakMemoryStats,
     METH_O,
     nullptr},
    {nullptr}};

PyMethodDef* THXPModule_methods() {
  return _THXPModule_methods;
}

namespace torch::xpu {

void initModule(PyObject* module) {
  registerXpuDeviceProperties(module);
  registerXpuPluggableAllocator(module);
  initXpuMethodBindings(module);
}

} // namespace torch::xpu
