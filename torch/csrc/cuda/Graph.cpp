#include <torch/csrc/python_headers.h>

#include <pybind11/chrono.h>

#include <torch/csrc/jit/python/pybind_utils.h>
#include <torch/csrc/cuda/CUDAPluggableAllocator.h>
#include <torch/csrc/utils/pybind.h>

#include <ATen/cuda/CUDAGraph.h>
#include <c10/cuda/CUDAGraphsC10Utils.h>
#include <c10/cuda/CUDAGuard.h>

#if !defined(USE_ROCM) && !defined(_WIN32)
#include <c10/cuda/driver_api.h>
#endif

#include <algorithm>
#include <mutex>
#include <unordered_map>

// Cargo culted partially from csrc/distributed/c10d/init.cpp
// and partially from csrc/cuda/Stream.cpp.
// THCPStream_init is also declared at global scope.

// Because THCPGraph_init is forward declared in the only consumer
// (csrc/Module.cpp) I don't think we need a Graph.h.

template <typename T>
using shared_ptr_class_ = py::class_<T, std::shared_ptr<T>>;

namespace {

std::shared_ptr<c10::cuda::CUDACachingAllocator::CUDAAllocator>
make_dynamic_cudagraph_allocator() {
#if !defined(USE_ROCM) && !defined(_WIN32)
  struct DynamicCUDAGraphMemoryAllocator {
    ~DynamicCUDAGraphMemoryAllocator() {
      for (const auto& [device, handle] : handles_) {
        CUresult err = c10::cuda::DriverAPI::get()->cuMemRelease_(handle);
        if (err != CUDA_SUCCESS) {
          const char* err_str = nullptr;
          if (c10::cuda::DriverAPI::get()->cuGetErrorString_(err, &err_str) ==
              CUDA_SUCCESS) {
            TORCH_WARN(
                "CUDA driver error releasing dynamic CUDAGraph allocator "
                "handle for device ",
                device,
                ": ",
                err_str);
          }
        }
      }
    }

    static constexpr size_t page_size() {
      return 2 * 1024 * 1024;
    }

    static size_t round_up_to_page(size_t size) {
      size = std::max<size_t>(size, 1);
      return page_size() * ((size + page_size() - 1) / page_size());
    }

    CUmemGenericAllocationHandle handle_for_device(int device) {
      std::lock_guard<std::mutex> lock(mutex_);
      auto it = handles_.find(device);
      if (it != handles_.end()) {
        return it->second;
      }

      CUmemAllocationProp prop{};
      prop.type = CU_MEM_ALLOCATION_TYPE_PINNED;
      prop.location.type = CU_MEM_LOCATION_TYPE_DEVICE;
      prop.location.id = device;

      CUmemGenericAllocationHandle handle{};
      C10_CUDA_DRIVER_CHECK(c10::cuda::DriverAPI::get()->cuMemCreate_(
          &handle, page_size(), &prop, 0ULL));
      handles_.emplace(device, handle);
      return handle;
    }

    void* malloc(size_t size, int device, cudaStream_t /*stream*/) {
      c10::cuda::OptionalCUDAGuard device_guard(device);
      size_t rounded_size = round_up_to_page(size);
      CUmemGenericAllocationHandle handle = handle_for_device(device);

      CUdeviceptr ptr = 0;
      C10_CUDA_DRIVER_CHECK(
          c10::cuda::DriverAPI::get()->cuMemAddressReserve_(
              &ptr, rounded_size, 0ULL, 0, 0ULL));

      size_t mapped_size = 0;
      try {
        for (; mapped_size < rounded_size; mapped_size += page_size()) {
          C10_CUDA_DRIVER_CHECK(c10::cuda::DriverAPI::get()->cuMemMap_(
              ptr + mapped_size, page_size(), 0ULL, handle, 0ULL));
        }

        CUmemAccessDesc desc{};
        desc.location.type = CU_MEM_LOCATION_TYPE_DEVICE;
        desc.location.id = device;
        desc.flags = CU_MEM_ACCESS_FLAGS_PROT_READWRITE;
        C10_CUDA_DRIVER_CHECK(c10::cuda::DriverAPI::get()->cuMemSetAccess_(
            ptr, rounded_size, &desc, 1));
      } catch (...) {
        if (mapped_size > 0) {
          c10::cuda::DriverAPI::get()->cuMemUnmap_(ptr, mapped_size);
        }
        c10::cuda::DriverAPI::get()->cuMemAddressFree_(ptr, rounded_size);
        throw;
      }

      return reinterpret_cast<void*>(ptr);
    }

    void free(void* ptr, size_t size, int device, cudaStream_t stream) {
      if (ptr == nullptr) {
        return;
      }

      c10::cuda::OptionalCUDAGuard device_guard(device);
      cudaStreamCaptureStatus status{};
      c10::CaptureId_t capture_id{};
      C10_CUDA_CHECK(cudaStreamGetCaptureInfo(stream, &status, &capture_id));
      TORCH_INTERNAL_ASSERT(
          status != cudaStreamCaptureStatusActive,
          "Dynamic CUDAGraph allocator memory must not be freed during CUDA "
          "graph capture");

      size_t rounded_size = round_up_to_page(size);
      CUdeviceptr base = reinterpret_cast<CUdeviceptr>(ptr);
      C10_CUDA_DRIVER_CHECK(
          c10::cuda::DriverAPI::get()->cuMemUnmap_(base, rounded_size));
      C10_CUDA_DRIVER_CHECK(
          c10::cuda::DriverAPI::get()->cuMemAddressFree_(base, rounded_size));
    }

    std::mutex mutex_;
    std::unordered_map<int, CUmemGenericAllocationHandle> handles_;
  };

  auto allocator_state = std::make_shared<DynamicCUDAGraphMemoryAllocator>();
  return torch::cuda::CUDAPluggableAllocator::createCustomAllocator(
      [allocator_state](size_t size, int device, cudaStream_t stream) {
        return allocator_state->malloc(size, device, stream);
      },
      [allocator_state](
          void* ptr, size_t size, int device, cudaStream_t stream) {
        allocator_state->free(ptr, size, device, stream);
      });
#else
  TORCH_CHECK(
      false,
      "Dynamic CUDAGraph memory allocator requires CUDA driver API support");
  return nullptr;
#endif
}

} // namespace

void THCPGraph_init(PyObject* module) {
  // Pybind11 patch notes say "py::module_" is more up-to-date syntax,
  // but CI linter and some builds prefer "module".
  auto torch_C_m = py::handle(module).cast<py::module>();

  torch_C_m.def("_graph_pool_handle", &::at::cuda::graph_pool_handle);
  torch_C_m.def(
      "_cuda_graph_apply_device_kernel_node_updates",
      torch::wrap_pybind_function_no_gil(
          &::at::cuda::apply_device_kernel_node_updates),
      py::arg("device_nodes"),
      py::arg("param_offsets"),
      py::arg("alloc_indices"),
      py::arg("alloc_offsets"),
      py::arg("dynamic_tensors"));

  shared_ptr_class_<::at::cuda::CUDAGraph>(torch_C_m, "_CUDAGraph")
      .def(py::init<bool>(), py::arg("keep_graph") = false)
      .def(
          "capture_begin",
          [](::at::cuda::CUDAGraph& self,
             std::optional<c10::cuda::MempoolId_t> pool_opt,
             const std::string& capture_error_mode) {
            cudaStreamCaptureMode capture_mode{};
            c10::cuda::MempoolId_t pool = pool_opt.has_value()
                ? pool_opt.value()
                : c10::cuda::MempoolId_t{0, 0};
            if (capture_error_mode == "global") {
              capture_mode = cudaStreamCaptureModeGlobal;
            } else if (capture_error_mode == "thread_local") {
              capture_mode = cudaStreamCaptureModeThreadLocal;
            } else if (capture_error_mode == "relaxed") {
              capture_mode = cudaStreamCaptureModeRelaxed;
            } else {
              TORCH_CHECK(
                  false,
                  "Unknown capture error mode. Expected `global`, `thread_local`, or `relaxed`, got ",
                  capture_error_mode);
            }
            return self.capture_begin(pool, capture_mode);
          },
          py::arg("pool"),
          py::arg("capture_error_mode"),
          py::call_guard<py::gil_scoped_release>())
      .def(
          "capture_end",
          torch::wrap_pybind_function_no_gil(&at::cuda::CUDAGraph::capture_end))
      .def(
          "instantiate",
          torch::wrap_pybind_function_no_gil(&at::cuda::CUDAGraph::instantiate))
      .def(
          "register_generator_state",
          [](::at::cuda::CUDAGraph& self, py::handle raw_generator) {
            auto generator = THPGenerator_Unwrap(raw_generator.ptr());
            // We've unwrapped Python object to C++ object,
            // so we could release GIL before calling into C++
            py::gil_scoped_release release;
            return self.register_generator_state(generator);
          },
          py::arg("generator"))
      .def(
          "replay",
          torch::wrap_pybind_function_no_gil(&at::cuda::CUDAGraph::replay))
      .def(
          "release_pool_memory",
          torch::wrap_pybind_function_no_gil(
              &at::cuda::CUDAGraph::release_pool_memory))
      .def(
          "get_mem_allocator",
          [](::at::cuda::CUDAGraph&) {
            return make_dynamic_cudagraph_allocator();
          },
          py::call_guard<py::gil_scoped_release>())
      .def(
          "reset",
          torch::wrap_pybind_function_no_gil(&at::cuda::CUDAGraph::reset))
      .def(
          "pool",
          torch::wrap_pybind_function_no_gil(&at::cuda::CUDAGraph::pool))
      .def(
          "debug_dump",
          torch::wrap_pybind_function_no_gil(
              &::at::cuda::CUDAGraph::debug_dump))
      .def(
          "enable_debug_mode",
          torch::wrap_pybind_function_no_gil(
              &::at::cuda::CUDAGraph::enable_debug_mode))
      .def(
          "debug_dump",
          torch::wrap_pybind_function_no_gil(
              &::at::cuda::CUDAGraph::debug_dump),
          py::arg("debug_path"))
      .def(
          "raw_cuda_graph",
          [](::at::cuda::CUDAGraph& self) {
            cudaGraph_t graph = self.raw_cuda_graph();
            // We return a raw int here, since otherwise pybind11 will
            // try to return the underlying struct of cudaGraph_t
            // points to, which is opaque and therefore causes a
            // compile error.
            return reinterpret_cast<uintptr_t>(graph);
          },
          py::call_guard<py::gil_scoped_release>())
      .def(
          "raw_cuda_graph_exec",
          [](::at::cuda::CUDAGraph& self) {
            cudaGraphExec_t graph_exec = self.raw_cuda_graph_exec();
            // We return a raw int here, since otherwise pybind11 will
            // try to return the underlying struct of cudaGraphExec_t
            // points to, which is opaque and therefore causes a
            // compile error.
            return reinterpret_cast<uintptr_t>(graph_exec);
          },
          py::call_guard<py::gil_scoped_release>())
      .def_static(
          "get_currently_capturing_graph",
          torch::wrap_pybind_function_no_gil(
              &::at::cuda::CUDAGraph::get_currently_capturing_graph),
          py::return_value_policy::reference)
      .def(
          "begin_capture_to_if_node",
          torch::wrap_pybind_function_no_gil(
              &::at::cuda::CUDAGraph::begin_capture_to_if_node),
          py::arg("scalar_cuda_pred_tensor"))
      .def(
          "end_capture_to_conditional_node",
          torch::wrap_pybind_function_no_gil(
              &::at::cuda::CUDAGraph::end_capture_to_conditional_node));
}
