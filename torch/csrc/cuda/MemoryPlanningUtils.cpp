// CUDA driver and graph calls for torch.cuda.MemoryPlannedGraph, bound to
// Python as thin wrappers: virtual address space and physical memory (through
// c10::cuda::DriverAPI, as expandable segments use them), the capture
// frontier of a stream, and the topology of a graph.
// Addresses and handles are passed as integers.

#include <c10/cuda/CUDAException.h>
#include <c10/util/Exception.h>
#include <torch/csrc/utils/pybind.h>

#if !defined(USE_ROCM)
#include <c10/cuda/driver_api.h>
#include <cuda.h>
#include <cuda_runtime_api.h>
#endif

#include <cstdint>
#include <tuple>
#include <utility>
#include <vector>

namespace torch::cuda::memory_planner {

namespace {

#if !defined(USE_ROCM)

using Handle = uint64_t;

c10::cuda::DriverAPI& driver() {
  return *c10::cuda::DriverAPI::get();
}

// Pinned device memory on `device`, not exportable (NCCL cannot register it).
CUmemAllocationProp deviceMemory(int64_t device) {
  CUmemAllocationProp prop{};
  prop.type = CU_MEM_ALLOCATION_TYPE_PINNED;
  prop.requestedHandleTypes = CU_MEM_HANDLE_TYPE_NONE;
  prop.location.type = CU_MEM_LOCATION_TYPE_DEVICE;
  prop.location.id = static_cast<int>(device);
  return prop;
}

int64_t vmmGranularity(int64_t device) {
  const auto prop = deviceMemory(device);
  size_t granularity = 0;
  C10_CUDA_DRIVER_CHECK(driver().cuMemGetAllocationGranularity_(
      &granularity, &prop, CU_MEM_ALLOC_GRANULARITY_MINIMUM));
  return static_cast<int64_t>(granularity);
}

Handle vmmCreate(size_t nbytes, int64_t device) {
  const auto prop = deviceMemory(device);
  CUmemGenericAllocationHandle handle = 0;
  C10_CUDA_DRIVER_CHECK(driver().cuMemCreate_(&handle, nbytes, &prop, 0));
  return handle;
}

uintptr_t vmmAddressReserve(size_t nbytes) {
  CUdeviceptr ptr = 0;
  C10_CUDA_DRIVER_CHECK(driver().cuMemAddressReserve_(&ptr, nbytes, 0, 0, 0));
  return ptr;
}

void vmmMap(uintptr_t ptr, size_t nbytes, Handle handle) {
  C10_CUDA_DRIVER_CHECK(driver().cuMemMap_(ptr, nbytes, 0, handle, 0));
}

// Read-write access from `device`.
void vmmSetAccess(uintptr_t ptr, size_t nbytes, int64_t device) {
  CUmemAccessDesc access{};
  access.location.type = CU_MEM_LOCATION_TYPE_DEVICE;
  access.location.id = static_cast<int>(device);
  access.flags = CU_MEM_ACCESS_FLAGS_PROT_READWRITE;
  C10_CUDA_DRIVER_CHECK(driver().cuMemSetAccess_(ptr, nbytes, &access, 1));
}

// The releases warn instead of throwing: they run on cleanup paths.
void vmmUnmap(uintptr_t ptr, size_t nbytes) {
  C10_CUDA_DRIVER_CHECK_WARN(driver().cuMemUnmap_(ptr, nbytes), "cuMemUnmap");
}

void vmmAddressFree(uintptr_t ptr, size_t nbytes) {
  C10_CUDA_DRIVER_CHECK_WARN(
      driver().cuMemAddressFree_(ptr, nbytes), "cuMemAddressFree");
}

void vmmRelease(Handle handle) {
  C10_CUDA_DRIVER_CHECK_WARN(driver().cuMemRelease_(handle), "cuMemRelease");
}

// The graph `stream` is capturing into and the nodes that work issued next on
// it would depend on; (0, []) if it is not being captured.
std::pair<uintptr_t, std::vector<uintptr_t>> streamCaptureInfo(
    uintptr_t stream) {
  cudaStreamCaptureStatus status{};
  cudaGraph_t graph = nullptr;
  const cudaGraphNode_t* deps = nullptr;
  // Edge data is requested only so that the query does not fail as lossy on
  // edges that carry it; it is not used: every edge counts as ordering.
  const cudaGraphEdgeData* edge_data = nullptr;
  size_t num_deps = 0;
#if CUDA_VERSION >= 13000
  C10_CUDA_CHECK(cudaStreamGetCaptureInfo(
      reinterpret_cast<cudaStream_t>(stream),
      &status,
      nullptr,
      &graph,
      &deps,
      &edge_data,
      &num_deps));
#else
  C10_CUDA_CHECK(cudaStreamGetCaptureInfo_v3(
      reinterpret_cast<cudaStream_t>(stream),
      &status,
      nullptr,
      &graph,
      &deps,
      &edge_data,
      &num_deps));
#endif
  if (status != cudaStreamCaptureStatusActive) {
    return {0, {}};
  }
  std::vector<uintptr_t> nodes(num_deps);
  for (size_t i = 0; i < num_deps; ++i) {
    nodes[i] = reinterpret_cast<uintptr_t>(deps[i]);
  }
  return {reinterpret_cast<uintptr_t>(graph), std::move(nodes)};
}

size_t graphEdges(
    cudaGraph_t graph,
    cudaGraphNode_t* from,
    cudaGraphNode_t* to,
    cudaGraphEdgeData* edge_data,
    size_t num_edges) {
#if CUDA_VERSION >= 13000
  C10_CUDA_CHECK(cudaGraphGetEdges(graph, from, to, edge_data, &num_edges));
#else
  C10_CUDA_CHECK(cudaGraphGetEdges_v2(graph, from, to, edge_data, &num_edges));
#endif
  return num_edges;
}

// The numbers of nodes and edges of `graph`.
std::pair<size_t, size_t> graphSize(uintptr_t graph) {
  auto g = reinterpret_cast<cudaGraph_t>(graph);
  size_t num_nodes = 0;
  C10_CUDA_CHECK(cudaGraphGetNodes(g, nullptr, &num_nodes));
  return {num_nodes, graphEdges(g, nullptr, nullptr, nullptr, 0)};
}

// The node handles of `graph` and its edges, as pairs of handles.
std::pair<std::vector<uintptr_t>, std::vector<std::pair<uintptr_t, uintptr_t>>>
graphTopology(uintptr_t graph) {
  auto g = reinterpret_cast<cudaGraph_t>(graph);
  size_t num_nodes = 0;
  C10_CUDA_CHECK(cudaGraphGetNodes(g, nullptr, &num_nodes));
  std::vector<cudaGraphNode_t> nodes(num_nodes);
  if (num_nodes != 0) { // (non-null arrays with a count of 0 are rejected)
    C10_CUDA_CHECK(cudaGraphGetNodes(g, nodes.data(), &num_nodes));
  }
  size_t num_edges = graphEdges(g, nullptr, nullptr, nullptr, 0);
  std::vector<cudaGraphNode_t> from(num_edges), to(num_edges);
  // (requested only so that the query does not fail on edges that carry data;
  // not used: every edge counts as ordering)
  std::vector<cudaGraphEdgeData> edge_data(num_edges);
  if (num_edges != 0) {
    num_edges =
        graphEdges(g, from.data(), to.data(), edge_data.data(), num_edges);
  }
  std::vector<uintptr_t> node_ids(num_nodes);
  for (size_t i = 0; i < num_nodes; ++i) {
    node_ids[i] = reinterpret_cast<uintptr_t>(nodes[i]);
  }
  std::vector<std::pair<uintptr_t, uintptr_t>> edges(num_edges);
  for (size_t i = 0; i < num_edges; ++i) {
    edges[i] = {
        reinterpret_cast<uintptr_t>(from[i]),
        reinterpret_cast<uintptr_t>(to[i])};
  }
  return {std::move(node_ids), std::move(edges)};
}

#endif // !defined(USE_ROCM)

} // namespace

void initMemoryPlanningUtilsBindings(PyObject* module) {
#if !defined(USE_ROCM)
  auto m = py::handle(module).cast<py::module>();
  m.def("_cuda_vmmGranularity", &vmmGranularity, py::arg("device"));
  m.def("_cuda_vmmCreate", &vmmCreate, py::arg("nbytes"), py::arg("device"));
  m.def("_cuda_vmmAddressReserve", &vmmAddressReserve, py::arg("nbytes"));
  m.def(
      "_cuda_vmmMap",
      &vmmMap,
      py::arg("ptr"),
      py::arg("nbytes"),
      py::arg("handle"));
  m.def(
      "_cuda_vmmSetAccess",
      &vmmSetAccess,
      py::arg("ptr"),
      py::arg("nbytes"),
      py::arg("device"));
  m.def("_cuda_vmmUnmap", &vmmUnmap, py::arg("ptr"), py::arg("nbytes"));
  m.def(
      "_cuda_vmmAddressFree",
      &vmmAddressFree,
      py::arg("ptr"),
      py::arg("nbytes"));
  m.def("_cuda_vmmRelease", &vmmRelease, py::arg("handle"));
  m.def("_cuda_streamCaptureInfo", &streamCaptureInfo, py::arg("stream"));
  m.def("_cuda_graphSize", &graphSize, py::arg("graph"));
  m.def("_cuda_graphTopology", &graphTopology, py::arg("graph"));
#endif
}

} // namespace torch::cuda::memory_planner
