#include <c10/cuda/CUDAGuard.h>
#include <ATen/cuda/CUDAContext.h>
#include <ATen/native/cuda/MemoryAccess.cuh>
#include <torch/csrc/distributed/c10d/NCCLUtils.hpp>
#include <torch/csrc/distributed/c10d/symm_mem/nccl_dev_cap.hpp>
#include <torch/csrc/distributed/c10d/symm_mem/nccl_extension.hpp>
#include <torch/csrc/distributed/c10d/symm_mem/nccl_devcomm_manager.hpp>
#include <torch/csrc/distributed/c10d/symm_mem/NCCLSymmetricMemory.hpp>
#include <cstdint>

// All-gather a rank-local bucket of parameter shards into a "parameter-
// contiguous" output, fusing the gather with the copy-out reorder that FSDP2
// would otherwise do with split_with_sizes_copy.
//
// Each rank holds its shards of N parameters laid out back-to-back: parameter i
// occupies input[off[i] : off[i]+size[i]] (read locally).  In the output, each
// parameter is stored contiguously across ranks (rather than the standard
// rank-major all-gather layout).  For parameter i and source rank r, the
// gathered output region is:
//   out[off[i]*W + r*size[i] : off[i]*W + (r+1)*size[i]]
// where W is the group size.  Every rank produces the full output.
//
// `out` must be a symmetric-memory tensor (each rank writes its own shard into
// `out` on every rank).  Two write strategies, picked at runtime:
//   - Multimem broadcast (NVLink SHARP): when `out` has multicast support, each
//     rank writes its shard once into the output's multicast mapping and the
//     switch replicates it to every rank (N writes per rank).
//   - LSA push (fallback): each rank writes its shard directly into every
//     peer's `out` window (N*W writes per rank).
//
// `split_sizes[i]` is the per-rank shard size of parameter i.  `split_offsets`
// gives the start offset of each parameter; if omitted it defaults to the
// exclusive prefix sum of `split_sizes` (a packed bucket).
//
// As a first step every slice is required to be 16-byte aligned so the copy can
// use 128-bit vectorized loads/stores unconditionally.

namespace c10d::nccl_extension {

using namespace c10d::symmetric_memory;

// The push kernel uses the symmetric-memory device API (ncclGetLsaPointer, LSA
// barriers); the multimem kernel additionally needs the device reduce-copy API
// (NCCL_DEVICE_HAS_REDUCE_COPY, NCCL >= 2.29.7).
#ifdef NCCL_HAS_SYMMEM_DEVICE_SUPPORT

// Threads per CTA.  The tile floor (AG_MIN_BYTES_PER_TILE) is far larger than
// AG_THREADS_PER_CTA * AG_ALIGN, so every tile keeps all threads busy.
constexpr int AG_THREADS_PER_CTA = 256;
// Upper bound on total CTAs (and thus LSA barrier slots).  All ranks launch
// the same grid, so each barrier slot maps to exactly one CTA per rank.
constexpr int AG_MAX_CTAS = 32;
// Vectorized copy width in bytes (128-bit). All slices are 16-byte aligned.
constexpr int AG_ALIGN = 16;
// Tile size floor, sized for efficient contiguous writes (LSA `st_vec` and
// multimem both want large runs).  A shard is cut into tiles of at least this
// size, so the common cases keep big contiguous writes while a huge shard
// still splits into enough tiles (>= AG_MAX_CTAS once it exceeds
// AG_MAX_CTAS * this) to load-balance a skewed bucket across CTAs.
constexpr int64_t AG_MIN_BYTES_PER_TILE = 2 * 1024 * 1024;
// Max parameters per call.  The schedule is passed by value as a kernel
// argument (copied by the launch, no device-side upload), so it must fit the
// kernel parameter space; 256 keeps it under 8 KB.
constexpr int AG_MAX_PARAMS = 256;

// Per-parameter schedule, passed by value to the kernel.  The work is flattened
// into fixed-size byte tiles: parameter i is cut into first_chunk[i+1] -
// first_chunk[i] tiles, so a large shard yields many tiles and a small one a
// single tile.  CTAs grid-stride over the flat tile space, which load-balances
// across shards of any sizes -- including skewed buckets -- without a per-shard
// grid dimension.  The output base of parameter i is offsets[i] * world_size.
struct AllGatherOffsetSchedule {
  int64_t offsets[AG_MAX_PARAMS];      // logical element offset of each param
  int64_t sizes[AG_MAX_PARAMS];        // per-rank shard size of each param
  int first_chunk[AG_MAX_PARAMS + 1]; // prefix sum of per-param tile counts
};

// Largest p in [0, n_params) with first_chunk[p] <= flat_chunk; i.e. the
// parameter that owns global chunk index flat_chunk.
__device__ __forceinline__ int ag_find_param(
    const int* first_chunk,
    int n_params,
    int flat_chunk) {
  int lo = 0, hi = n_params;
  while (lo + 1 < hi) {
    const int mid = (lo + hi) >> 1;
    if (first_chunk[mid] <= flat_chunk) {
      lo = mid;
    } else {
      hi = mid;
    }
  }
  return lo;
}

// LSA push kernel.  Each CTA grid-strides over the flat (param-chunk, dest-peer)
// tile space, writing this rank's shard bytes of one tile into the parameter-
// contiguous slot of one peer's `out` window with 128-bit copies.
//
// The acquire sync orders against any prior use of `out`; the trailing
// release+acquire guarantees that, on return, every rank has finished its
// writes (release) and observes all peers' writes to its output (acquire).
__global__ void all_gather_offset_push_kernel(
    ncclWindow_t out_window,
    size_t out_window_base_offset,
    const char* input_ptr, // local base pointer of `input`
    AllGatherOffsetSchedule sched,
    int n_params,
    int total_chunks, // total chunks across all shards
    int64_t tile_bytes,
    int my_rank,
    int world_size,
    int elem_size,
    ncclDevComm devComm) {
  const ncclCoopCta coop{};
  ncclLsaBarrierSession<ncclCoopCta> bar{
      coop, devComm, ncclTeamLsa(devComm), devComm.lsaBarrier, blockIdx.x};
  bar.sync(coop, cuda::memory_order_acquire);

  const int64_t total_tiles = static_cast<int64_t>(total_chunks) * world_size;
  // peer is the fast-varying dimension so consecutive CTAs span all peers and
  // keep every outbound NVLink busy; clustering CTAs onto a peer subset (e.g.
  // peer = t / total_chunks when total_chunks < gridDim) leaves links idle.
  for (int64_t t = blockIdx.x; t < total_tiles; t += gridDim.x) {
    const int dst_peer = static_cast<int>(t % world_size);
    const int flat_chunk = static_cast<int>(t / world_size);
    const int param = ag_find_param(sched.first_chunk, n_params, flat_chunk);
    const int chunk = flat_chunk - sched.first_chunk[param];
    const int64_t count = sched.sizes[param];
    const int64_t nbytes = count * elem_size;
    const int64_t tile_off = static_cast<int64_t>(chunk) * tile_bytes;
    const int64_t tile_len =
        (nbytes - tile_off < tile_bytes) ? (nbytes - tile_off) : tile_bytes;
    const char* src =
        input_ptr + sched.offsets[param] * elem_size + tile_off;
    const int64_t out_base = sched.offsets[param] * world_size;
    const size_t dst_byte = out_window_base_offset +
        static_cast<size_t>(out_base + static_cast<int64_t>(my_rank) * count) *
            elem_size +
        static_cast<size_t>(tile_off);
    char* dst = reinterpret_cast<char*>(
        ncclGetLsaPointer(out_window, dst_byte, dst_peer));
    const int64_t nvec = tile_len / AG_ALIGN;
    for (int64_t k = threadIdx.x; k < nvec; k += blockDim.x) {
      const int64_t b = k * AG_ALIGN;
      at::native::memory::st_vec<AG_ALIGN>(
          dst + b, at::native::memory::ld_vec<AG_ALIGN>(src + b));
    }
  }

  // Publish this rank's writes (release) and observe all peers' writes
  // (acquire) in a single barrier round before returning.
  bar.sync(coop, cuda::memory_order_acq_rel);
}

#endif // NCCL_HAS_SYMMEM_DEVICE_SUPPORT

#ifdef NCCL_DEVICE_HAS_REDUCE_COPY

// Multimem broadcast kernel.  Each CTA grid-strides over the flat param-chunk
// tile space and broadcasts one tile of the rank's shard into the output's
// multicast mapping via ncclMultimemCopy; the NVLink switch replicates each
// store to every rank's output.  Data is treated as 32-bit words (every slice
// is 16-byte aligned, hence a multiple of 4 bytes), keeping the kernel
// dtype-agnostic.  ncclMultimemCopy is a CTA-collective, so the tile bounds are
// uniform across the block (they depend only on blockIdx).
__global__ void all_gather_offset_mm_kernel(
    ncclWindow_t out_window,
    size_t out_window_base_offset,
    const char* input_ptr, // local base pointer of `input`
    AllGatherOffsetSchedule sched,
    int n_params,
    int total_chunks, // total chunks across all shards
    int64_t tile_bytes,
    int my_rank,
    int world_size,
    int elem_size,
    ncclMultimemHandle mm_handle,
    ncclDevComm devComm) {
  const ncclCoopCta coop{};
  ncclLsaBarrierSession<ncclCoopCta> bar{
      coop, devComm, ncclTeamLsa(devComm), devComm.lsaBarrier, blockIdx.x};
  bar.sync(coop, cuda::memory_order_acquire);

  for (int flat_chunk = blockIdx.x; flat_chunk < total_chunks; flat_chunk += gridDim.x) {
    const int param = ag_find_param(sched.first_chunk, n_params, flat_chunk);
    const int chunk = flat_chunk - sched.first_chunk[param];
    const int64_t count = sched.sizes[param];
    const int64_t nbytes = count * elem_size;
    const int64_t tile_off = static_cast<int64_t>(chunk) * tile_bytes;
    const int64_t tile_len =
        (nbytes - tile_off < tile_bytes) ? (nbytes - tile_off) : tile_bytes;
    const int64_t words = tile_len / 4;
    const int64_t out_base = sched.offsets[param] * world_size;
    uint32_t* src = reinterpret_cast<uint32_t*>(const_cast<char*>(
        input_ptr + sched.offsets[param] * elem_size + tile_off));
    const size_t dst_byte = out_window_base_offset +
        static_cast<size_t>(out_base + static_cast<int64_t>(my_rank) * count) *
            elem_size +
        static_cast<size_t>(tile_off);
    ncclMultimemCopy(coop, src, out_window, dst_byte, words, mm_handle);
  }

  // Publish this rank's broadcasts (release) and observe all peers' broadcasts
  // (acquire) in a single barrier round before returning.
  bar.sync(coop, cuda::memory_order_acq_rel);
}

#endif // NCCL_DEVICE_HAS_REDUCE_COPY

// Host entry point.  Validates arguments, resolves the default offsets, picks
// the multimem or LSA-push path, fills the schedule, and launches.
void nccl_all_gather_offset(
    const at::Tensor& input,
    at::Tensor& out,
    const std::string& group_name,
    at::IntArrayRef split_sizes,
    std::optional<at::IntArrayRef> split_offsets) {
#ifdef NCCL_HAS_SYMMEM_DEVICE_SUPPORT
  TORCH_CHECK(input.dim() == 1, "nccl_all_gather_offset: input must be 1-D");
  TORCH_CHECK(out.dim() == 1, "nccl_all_gather_offset: out must be 1-D");
  TORCH_CHECK(
      input.is_contiguous(),
      "nccl_all_gather_offset: input must be contiguous");
  TORCH_CHECK(
      out.is_contiguous(), "nccl_all_gather_offset: out must be contiguous");
  TORCH_CHECK(
      out.scalar_type() == input.scalar_type(),
      "nccl_all_gather_offset: out must have the same dtype as input");
  TORCH_CHECK(
      out.device() == input.device(),
      "nccl_all_gather_offset: out must be on the same device as input, got ",
      out.device(), " vs ", input.device());

  const int n_params = static_cast<int>(split_sizes.size());
  TORCH_CHECK(
      n_params > 0, "nccl_all_gather_offset: split_sizes must be non-empty");
  TORCH_CHECK(
      n_params <= AG_MAX_PARAMS,
      "nccl_all_gather_offset: too many parameters: ", n_params,
      " (max ", AG_MAX_PARAMS, "); split the call");

  c10::cuda::CUDAGuard guard(input.device());
  auto stream = at::cuda::getCurrentCUDAStream();
  auto device = input.device();

  // `out` is the gather destination written across ranks, so it must be a
  // symmetric-memory tensor.  `input` is read locally on each rank.
  TORCH_CHECK(
      is_symm_mem_tensor(out),
      "nccl_all_gather_offset: out must be allocated via NCCL symmetric memory "
      "(use empty_strided_p2p with NCCL backend)");
  auto out_symm = c10d::symmetric_memory::rendezvous(out, group_name);
  auto* out_hdl = dynamic_cast<NCCLSymmetricMemory*>(out_symm.get());
  TORCH_CHECK(
      out_hdl != nullptr,
      "nccl_all_gather_offset: requires NCCL symmetric memory backend");

  // Use multimem broadcast when the output supports multicast (NVLink SHARP)
  // and the device reduce-copy API is available; otherwise push over LSA.
  bool use_multimem = false;
#ifdef NCCL_DEVICE_HAS_REDUCE_COPY
  use_multimem = out_hdl->has_multicast_support();
#endif

  auto& manager = c10d::symmetric_memory::NCCLDevCommManager::get(device);
  ncclComm_t comm = manager.get_comm(group_name);

  // Distinct devcomm per path so each is created with the right lsaMultimem
  // requirement; both reuse a cached instance after the first call.
  const char* devcomm_key =
      use_multimem ? "nccl_all_gather_offset_mm" : "nccl_all_gather_offset_lsa";
  auto devcomm_opt = manager.get_devcomm(group_name, devcomm_key);
  if (!devcomm_opt) {
    ncclDevCommRequirements reqs = NCCL_DEV_COMM_REQUIREMENTS_INITIALIZER;
    reqs.lsaBarrierCount = AG_MAX_CTAS;
    reqs.lsaMultimem = use_multimem;
    ncclDevComm devcomm;
    C10D_NCCL_CHECK(
        ncclDevCommCreate(comm, &reqs, &devcomm),
        "ncclDevCommCreate failed in nccl_all_gather_offset");
    devcomm_opt = manager.register_devcomm(group_name, devcomm, devcomm_key);
  }
  ncclDevComm& devcomm = devcomm_opt->get();

  const int world_size = devcomm.nRanks;
  const int my_rank = devcomm.rank;
  const int elem_size = static_cast<int>(input.element_size());

  // Both paths address every rank within a single LSA (NVLink) team -- the
  // multimem multicast and the LSA peer pointers only reach intra-node peers.
  // Multi-node groups (lsaSize < nRanks) are not yet supported.
  TORCH_CHECK(
      devcomm.lsaSize == world_size,
      "nccl_all_gather_offset currently requires all ranks to be in a single "
      "LSA (NVLink) team; got lsaSize=",
      devcomm.lsaSize,
      ", world_size=",
      world_size,
      " (multi-node groups are not yet supported)");

  // Resolve effective offsets: explicit, or exclusive prefix sum of sizes.
  std::vector<int64_t> offsets_vec;
  at::IntArrayRef eff_offsets;
  if (split_offsets.has_value()) {
    eff_offsets = *split_offsets;
    TORCH_CHECK(
        static_cast<int>(eff_offsets.size()) == n_params,
        "nccl_all_gather_offset: split_offsets.size() must match "
        "split_sizes.size()");
  } else {
    offsets_vec.resize(n_params);
    int64_t acc = 0;
    for (int i = 0; i < n_params; i++) {
      offsets_vec[i] = acc;
      acc += split_sizes[i];
    }
    eff_offsets = at::IntArrayRef(offsets_vec);
  }

  // The input base pointer and the output window base are at least 16-byte
  // aligned (the CUDA caching allocator and symmetric memory windows guarantee
  // this); per-slice alignment is then determined by the byte offsets checked
  // below.  `input` is read from input.data_ptr() (already includes any storage
  // offset), so the source offset is simply off[i].
  const int64_t out_numel = out.numel();
  auto out_window = out_hdl->get_window();
  TORCH_CHECK(
      out_window != nullptr, "nccl_all_gather_offset: out window is null");
  const size_t out_window_base_offset = out_hdl->get_offset();
  TORCH_CHECK(
      reinterpret_cast<uintptr_t>(input.data_ptr()) % AG_ALIGN == 0,
      "nccl_all_gather_offset: input must be 16-byte aligned");
  TORCH_CHECK(
      out_window_base_offset % AG_ALIGN == 0,
      "nccl_all_gather_offset: out must be 16-byte aligned in the window");

  // Fill the per-parameter schedule (passed by value to the kernel) and track
  // the largest shard, which sizes the tile.
  AllGatherOffsetSchedule sched;
  int64_t max_bytes = 0;
  for (int i = 0; i < n_params; i++) {
    const int64_t off = eff_offsets[i];
    const int64_t sz = split_sizes[i];
    max_bytes = std::max(max_bytes, sz * elem_size);
    TORCH_CHECK(
        off >= 0 && sz >= 0 && off + sz <= input.numel(),
        "nccl_all_gather_offset: param ", i, " range [", off, ", ", off + sz,
        ") is out of bounds for input of numel ", input.numel());
    const int64_t out_base = off * world_size;
    TORCH_CHECK(
        out_base + static_cast<int64_t>(world_size) * sz <= out_numel,
        "nccl_all_gather_offset: param ", i,
        " output region exceeds out numel ", out_numel);
    // Require every slice to be 16-byte aligned.  The per-rank destination
    // offset is (out_base + r*sz)*elem_size, so checking out_base and sz (which
    // also bounds the per-rank stride and copy length) covers every rank.
    TORCH_CHECK(
        (off * elem_size) % AG_ALIGN == 0 &&
            (out_base * elem_size) % AG_ALIGN == 0 &&
            (sz * elem_size) % AG_ALIGN == 0,
        "nccl_all_gather_offset: param ", i,
        " slices are not 16-byte aligned (offset=", off, ", size=", sz,
        ", elem_size=", elem_size,
        "); all per-parameter offsets and sizes must be 16-byte aligned");
    sched.offsets[i] = off;
    sched.sizes[i] = sz;
  }

  // Tile the work: the largest shard is split into at most AG_MAX_CTAS tiles
  // (so it can use the whole grid) but never below AG_MIN_BYTES_PER_TILE, and
  // every tile boundary stays 16-byte aligned.  Each shard then yields
  // ceil(bytes / tile_bytes) tiles; the prefix sum lets the kernel map a flat
  // tile index back to (param, chunk).
  int64_t tile_bytes = (max_bytes + AG_MAX_CTAS - 1) / AG_MAX_CTAS;
  tile_bytes = std::max(tile_bytes, AG_MIN_BYTES_PER_TILE);
  tile_bytes = ((tile_bytes + AG_ALIGN - 1) / AG_ALIGN) * AG_ALIGN;
  int total_chunks = 0;
  for (int i = 0; i < n_params; i++) {
    sched.first_chunk[i] = total_chunks;
    const int64_t nbytes = sched.sizes[i] * elem_size;
    const int n_chunks =
        std::max<int64_t>(1, (nbytes + tile_bytes - 1) / tile_bytes);
    total_chunks += n_chunks;
  }
  sched.first_chunk[n_params] = total_chunks;

  const char* input_ptr = reinterpret_cast<const char*>(input.data_ptr());

#ifdef NCCL_DEVICE_HAS_REDUCE_COPY
  if (use_multimem) {
    const int n_ctas = std::min(total_chunks, AG_MAX_CTAS);
    all_gather_offset_mm_kernel<<<n_ctas, AG_THREADS_PER_CTA, 0, stream>>>(
        out_window,
        out_window_base_offset,
        input_ptr,
        sched,
        n_params,
        total_chunks,
        tile_bytes,
        my_rank,
        world_size,
        elem_size,
        devcomm.lsaMultimem,
        devcomm);
    C10_CUDA_KERNEL_LAUNCH_CHECK();
    return;
  }
#endif // NCCL_DEVICE_HAS_REDUCE_COPY

  const int64_t total_tiles = static_cast<int64_t>(total_chunks) * world_size;
  const int n_ctas =
      static_cast<int>(std::min<int64_t>(total_tiles, AG_MAX_CTAS));
  all_gather_offset_push_kernel<<<n_ctas, AG_THREADS_PER_CTA, 0, stream>>>(
      out_window,
      out_window_base_offset,
      input_ptr,
      sched,
      n_params,
      total_chunks,
      tile_bytes,
      my_rank,
      world_size,
      elem_size,
      devcomm);
  C10_CUDA_KERNEL_LAUNCH_CHECK();
#else
  TORCH_CHECK(
      false,
      "nccl_all_gather_offset requires NCCL >= 2.28.4 with symmetric memory device API support");
#endif // NCCL_HAS_SYMMEM_DEVICE_SUPPORT
}

} // namespace c10d::nccl_extension
