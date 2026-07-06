#ifdef USE_C10D_TCCL

#include <torch/csrc/distributed/c10d/TCCLUtils.hpp>

#include <cstdlib>
#include <cstring>
#include <dlfcn.h>
#include <ifaddrs.h>
#include <netinet/in.h>
#include <arpa/inet.h>
#include <unistd.h>

#include <infiniband/verbs.h>

#include <c10/util/Exception.h>
#include <c10/util/StringUtil.h>
#include <torch/csrc/distributed/c10d/Store.hpp>
#include <torch/csrc/distributed/c10d/exception.h>

namespace c10d {

// Compile-time check that the layout assumption used by the Store wire
// format holds. ibv_gid is a 16-byte union (per the standard infiniband
// header), so TCCLDestination::gid[16] is the right size and our
// memcpy-based conversion at the verbs boundary is well-defined.
static_assert(
    sizeof(ibv_gid) == 16,
    "ibv_gid is expected to be 16 bytes; TCCLDestination::gid[16] depends "
    "on this for the memcpy bridge.");


// 32-byte layout asserted at compile time. Serialization is reinterpret_cast
// (POD memcpy) so this MUST hold. Catches struct-padding surprises on any
// future platform before bytes go on the wire.
static_assert(
    sizeof(TCCLDestination) == 32,
    "TCCLDestination must be 32 bytes for POD serialization compatibility "
    "across peers. If you changed the struct, also bump the wire format "
    "(see allgatherDestinationsViaStore).");

namespace {

constexpr const char* kInitCounterKey = "tccl_init_counter";
constexpr const char* kInitSeqBroadcastKey = "tccl_init_seq";
constexpr int kQpMaxSendWr = 32;
constexpr int kQpMaxRecvWr = 32;
constexpr int kQpMaxSge = 1;
constexpr int kCqCapacity = 64;
constexpr int kQpPort = 1;
constexpr int kGidIndex = 1;
constexpr int kPsn = 7;  // initial packet sequence number
constexpr int kQpAccessFlags =
    IBV_ACCESS_LOCAL_WRITE | IBV_ACCESS_REMOTE_READ | IBV_ACCESS_REMOTE_WRITE;

// Bridge between our portable TCCLDestination::gid[16] and the verbs-native
// ibv_gid union. Both are 16 bytes (static_assert above guards this).
void gidToBytes(const ibv_gid& src, uint8_t (&dst)[16]) {
  std::memcpy(dst, &src, sizeof(ibv_gid));
}

ibv_gid bytesToGid(const uint8_t (&src)[16]) {
  ibv_gid g{};
  std::memcpy(&g, src, sizeof(ibv_gid));
  return g;
}

// Walk getifaddrs for a BSD interface and return true iff it has at least
// one AF_INET address that is NOT in the 169.254/16 link-local block.

bool ifaceHasStaticIPv4(const std::string& bsd_iface) {
  struct ifaddrs* ifaddr = nullptr;
  if (getifaddrs(&ifaddr) != 0) {
    // Treat as "not found" — the caller will throw with a clear message.
    return false;
  }
  bool found = false;
  for (auto* ifa = ifaddr; ifa != nullptr; ifa = ifa->ifa_next) {
    if (ifa->ifa_name == nullptr || ifa->ifa_addr == nullptr) {
      continue;
    }
    if (bsd_iface != ifa->ifa_name) {
      continue;
    }
    if (ifa->ifa_addr->sa_family != AF_INET) {
      continue;
    }
    auto* sin = reinterpret_cast<sockaddr_in*>(ifa->ifa_addr);
    uint32_t host_addr = ntohl(sin->sin_addr.s_addr);
    // 169.254.0.0/16 — RFC 3927 link-local autoconf. The Thunderbolt RDMA
    // stack hands these out when no static IP is configured; RTR fails with
    // them as the source GID.
    if ((host_addr & 0xFFFF0000u) == 0xA9FE0000u) {
      continue;
    }
    found = true;
    break;
  }
  freeifaddrs(ifaddr);
  return found;
}

} // namespace

// =============================================================================
// TCCLIBVWrapper
// =============================================================================

TCCLIBVWrapper::TCCLIBVWrapper() {
  // RTLD_NOW: resolve all symbols at dlopen time so we fail fast if the
  // library is missing or incompatible.
  // RTLD_GLOBAL: makes librdma symbols available to subsequently dlopen'd
  // libraries
  handle_ = dlopen("librdma.dylib", RTLD_NOW | RTLD_GLOBAL);
  TORCH_CHECK_WITH(
      DistBackendError,
      handle_ != nullptr,
      "TCCL: failed to load librdma.dylib via dlopen: ",
      dlerror(),
      ". Requires macOS 26.2 or later with Thunderbolt RDMA enabled.");

  // Resolve each symbol; complain with the symbol name on failure rather
  // than the generic dlerror text alone.
#define TCCL_LOAD_SYM(var, name)                                              \
  do {                                                                        \
    (var) = reinterpret_cast<decltype(var)>(dlsym(handle_, name));            \
    TORCH_CHECK_WITH(                                                         \
        DistBackendError,                                                     \
        (var) != nullptr,                                                     \
        "TCCL: dlsym(\"",                                                     \
        name,                                                                 \
        "\") returned null: ",                                                \
        dlerror());                                                           \
  } while (0)

  TCCL_LOAD_SYM(get_device_list, "ibv_get_device_list");
  TCCL_LOAD_SYM(get_device_name, "ibv_get_device_name");
  TCCL_LOAD_SYM(open_device, "ibv_open_device");
  TCCL_LOAD_SYM(free_device_list, "ibv_free_device_list");
  TCCL_LOAD_SYM(close_device, "ibv_close_device");
  TCCL_LOAD_SYM(alloc_pd, "ibv_alloc_pd");
  TCCL_LOAD_SYM(create_cq, "ibv_create_cq");
  TCCL_LOAD_SYM(create_qp, "ibv_create_qp");
  TCCL_LOAD_SYM(destroy_qp, "ibv_destroy_qp");
  TCCL_LOAD_SYM(destroy_cq, "ibv_destroy_cq");
  TCCL_LOAD_SYM(dealloc_pd, "ibv_dealloc_pd");
  TCCL_LOAD_SYM(query_port, "ibv_query_port");
  TCCL_LOAD_SYM(query_gid, "ibv_query_gid");
  TCCL_LOAD_SYM(modify_qp, "ibv_modify_qp");
  TCCL_LOAD_SYM(reg_mr, "ibv_reg_mr");
  TCCL_LOAD_SYM(dereg_mr, "ibv_dereg_mr");

#undef TCCL_LOAD_SYM
}

TCCLIBVWrapper& TCCLIBVWrapper::instance() {
  // Lives for the process lifetime; we deliberately do not dlclose — the
  // function pointers we hold would dangle if the library unmapped.
  static TCCLIBVWrapper inst;
  return inst;
}

// =============================================================================
// listRdmaDevices
// =============================================================================

std::vector<std::string> listRdmaDevices() {
  auto& ibv = TCCLIBVWrapper::instance();
  int num_devices = 0;
  ibv_device** devices = ibv.get_device_list(&num_devices);
  if (devices == nullptr || num_devices == 0) {
    if (devices != nullptr) {
      ibv.free_device_list(devices);
    }
    return {};
  }
  std::vector<std::string> names;
  names.reserve(num_devices);
  for (int i = 0; i < num_devices; i++) {
    const char* name = ibv.get_device_name(devices[i]);
    names.emplace_back(name ? name : "");
  }
  ibv.free_device_list(devices);
  return names;
}

std::string resolveTcclDeviceName(const std::string& explicit_name) {
  // Precedence 1: explicit Options::device_name. Trust the caller.
  if (!explicit_name.empty()) {
    return explicit_name;
  }
  // Precedence 2: TCCL_DEVICE env var. Convenient per-rank override for
  // launch scripts that share a single Python entry point but bind each
  // rank to a different device.
  if (const char* env = std::getenv("TCCL_DEVICE");
      env != nullptr && *env != '\0') {
    return std::string(env);
  }
  // Precedence 3: auto-detect. Succeeds iff exactly one device is visible
  // — the common case on Macs with a single Thunderbolt RDMA fabric.
  auto devices = listRdmaDevices();
  TORCH_CHECK_WITH(
      DistBackendError,
      !devices.empty(),
      "TCCL: no RDMA devices visible. Check that macOS is >= 26.2, RDMA "
      "is enabled, and Thunderbolt is unbridged (`sudo tbtrdmactl unbridge`). Run "
      "torch.distributed.list_tccl_devices() for a standalone diagnostic.");
  TORCH_CHECK_WITH(
      DistBackendError,
      devices.size() == 1,
      "TCCL: multiple RDMA devices visible (",
      c10::Join(", ", devices),
      "); cannot auto-select. Set the device explicitly via the "
      "TCCL_DEVICE environment variable or Options.device_name.");
  return devices[0];
}

std::vector<std::string> resolveTcclPeerDevices(
    int rank,
    int size,
    const std::string& explicit_name,
    bool ring_topology) {
  TORCH_CHECK_WITH(
      DistBackendError,
      size > 0 && rank >= 0 && rank < size,
      "TCCL: resolveTcclPeerDevices invalid (rank=",
      rank,
      ", size=",
      size,
      ").");

  const int left = (rank - 1 + size) % size;
  const int right = (rank + 1) % size;
  // Which peers this rank is physically cabled to. Mesh: every non-self peer.
  // Ring: only the two neighbors ((rank±1)%size) — every other slot (incl. self)
  // must be empty, so the sparse row round-trips to null connection slots.
  const auto isConnectedPeer = [&](int p) {
    if (p == rank) {
      return false;
    }
    return ring_topology ? (p == left || p == right) : true;
  };

  std::vector<std::string> devices(static_cast<size_t>(size));

  // Precedence 1: TCCL_PEER_DEVICES — the rank's row of the device matrix.
  // Comma-separated, exactly `size` fields, self-slot empty. The launcher
  // derives it from the auto-discovered hostfile so each peer maps to the
  // physical port (rdma_enX) that is cabled to it. Ring rows are sparse:
  // only the two neighbor fields are non-empty.
  if (const char* env = std::getenv("TCCL_PEER_DEVICES");
      env != nullptr && *env != '\0') {
    const std::string s(env);
    std::vector<std::string> parts;
    size_t start = 0;
    while (true) {
      const size_t comma = s.find(',', start);
      parts.push_back(
          s.substr(start, comma == std::string::npos ? comma : comma - start));
      if (comma == std::string::npos) {
        break;
      }
      start = comma + 1;
    }
    TORCH_CHECK_WITH(
        DistBackendError,
        static_cast<int>(parts.size()) == size,
        "TCCL: TCCL_PEER_DEVICES has ",
        parts.size(),
        " comma-separated entries; expected world_size=",
        size,
        " (self-slot empty). Value: '",
        s,
        "'.");
    for (int p = 0; p < size; p++) {
      if (isConnectedPeer(p)) {
        TORCH_CHECK_WITH(
            DistBackendError,
            !parts[p].empty(),
            "TCCL: TCCL_PEER_DEVICES entry for peer ",
            p,
            " is empty; this peer must name a device (",
            ring_topology ? "ring neighbor" : "mesh peer",
            "). Value: '",
            s,
            "'.");
      } else {
        TORCH_CHECK_WITH(
            DistBackendError,
            parts[p].empty(),
            "TCCL: TCCL_PEER_DEVICES entry for peer ",
            p,
            " must be empty (",
            p == rank ? "self-slot" : "non-neighbor under ring topology",
            "), got '",
            parts[p],
            "'.");
      }
      devices[static_cast<size_t>(p)] = parts[p];
    }
    return devices;
  }

  // Precedence 2: a single device for every connected peer — the single-link
  // case. resolveTcclDeviceName applies its own explicit/env/auto precedence.
  // Under ring topology only the two neighbor slots are filled.
  const std::string one = resolveTcclDeviceName(explicit_name);
  for (int p = 0; p < size; p++) {
    devices[static_cast<size_t>(p)] =
        isConnectedPeer(p) ? one : std::string();
  }
  return devices;
}

// =============================================================================
// TCCLConnection
// =============================================================================

namespace {

// Open the device matching device_name. Returns nullptr if not found.
// Caller owns the returned context (must close_device on it).
ibv_context* openDeviceByName(const std::string& device_name) {
  auto& ibv = TCCLIBVWrapper::instance();
  int num_devices = 0;
  ibv_device** devices = ibv.get_device_list(&num_devices);
  if (devices == nullptr) {
    return nullptr;
  }
  ibv_context* ctx = nullptr;
  for (int i = 0; i < num_devices; i++) {
    const char* name = ibv.get_device_name(devices[i]);
    if (name != nullptr && device_name == name) {
      ctx = ibv.open_device(devices[i]);
      break;
    }
  }
  ibv.free_device_list(devices);
  return ctx;
}

} // namespace

TCCLConnection::TCCLConnection(const std::string& device_name) {
  auto& ibv = TCCLIBVWrapper::instance();

  ctx_ = openDeviceByName(device_name);
  TORCH_CHECK_WITH(
      DistBackendError,
      ctx_ != nullptr,
      "TCCL: failed to open RDMA device '",
      device_name,
      "'. Available devices: see torch.distributed.list_tccl_devices(). ",
      "If empty, Thunderbolt may be bridged — run `sudo tbtrdmactl unbridge`.");

  // PD/CQ/QP allocation: any failure here means we partially constructed —
  // call destroy() to free what we did get before rethrowing.
  try {
    pd_ = ibv.alloc_pd(ctx_);
    TORCH_CHECK_WITH(
        DistBackendError, pd_ != nullptr, "TCCL: ibv_alloc_pd failed.");

    cq_ = ibv.create_cq(
        ctx_,
        kCqCapacity,
        /*cq_context=*/nullptr,
        /*channel=*/nullptr,
        /*comp_vector=*/0);
    TORCH_CHECK_WITH(
        DistBackendError, cq_ != nullptr, "TCCL: ibv_create_cq failed.");

    ibv_qp_init_attr qp_init{};
    qp_init.qp_context = ctx_;
    qp_init.send_cq = cq_;
    qp_init.recv_cq = cq_;
    qp_init.srq = nullptr;
    qp_init.qp_type = IBV_QPT_UC;
    qp_init.cap.max_send_wr = kQpMaxSendWr;
    qp_init.cap.max_recv_wr = kQpMaxRecvWr;
    qp_init.cap.max_send_sge = kQpMaxSge;
    qp_init.cap.max_recv_sge = kQpMaxSge;
    qp_init.cap.max_inline_data = 0;
    qp_init.sq_sig_all = 0;  // Only explicitly-signaled sends get a CQE.

    qp_ = ibv.create_qp(pd_, &qp_init);
    TORCH_CHECK_WITH(
        DistBackendError,
        qp_ != nullptr,
        "TCCL: ibv_create_qp(UC) failed. Most common cause on Thunderbolt: "
        "the device's 10-QP-per-context limit is exhausted (Apple TN3205 §12.1). "
        "Spread peers across more devices or reduce num_wires.");

    // QP -> INIT (the first transition; INIT->RTR happens once the peer's
    // destination has been exchanged via Store).
    {
      ibv_qp_attr attr{};
      attr.qp_state = IBV_QPS_INIT;
      attr.port_num = kQpPort;
      attr.pkey_index = 0;
      attr.qp_access_flags = kQpAccessFlags;
      int mask = IBV_QP_STATE | IBV_QP_PKEY_INDEX | IBV_QP_PORT |
          IBV_QP_ACCESS_FLAGS;
      int ret = ibv.modify_qp(qp_, &attr, mask);
      TORCH_CHECK_WITH(
          DistBackendError,
          ret == 0,
          "TCCL: ibv_modify_qp(QP -> INIT) failed with errno=",
          ret);
    }

    // Populate local routing info now so the caller can publish it via Store
    // immediately after construction. ibv_query_port reports link state
    // (must be ACTIVE) and LID; ibv_query_gid at index 1 returns the
    // IPv4-mapped GID that Thunderbolt uses for addressing.
    ibv_port_attr port_attr{};
    int qp_ret = ibv.query_port(ctx_, kQpPort, &port_attr);
    TORCH_CHECK_WITH(
        DistBackendError,
        qp_ret == 0,
        "TCCL: ibv_query_port(port=",
        kQpPort,
        ") failed with errno=",
        qp_ret);
    TORCH_CHECK_WITH(
        DistBackendError,
        port_attr.state == IBV_PORT_ACTIVE,
        "TCCL: port ",
        kQpPort,
        " on device '",
        device_name,
        "' is not ACTIVE (state=",
        static_cast<int>(port_attr.state),
        "). Check that the Thunderbolt cable is connected and the BSD "
        "interface is up.");

    ibv_gid gid{};
    int gid_ret = ibv.query_gid(ctx_, kQpPort, kGidIndex, &gid);
    TORCH_CHECK_WITH(
        DistBackendError,
        gid_ret == 0,
        "TCCL: ibv_query_gid(port=",
        kQpPort,
        ", index=",
        kGidIndex,
        ") failed with errno=",
        gid_ret);

    local_destination_.lid = port_attr.lid;
    local_destination_.qp_num = qp_->qp_num;
    local_destination_.psn = kPsn;
    local_destination_._pad = 0;
    gidToBytes(gid, local_destination_.gid);
  } catch (...) {
    destroy();
    throw;
  }
}

TCCLConnection::~TCCLConnection() {
  destroy();
}

TCCLConnection::TCCLConnection(TCCLConnection&& other) noexcept {
  swap(other);
}

TCCLConnection& TCCLConnection::operator=(TCCLConnection&& other) noexcept {
  if (this != &other) {
    destroy();
    swap(other);
  }
  return *this;
}

void TCCLConnection::swap(TCCLConnection& other) noexcept {
  std::swap(ctx_, other.ctx_);
  std::swap(pd_, other.pd_);
  std::swap(cq_, other.cq_);
  std::swap(qp_, other.qp_);
  std::swap(local_destination_, other.local_destination_);
}

void TCCLConnection::destroy() noexcept {
  // Teardown order: QP -> CQ -> PD -> context. Each tier depends on the
  // ones below it, so we unwind in reverse-allocation order. Errors are
  // intentionally ignored — there's nothing useful we can do on a teardown
  // failure in a destructor, and throwing from noexcept would call std::terminate.
  auto& ibv = TCCLIBVWrapper::instance();
  if (qp_ != nullptr) {
    ibv.destroy_qp(qp_);
    qp_ = nullptr;
  }
  if (cq_ != nullptr) {
    ibv.destroy_cq(cq_);
    cq_ = nullptr;
  }
  if (pd_ != nullptr) {
    ibv.dealloc_pd(pd_);
    pd_ = nullptr;
  }
  if (ctx_ != nullptr) {
    ibv.close_device(ctx_);
    ctx_ = nullptr;
  }
  local_destination_ = TCCLDestination{};
}

void TCCLConnection::transitionToRTR(const TCCLDestination& remote) {
  TORCH_CHECK_WITH(
      DistBackendError,
      qp_ != nullptr,
      "TCCL: transitionToRTR called on a moved-from / destroyed Connection.");

  ibv_qp_attr attr{};
  attr.qp_state = IBV_QPS_RTR;
  // IBV_MTU_1024 — TN3205 recommends 4096 but it consistently caused RTR
  // timeouts on the Thunderbolt RDMA stack; the hardware appears to negotiate
  // 1024 as the supported MTU.
  attr.path_mtu = IBV_MTU_1024;
  attr.rq_psn = remote.psn;
  attr.dest_qp_num = remote.qp_num;
  attr.ah_attr.dlid = remote.lid;
  attr.ah_attr.sl = 0;
  attr.ah_attr.src_path_bits = 0;
  attr.ah_attr.port_num = kQpPort;
  attr.ah_attr.is_global = 0;

  // Bridge our portable byte array back to ibv_gid for the address-handle.
  ibv_gid remote_gid = bytesToGid(remote.gid);
  // Only set GRH (global routing header) when the remote GID's interface_id
  // is non-zero. On link-local addresses interface_id is 0 and setting GRH
  // confuses the hardware.
  if (remote_gid.global.interface_id != 0) {
    attr.ah_attr.is_global = 1;
    attr.ah_attr.grh.hop_limit = 1;
    attr.ah_attr.grh.dgid = remote_gid;
    attr.ah_attr.grh.sgid_index = kGidIndex;
  }

  int mask = IBV_QP_STATE | IBV_QP_AV | IBV_QP_PATH_MTU | IBV_QP_DEST_QPN |
      IBV_QP_RQ_PSN;
  auto& ibv = TCCLIBVWrapper::instance();
  int ret = ibv.modify_qp(qp_, &attr, mask);
  TORCH_CHECK_WITH(
      DistBackendError,
      ret == 0,
      "TCCL: ibv_modify_qp(QP -> RTR) failed with errno=",
      ret,
      ". remote: lid=",
      remote.lid,
      ", qp_num=",
      remote.qp_num,
      ", psn=",
      remote.psn,
      ". errno=60 (ETIMEDOUT) usually means the link-layer subnet is not "
      "configured — call check_tccl_link_layer() before init, or configure a "
      "static /30 on the Thunderbolt interface.");
}

void TCCLConnection::transitionToRTS() {
  TORCH_CHECK_WITH(
      DistBackendError,
      qp_ != nullptr,
      "TCCL: transitionToRTS called on a moved-from / destroyed Connection.");

  ibv_qp_attr attr{};
  attr.qp_state = IBV_QPS_RTS;
  attr.sq_psn = local_destination_.psn;
  int mask = IBV_QP_STATE | IBV_QP_SQ_PSN;
  auto& ibv = TCCLIBVWrapper::instance();
  int ret = ibv.modify_qp(qp_, &attr, mask);
  TORCH_CHECK_WITH(
      DistBackendError,
      ret == 0,
      "TCCL: ibv_modify_qp(QP -> RTS) failed with errno=",
      ret);
}

// ---- TCCLConnection data-path methods --------------------------------------

void TCCLConnection::postSend(
    const TCCLSharedBuffer& buf,
    uint64_t length,
    uint64_t wr_id) {
  TORCH_CHECK_WITH(
      DistBackendError,
      qp_ != nullptr,
      "TCCL: postSend on a moved-from / destroyed Connection.");

  ibv_sge sge = buf.toSge(pd_, length);

  ibv_send_wr wr{};
  wr.wr_id = wr_id;
  wr.sg_list = &sge;
  wr.num_sge = 1;
  wr.opcode = IBV_WR_SEND;
  wr.send_flags = IBV_SEND_SIGNALED;
  wr.next = nullptr;

  ibv_send_wr* bad = nullptr;
  // ibv_post_send is inline in <infiniband/verbs.h> — calls through
  // qp->context->ops, no syscall (Apple TN3205 §12.5).
  int rc = ibv_post_send(qp_, &wr, &bad);
  TORCH_CHECK_WITH(
      DistBackendError,
      rc == 0,
      "TCCL: ibv_post_send returned ",
      rc,
      " (length=",
      length,
      ", wr_id=",
      wr_id,
      ").");
}

void TCCLConnection::postRecv(
    TCCLSharedBuffer& buf,
    uint64_t length,
    uint64_t wr_id) {
  TORCH_CHECK_WITH(
      DistBackendError,
      qp_ != nullptr,
      "TCCL: postRecv on a moved-from / destroyed Connection.");

  ibv_sge sge = buf.toSge(pd_, length);

  ibv_recv_wr wr{};
  wr.wr_id = wr_id;
  wr.sg_list = &sge;
  wr.num_sge = 1;
  wr.next = nullptr;

  ibv_recv_wr* bad = nullptr;
  int rc = ibv_post_recv(qp_, &wr, &bad);
  TORCH_CHECK_WITH(
      DistBackendError,
      rc == 0,
      "TCCL: ibv_post_recv returned ",
      rc,
      " (length=",
      length,
      ", wr_id=",
      wr_id,
      ").");
}

int TCCLConnection::pollCq(int max_completions, ibv_wc* wcs) {
  TORCH_CHECK_WITH(
      DistBackendError,
      cq_ != nullptr,
      "TCCL: pollCq on a moved-from / destroyed Connection.");
  // ibv_poll_cq is inline; non-blocking; returns -errno on failure.
  int n = ibv_poll_cq(cq_, max_completions, wcs);
  TORCH_CHECK_WITH(
      DistBackendError,
      n >= 0,
      "TCCL: ibv_poll_cq returned errno=",
      -n);
  return n;
}

// ---- TCCLSharedBuffer ------------------------------------------------------

TCCLSharedBuffer::TCCLSharedBuffer(size_t num_bytes) : size_(num_bytes) {
  TORCH_CHECK_WITH(
      DistBackendError,
      num_bytes > 0,
      "TCCL: TCCLSharedBuffer(size=0) is invalid; use the default ctor for "
      "an empty placeholder.");
  long page_size = sysconf(_SC_PAGESIZE);
  TORCH_CHECK_WITH(
      DistBackendError,
      page_size > 0,
      "TCCL: sysconf(_SC_PAGESIZE) returned ",
      page_size);
  int rc = posix_memalign(&data_, static_cast<size_t>(page_size), num_bytes);
  TORCH_CHECK_WITH(
      DistBackendError,
      rc == 0 && data_ != nullptr,
      "TCCL: posix_memalign(align=",
      page_size,
      ", size=",
      num_bytes,
      ") returned ",
      rc);
  std::memset(data_, 0, num_bytes);
}

TCCLSharedBuffer::~TCCLSharedBuffer() {
  destroy();
}

TCCLSharedBuffer::TCCLSharedBuffer(TCCLSharedBuffer&& other) noexcept {
  swap(other);
}

TCCLSharedBuffer& TCCLSharedBuffer::operator=(
    TCCLSharedBuffer&& other) noexcept {
  if (this != &other) {
    destroy();
    swap(other);
  }
  return *this;
}

void TCCLSharedBuffer::swap(TCCLSharedBuffer& other) noexcept {
  std::swap(data_, other.data_);
  std::swap(size_, other.size_);
  mrs_.swap(other.mrs_);
}

void TCCLSharedBuffer::destroy() noexcept {
  // Deregister all MRs before freeing the underlying memory.
  if (!mrs_.empty()) {
    auto& ibv = TCCLIBVWrapper::instance();
    for (auto& kv : mrs_) {
      if (kv.second != nullptr) {
        ibv.dereg_mr(kv.second);
      }
    }
    mrs_.clear();
  }
  if (data_ != nullptr) {
    free(data_);
    data_ = nullptr;
  }
  size_ = 0;
}

void TCCLSharedBuffer::registerToPD(ibv_pd* pd) {
  TORCH_CHECK_WITH(
      DistBackendError,
      data_ != nullptr,
      "TCCL: registerToPD on an empty TCCLSharedBuffer.");
  TORCH_CHECK_WITH(
      DistBackendError,
      pd != nullptr,
      "TCCL: registerToPD called with null pd.");
  // Access flag set (LOCAL_WRITE | REMOTE_READ | REMOTE_WRITE). TN3205 §12.2
  // notes LOCAL_WRITE alone is enough for 2-sided UC; the broader set is also
  // valid and is what this backend uses.
  auto& ibv = TCCLIBVWrapper::instance();
  ibv_mr* mr = ibv.reg_mr(
      pd,
      data_,
      size_,
      IBV_ACCESS_LOCAL_WRITE | IBV_ACCESS_REMOTE_READ |
          IBV_ACCESS_REMOTE_WRITE);
  TORCH_CHECK_WITH(
      DistBackendError,
      mr != nullptr,
      "TCCL: ibv_reg_mr(size=",
      size_,
      ") returned null.");
  auto [it, inserted] = mrs_.emplace(pd, mr);
  if (!inserted) {
    // Already had an MR for this pd — that's a programming error in our
    // caller. Free the new MR and throw.
    ibv.dereg_mr(mr);
    TORCH_CHECK_WITH(
        DistBackendError,
        false,
        "TCCL: TCCLSharedBuffer is already registered to this protection "
        "domain. Call sites must register exactly once per (buffer, pd) "
        "pair at init time.");
  }
}

ibv_sge TCCLSharedBuffer::toSge(ibv_pd* pd, uint64_t length) const {
  TORCH_CHECK_WITH(
      DistBackendError,
      data_ != nullptr,
      "TCCL: toSge on an empty TCCLSharedBuffer.");
  TORCH_CHECK_WITH(
      DistBackendError,
      length > 0 && length <= size_,
      "TCCL: toSge length=",
      length,
      " out of range for buffer size=",
      size_);
  auto it = mrs_.find(pd);
  TORCH_CHECK_WITH(
      DistBackendError,
      it != mrs_.end(),
      "TCCL: TCCLSharedBuffer is not registered to the requested "
      "protection domain. Call registerToPD(pd) first.");
  ibv_sge sge{};
  sge.addr = reinterpret_cast<uintptr_t>(data_);
  sge.length = static_cast<uint32_t>(length);
  sge.lkey = it->second->lkey;
  return sge;
}

void checkLinkLayer(const std::string& rdma_device) {
  // The Thunderbolt RDMA device naming convention is "rdma_" + BSD interface
  // name (rdma_en2, rdma_en4, ...). Strip the prefix to derive the BSD
  // interface we should inspect with getifaddrs.
  TORCH_CHECK_WITH(
      DistBackendError,
      rdma_device.rfind("rdma_", 0) == 0,
      "TCCL: device_name must start with 'rdma_' (e.g. 'rdma_en2'), got '",
      rdma_device,
      "'.");
  std::string bsd_iface = rdma_device.substr(5);
  TORCH_CHECK_WITH(
      DistBackendError,
      !bsd_iface.empty(),
      "TCCL: device_name '",
      rdma_device,
      "' has no BSD interface suffix after 'rdma_'.");

  TORCH_CHECK_WITH(
      DistBackendError,
      ifaceHasStaticIPv4(bsd_iface),
      "TCCL: interface '",
      bsd_iface,
      "' has no non-link-local IPv4 address. Thunderbolt RDMA requires a "
      "static /30 subnet on this interface or ibv_modify_qp(QP -> RTR) "
      "will time out with errno=60 (ETIMEDOUT). Run on each node "
      "(one-time per boot, requires sudo):\n"
      "  sudo ifconfig bridge0 down\n"
      "  sudo ifconfig ",
      bsd_iface,
      " inet 192.168.0.X netmask 255.255.255.252\n"
      "  sudo route change 192.168.0.Y -interface ",
      bsd_iface,
      "\nwhere X is this node's address (e.g. 1) and Y is the peer's "
      "(e.g. 2). The /30 is only on the TB cable and does not affect LAN "
      "routing.");
}

int64_t tcclInitSequence(
    Store& store,
    int rank,
    std::chrono::milliseconds timeout) {
  // All ranks must agree on ONE sequence number for this PG instance. Only
  // rank 0 reserves it via Store::add (a server-side atomic on TCPStore,
  // fresh on every (re)creation) and broadcasts it via
  // the Store; every other rank reads rank 0's value. A per-rank Store::add
  // would return a different post-increment value to each caller, so the
  // ranks would build different key prefixes and never find each other's
  // destinations. Mirrors ProcessGroupNCCL's unique-id broadcast.
  if (rank == 0) {
    const int64_t seq = store.add(kInitCounterKey, 1);
    const std::string s = std::to_string(seq);
    store.set(
        kInitSeqBroadcastKey, std::vector<uint8_t>(s.begin(), s.end()));
    return seq;
  }
  // Non-root ranks block here until rank 0 publishes the number. Bound by the
  // backend timeout instead of the Store's default.
  StoreTimeoutGuard guard(store, timeout);
  const std::vector<uint8_t> bytes = store.get(kInitSeqBroadcastKey);
  return std::stoll(std::string(bytes.begin(), bytes.end()));
}

void allgatherDestinationsViaStore(
    Store& store,
    int rank,
    int size,
    const std::vector<TCCLDestination>& local,
    std::vector<std::vector<TCCLDestination>>& remote,
    const std::string& keyPrefix,
    std::chrono::milliseconds timeout) {
  TORCH_CHECK_WITH(
      DistBackendError,
      rank >= 0 && rank < size,
      "TCCL: rank=",
      rank,
      " out of range for size=",
      size);
  TORCH_CHECK_WITH(
      DistBackendError,
      !local.empty(),
      "TCCL: local destinations vector is empty; expected num_wires >= 1.");

  const size_t bytes_per_rank = local.size() * sizeof(TCCLDestination);

  // Reinterpret_cast to bytes — same pattern as ProcessGroupNCCL.cpp:
  // 2916-2918 for ncclUniqueId. POD, well-defined, single-platform (macOS
  // arm64) on both sides.
  std::vector<uint8_t> myBytes(
      reinterpret_cast<const uint8_t*>(local.data()),
      reinterpret_cast<const uint8_t*>(local.data()) + bytes_per_rank);

  std::vector<std::string> keys;
  keys.reserve(size);
  for (int r = 0; r < size; r++) {
    keys.push_back(keyPrefix + std::to_string(r));
  }

  // Publish OUR destinations before reading peers'. If every rank read
  // before writing, multiGet would block forever on every rank.
  store.set(keys[rank], myBytes);

  // multiGet blocks until all listed keys exist. Single batched round-trip on
  // TCPStore. Use StoreTimeoutGuard to apply our backend timeout instead of the
  // Store's default.
  std::vector<std::vector<uint8_t>> results;
  {
    StoreTimeoutGuard guard(store, timeout);
    try {
      results = store.multiGet(keys);
    } catch (const std::exception& e) {
      // Identify which peers failed to publish so the error message can
      // point at the real culprit. Store::check is non-blocking.
      std::vector<std::string> missing;
      for (int r = 0; r < size; r++) {
        if (r == rank) {
          continue;
        }
        try {
          if (!store.check({keys[r]})) {
            missing.push_back(std::to_string(r));
          }
        } catch (...) {
          // Best-effort diagnostic; ignore secondary failures.
        }
      }
      std::string missingStr =
          missing.empty() ? std::string("(none, store reported error)")
                          : c10::Join(",", missing);
      C10D_THROW_ERROR(
          DistStoreError,
          c10::str(
              "TCCL: timed out waiting for destinations from peers: [",
              missingStr,
              "]. Most likely those peers crashed before reaching the "
              "TCCL bootstrap. Original store error: ",
              e.what()));
    }
  }

  remote.assign(size, {});
  for (int r = 0; r < size; r++) {
    if (r == rank) {
      continue;
    }
    TORCH_CHECK_WITH(
        DistBackendError,
        results[r].size() == bytes_per_rank,
        "TCCL: peer rank ",
        r,
        " published ",
        results[r].size(),
        " bytes; expected ",
        bytes_per_rank,
        " (",
        local.size(),
        " destinations * ",
        sizeof(TCCLDestination),
        " bytes). Most likely a num_wires mismatch between ranks or a "
        "TCCLDestination ABI mismatch between builds.");
    remote[r].assign(local.size(), TCCLDestination{});
    std::memcpy(remote[r].data(), results[r].data(), bytes_per_rank);
  }
}

void tcclRtsBarrier(
    Store& store,
    int size,
    const std::string& key,
    std::chrono::milliseconds timeout) {
  // Store::barrier is the optimized server-side primitive: it combines
  // increment + wait into a single round-trip per rank, and is implemented by
  // every concrete Store (TCPStore, FileStore, HashStore).
  store.barrier(key, static_cast<int64_t>(size), timeout);
}

} // namespace c10d

#endif // USE_C10D_TCCL
