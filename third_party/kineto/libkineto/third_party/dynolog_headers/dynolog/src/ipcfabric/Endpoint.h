// Copyright (c) Meta Platforms, Inc. and affiliates.
//
// This source code is licensed under the MIT license found in the
// LICENSE file in the root directory of this source tree.

#include <asm/unistd.h>
#include <stdlib.h>
#include <sys/socket.h>
#include <sys/stat.h>
#include <sys/un.h>
#include <unistd.h>
#include <stdexcept>

#include <cstring>
#include <memory>
#include <string>
#include <vector>

// @lint-ignore-every CLANGTIDY facebook-hte-BadCall-strerror

/*
 * Design: We are using the following components:
 * 1) Use UNIX sockets and sendmsg/rcvmsg to pass ancillary control message and
 * payloads. see https://sarata.com/manpages/CMSG_ALIGN.3.html 2) Use abstract
 * sockets (https://man7.org/linux/man-pages/man7/unix.7.html) to avoid choosing
 * a path. Note that abstract sockets are indicated by pathname[0]='\0' and the
 * actual name in the remaining characters up to struct size + null terminator.
 * 3) Use connectionless datagram sockets
 *    (SOCK_DGRM https://man7.org/linux/man-pages/man2/socket.2.html) - in Linux
 *    they are guaranteed to be always reliable and don't reorder - to avoid
 * having to mantain multiple connections. 4) Use a stateless protocol
 * connecting based on destination socket name passed in, using scatter/gather
 * vectors for data communication, with user input vector sizes. 5) Non-blocking
 * send/receive. 6) Keep serialization/deserialization simple by only supporting
 * class that are trivially copyable (i.e. std::is_trivially_copyable), like
 * Plain Old Data (POD) classes.
 *    (https://en.cppreference.com/w/cpp/types/is_trivially_copyable)
 *
 *  You can list local domain sockets using-
 *    netstat -a -p --unix on Linux
 */

namespace dynolog {
namespace ipcfabric {

[[noreturn]] inline void terminate_with_message(const std::string& message) {
  std::fputs(message.c_str(), stderr);
  std::terminate();
}

// Define a type for fds to improve readibility.
using fd_t = int;

struct Payload {
  Payload(size_t size, void* data) : size(size), data(data) {}
  size_t size;
  void* data;
};

template <size_t kMaxNumFds = 0>
struct EndPointCtxt {
  explicit EndPointCtxt(size_t n) : iov{std::vector<struct iovec>(n)} {}
  struct sockaddr_un msg_name;
  size_t msg_namelen;
  struct msghdr msghdr;
  std::vector<struct iovec> iov;
  fd_t* ctrl_fd_ptr;

  // Ancillary data buffer sized to contain kMaxNumFds in data section.
  char ancillary_buf[CMSG_SPACE(kMaxNumFds * sizeof(fd_t))];
};

template <size_t kMaxNumFds = 0>
class EndPoint final {
  using TCtxt = EndPointCtxt<kMaxNumFds>;

 public:
  // Maximum defined in man unix, but minus 2 because abstract sockets, first
  // and last are '\0'.
  constexpr static size_t kMaxNameLen = 108 - 2;

  explicit EndPoint(const std::string& address) {
    socket_fd_ = socket(AF_UNIX, SOCK_DGRAM, 0);
    if (socket_fd_ == -1) {
      throw std::runtime_error(std::strerror(errno));
    }
    struct sockaddr_un addr;
    size_t addrlen = setAddress_(address, addr);
    if (addr.sun_path[0] != '\0') {
      // delete domain socket file just in case before binding
      unlink(addr.sun_path);
    }

    int ret = bind(socket_fd_, (const struct sockaddr*)&addr, addrlen);
    if (ret == -1) {
      throw std::runtime_error(std::strerror(errno));
    }
    if (addr.sun_path[0] != '\0') {
      // set correct permissions for the socket. We avoid using umask because
      // of multithreaded nature. A short window exists between bind and chmod
      // where the permissions are wrong but it's ok for our use case.
      chmod(addr.sun_path, 0666);
    }
  }

  ~EndPoint() {
    close(socket_fd_);
  }

  [[nodiscard]] auto buildSendCtxt(
      const std::string& dest_name,
      const std::vector<Payload>& payload,
      const std::vector<fd_t>& fds) {
    if (fds.size() > kMaxNumFds) {
      throw std::invalid_argument("Requested to fill more than kMaxNumFds");
    }
    if (dest_name.size() == 0) {
      throw std::invalid_argument("Cannot send to empty socket name");
    }

    auto ctxt = buildCtxt_(payload, fds.size());
    ctxt->msghdr.msg_namelen = setAddress_(dest_name, ctxt->msg_name);
    if (fds.size()) {
      memcpy(ctxt->ctrl_fd_ptr, fds.data(), fds.size() * sizeof(int));
    }
    return ctxt;
  }

  // non-blocking. The error ECONNREFUSED may be caused by socket not being yet
  // initialized. See man unix.
  [[nodiscard]] bool trySendMsg(
      TCtxt const& ctxt,
      bool retryOnConnRefused = true) {
    ssize_t ret = sendmsg(socket_fd_, &ctxt.msghdr, MSG_DONTWAIT);
    if (ret > 0) { // XXX: Enforce correct number of bytes sent.
      return true;
    }
    if (ret == -1 && (errno == EAGAIN || errno == EWOULDBLOCK)) {
      return false;
    }
    if (ret == -1 && retryOnConnRefused && errno == ECONNREFUSED) {
      return false;
    }
    throw std::runtime_error(std::strerror(errno));
  }

  [[nodiscard]] auto buildRcvCtxt(const std::vector<Payload>& payload) {
    return buildCtxt_(payload, kMaxNumFds);
  }

  // If false, must retry. Only enabled for bound sockets.
  [[nodiscard]] bool tryRcvMsg(TCtxt& ctxt) noexcept {
    ssize_t ret = recvmsg(socket_fd_, &ctxt.msghdr, MSG_DONTWAIT);

    if (ret > 0) { // XXX: Enforce correct number of bytes sent.
      return true;
    }
    if (ret == 0) {
      return false; // Receiver is down.
    }
    if (errno == EAGAIN || errno == EWOULDBLOCK) {
      return false;
    }

    terminate_with_message(
        "tryRcvMsg() got " + std::string(std::strerror(errno)));
  }

  [[nodiscard]] bool tryPeekMsg(TCtxt& ctxt) noexcept {
    ssize_t ret = recvmsg(socket_fd_, &ctxt.msghdr, MSG_DONTWAIT | MSG_PEEK);
    if (ret > 0) { // XXX: Enforce correct number of bytes sent.
      return true;
    }
    if (ret == 0) {
      return false; // Receiver is down.
    }
    if (errno == EAGAIN || errno == EWOULDBLOCK) {
      return false;
    }
    terminate_with_message(
        "tryPeekMsg() got " + std::string(std::strerror(errno)));
  }

  const char* getName(TCtxt const& ctxt) const noexcept {
    const char* socket_dir = getenv("KINETO_IPC_SOCKET_DIR");
    bool is_domain_socket = socket_dir && socket_dir[0];
    if (is_domain_socket) {
      int socket_dirname_len = strlen(socket_dir);
      if (strncmp(socket_dir, ctxt.msg_name.sun_path, socket_dirname_len) !=
              0 ||
          ctxt.msg_name.sun_path[socket_dirname_len] != '/') {
        terminate_with_message(
            "Unexpected socket name: " + std::string(ctxt.msg_name.sun_path) +
            ". Expected to start with " + std::string(socket_dir));
      }
      return ctxt.msg_name.sun_path + socket_dirname_len + 1;
    } else {
      if (ctxt.msg_name.sun_path[0] != '\0') {
        terminate_with_message(
            "Expected abstract socket, got " +
            std::string(ctxt.msg_name.sun_path));
      }
      return ctxt.msg_name.sun_path + 1;
    }
  }

  std::vector<fd_t> getFds(const TCtxt& ctxt) const {
    struct cmsghdr* cmsg = CMSG_FIRSTHDR(&ctxt.msghdr);
    unsigned num_fds = (cmsg->cmsg_len - sizeof(struct cmsghdr)) / sizeof(fd_t);
    return {ctxt.ctrl_fd_ptr, ctxt.ctrl_fd_ptr + num_fds};
  }

 protected:
  fd_t socket_fd_;

  // Initialize <dest> with address provided in <src>.
  size_t setAddress_(const std::string& src, struct sockaddr_un& dest) {
    if (src.size() >
        kMaxNameLen) { // First and last bytes are used for '/0' characters.
      throw std::invalid_argument(
          "Abstract UNIX Socket path cannot be larger than kMaxNameLen - 2");
    }
    dest.sun_family = AF_UNIX;
    const char* socket_dir = getenv("KINETO_IPC_SOCKET_DIR");
    bool is_domain_socket = socket_dir && socket_dir[0];
    if (is_domain_socket) {
      std::string full_path = std::string(socket_dir) + "/" + src;
      full_path.copy(dest.sun_path, full_path.size());
      dest.sun_path[full_path.size()] = '\0';
      return sizeof(sa_family_t) + full_path.size() + 1;
    } else {
      dest.sun_path[0] = '\0';
      if (src.size() == 0) {
        return sizeof(sa_family_t); // autobind socket.
      }
      src.copy(dest.sun_path + 1, src.size());
      dest.sun_path[src.size() + 1] = '\0';
      return sizeof(sa_family_t) + src.size() + 2;
    }
  }

  auto buildCtxt_(const std::vector<Payload>& payload, unsigned num_fds) {
    auto ctxt = std::make_unique<TCtxt>(payload.size());
    memset(&ctxt->msghdr, 0, sizeof(decltype(ctxt->msghdr)));
    for (int i = 0; i < payload.size(); i++) {
      ctxt->iov[i] = {payload[i].data, payload[i].size};
    }
    ctxt->msghdr.msg_name = &ctxt->msg_name;
    ctxt->msghdr.msg_namelen = sizeof(decltype(ctxt->msg_name));
    ctxt->msghdr.msg_iov = ctxt->iov.data();
    ctxt->msghdr.msg_iovlen = payload.size();
    ctxt->ctrl_fd_ptr = nullptr;

    if (num_fds == 0) {
      return ctxt;
    }

    const size_t fds_size = sizeof(fd_t) * num_fds;
    ctxt->msghdr.msg_control = ctxt->ancillary_buf;
    ctxt->msghdr.msg_controllen = CMSG_SPACE(fds_size);

    struct cmsghdr* cmsg = CMSG_FIRSTHDR(&ctxt->msghdr);
    cmsg->cmsg_level = SOL_SOCKET;
    cmsg->cmsg_type = SCM_RIGHTS;
    cmsg->cmsg_len = CMSG_LEN(fds_size);
    ctxt->ctrl_fd_ptr = (fd_t*)CMSG_DATA(cmsg);
    return ctxt;
  }
};

} // namespace ipcfabric
} // namespace dynolog
