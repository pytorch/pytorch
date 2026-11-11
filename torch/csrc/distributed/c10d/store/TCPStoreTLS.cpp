#include <torch/csrc/distributed/c10d/store/TCPStoreTLS.hpp>

#include <algorithm>
#include <climits>
#include <string>
#include <vector>

#include <ATen/DynamicLibrary.h>
#include <c10/util/StringUtil.h>
#include <fmt/format.h>
#include <torch/csrc/distributed/c10d/Utils.hpp>
#include <torch/csrc/distributed/c10d/logging.h>

#ifdef _WIN32
#include <torch/csrc/distributed/c10d/WinSockUtils.hpp>
#else
#include <torch/csrc/distributed/c10d/UnixSockUtils.hpp>
#endif

namespace c10d::detail {
namespace {

// OpenSSL ABI constants, so no OpenSSL headers are needed at build time.
constexpr int kSslErrorWantRead = 2;
constexpr int kSslErrorZeroReturn = 6;
constexpr long kX509VOk = 0;

// Plaintext per SSL_write, bounding the BIO buffer, not the full response.
constexpr size_t kWriteChunk = 1 << 20;
// TLS records carry up to 16KiB of plaintext and less than 64 bytes overhead.
constexpr size_t kRecordSize = 16 * 1024;
constexpr size_t kRecordOverhead = 64;

using PasswordCallback = int (*)(char*, int, int, void*);

#define TCPSTORE_OPENSSL_FUNCTIONS(_)                                      \
  _(const void*, TLS_method, ())                                           \
  _(void*, SSL_CTX_new, (const void*))                                     \
  _(void, SSL_CTX_free, (void*))                                           \
  _(long, SSL_CTX_ctrl, (void*, int, long, void*))                         \
  _(int, SSL_CTX_use_certificate_chain_file, (void*, const char*))         \
  _(int, SSL_CTX_use_PrivateKey_file, (void*, const char*, int))           \
  _(int, SSL_CTX_check_private_key, (const void*))                         \
  _(void, SSL_CTX_set_default_passwd_cb, (void*, PasswordCallback))        \
  _(void, SSL_CTX_set_default_passwd_cb_userdata, (void*, void*))          \
  _(int, SSL_CTX_load_verify_locations, (void*, const char*, const char*)) \
  _(void, SSL_CTX_set_verify, (void*, int, void*))                         \
  _(int, SSL_CTX_set_num_tickets, (void*, size_t))                         \
  _(uint64_t, SSL_CTX_set_options, (void*, uint64_t))                      \
  _(void*, SSL_new, (void*))                                               \
  _(void, SSL_free, (void*))                                               \
  _(void, SSL_set_bio, (void*, void*, void*))                              \
  _(void, SSL_set_connect_state, (void*))                                  \
  _(void, SSL_set_accept_state, (void*))                                   \
  _(int, SSL_do_handshake, (void*))                                        \
  _(int, SSL_read, (void*, void*, int))                                    \
  _(int, SSL_peek, (void*, void*, int))                                    \
  _(int, SSL_write, (void*, const void*, int))                             \
  _(int, SSL_get_error, (const void*, int))                                \
  _(long, SSL_get_verify_result, (const void*))                            \
  _(const void*, BIO_s_mem, ())                                            \
  _(void*, BIO_new, (const void*))                                         \
  _(int, BIO_read, (void*, void*, int))                                    \
  _(int, BIO_write, (void*, const void*, int))                             \
  _(size_t, BIO_ctrl_pending, (void*))                                     \
  _(unsigned long, ERR_get_error, ())                                      \
  _(void, ERR_error_string_n, (unsigned long, char*, size_t))              \
  _(void, ERR_clear_error, ())                                             \
  _(const char*, X509_verify_cert_error_string, (long))

struct OpenSSL {
#define TCPSTORE_DECLARE(ret, name, args) ret(*name) args = nullptr;
  TCPSTORE_OPENSSL_FUNCTIONS(TCPSTORE_DECLARE)
#undef TCPSTORE_DECLARE

  OpenSSL() {
    std::string errors;
    for (const char* name : {"libssl.so.3", "libssl.so.1.1", "libssl.so"}) {
      try {
        // libcrypto symbols resolve through this libssl's own dependency, so
        // the versions always match.
        at::DynamicLibrary lib(name, nullptr, /*leak_handle=*/true);
#define TCPSTORE_LOAD(ret, name, args) \
  name = reinterpret_cast<ret(*) args>(lib.sym(#name));
        TCPSTORE_OPENSSL_FUNCTIONS(TCPSTORE_LOAD)
#undef TCPSTORE_LOAD
        return;
      } catch (const c10::DynamicLibraryError& e) {
        errors += fmt::format("; {}", e.what_without_backtrace());
      }
    }
    C10_THROW_ERROR(
        DistStoreError,
        "TCPStore TLS requires OpenSSL >= 1.1.1 (libssl.so.3, libssl.so.1.1 or "
        "libssl.so)" +
            errors);
  }
};

const OpenSSL& openssl() {
  static const OpenSSL lib;
  return lib;
}

std::string errorString() {
  std::string res;
  while (auto err = openssl().ERR_get_error()) {
    std::array<char, 256> buf{};
    openssl().ERR_error_string_n(err, buf.data(), buf.size());
    if (!res.empty()) {
      res += "; ";
    }
    res += buf.data();
  }
  return res;
}

int chunk(size_t length) {
  return static_cast<int>(std::min<size_t>(length, INT_MAX));
}

// Fails instead of prompting for the password of an encrypted key.
[[maybe_unused]] int noPassword(
    char* /*buf*/,
    int /*size*/,
    int /*rwflag*/,
    void* requested) {
  if (requested != nullptr) {
    *static_cast<bool*>(requested) = true;
  }
  return -1;
}

} // namespace

TLSConfig tlsConfig(const TCPStoreOptions& opts) {
  TLSConfig config;
  if (opts.tlsCert || opts.tlsKey || opts.tlsCa) {
    config = {
        opts.tlsCert.value_or(""),
        opts.tlsKey.value_or(""),
        opts.tlsCa.value_or("")};
  } else {
    config = {
        getCvarString({"TORCH_TCPSTORE_TLS_CERT"}, ""),
        getCvarString({"TORCH_TCPSTORE_TLS_KEY"}, ""),
        getCvarString({"TORCH_TCPSTORE_TLS_CA"}, ""),
        /*fromEnv=*/true};
  }
  if (!config.enabled()) {
    return config;
  }
  std::vector<std::string> missing;
  auto require =
      [&](const std::string& value, const char* opt, const char* env) {
        if (value.empty()) {
          missing.emplace_back(config.fromEnv ? env : opt);
        }
      };
  require(config.cert, "tls_cert", "TORCH_TCPSTORE_TLS_CERT");
  require(config.key, "tls_key", "TORCH_TCPSTORE_TLS_KEY");
  require(config.ca, "tls_ca", "TORCH_TCPSTORE_TLS_CA");
  TORCH_CHECK_WITH(
      DistStoreError,
      missing.empty(),
      "TCPStore TLS requires a certificate, key and CA; missing ",
      c10::Join(", ", missing),
      config.fromEnv ? " (from environment variables)"
                     : " (from TCPStore options)");
  return config;
}

std::shared_ptr<TLSContext> TLSContext::fromConfig(
    const TLSConfig& config,
    bool isServer) {
  if (!config.enabled()) {
    return nullptr;
  }
#ifndef __linux__
  C10_THROW_ERROR(
      DistStoreError, "TCPStore TLS is not supported on this platform");
#else
  constexpr int kSslCtrlSetMinProtoVersion = 123;
  constexpr long kTls12Version = 0x0303;
  constexpr int kSslFiletypePem = 1;
  constexpr int kSslVerifyPeer = 0x01;
  constexpr int kSslVerifyFailIfNoPeerCert = 0x02;
  constexpr uint64_t kSslOpNoRenegotiation = 1U << 30;
  const auto& cert = config.cert;
  const auto& key = config.key;
  const auto& ca = config.ca;
  const auto& ssl = openssl();
  ssl.ERR_clear_error();
  void* ctx = ssl.SSL_CTX_new(ssl.TLS_method());
  TORCH_CHECK_WITH(
      DistStoreError, ctx, "TCPStore TLS: SSL_CTX_new failed: ", errorString());
  auto res = std::make_shared<TLSContext>(ctx, isServer);

  auto check = [](bool ok, const char* what, const std::string& path) {
    TORCH_CHECK_WITH(
        DistStoreError,
        ok,
        "TCPStore TLS: failed to load ",
        what,
        " ",
        path,
        ": ",
        errorString());
  };
  check(
      ssl.SSL_CTX_ctrl(
          ctx, kSslCtrlSetMinProtoVersion, kTls12Version, nullptr) == 1,
      "minimum TLS version",
      "1.2");
  bool encryptedKey = false;
  ssl.SSL_CTX_set_default_passwd_cb(ctx, noPassword);
  ssl.SSL_CTX_set_default_passwd_cb_userdata(ctx, &encryptedKey);
  check(
      ssl.SSL_CTX_use_certificate_chain_file(ctx, cert.c_str()) == 1,
      "certificate",
      cert);
  bool keyLoaded =
      ssl.SSL_CTX_use_PrivateKey_file(ctx, key.c_str(), kSslFiletypePem) == 1;
  ssl.SSL_CTX_set_default_passwd_cb_userdata(ctx, nullptr);
  if (encryptedKey) {
    ssl.ERR_clear_error();
    C10_THROW_ERROR(
        DistStoreError,
        fmt::format(
            "TCPStore TLS: encrypted private key {} is not supported", key));
  }
  check(keyLoaded, "private key", key);
  check(ssl.SSL_CTX_check_private_key(ctx) == 1, "matching private key", key);
  check(
      ssl.SSL_CTX_load_verify_locations(ctx, ca.c_str(), nullptr) == 1,
      "CA",
      ca);
  ssl.SSL_CTX_set_verify(
      ctx, kSslVerifyPeer | kSslVerifyFailIfNoPeerCert, nullptr);
  ssl.SSL_CTX_set_options(ctx, kSslOpNoRenegotiation);
  if (isServer) {
    // Session tickets are unused and would make a client socket readable
    // without a response.
    ssl.SSL_CTX_set_num_tickets(ctx, 0);
  }
  return res;
#endif
}

TLSContext::~TLSContext() {
  openssl().SSL_CTX_free(ctx_);
}

TLSSession::TLSSession(std::shared_ptr<TLSContext> ctx) : ctx_{std::move(ctx)} {
  const auto& ssl = openssl();
  ssl_ = ssl.SSL_new(ctx_->get());
  TORCH_CHECK_WITH(
      DistStoreError, ssl_, "TCPStore TLS: SSL_new failed: ", errorString());
  rbio_ = ssl.BIO_new(ssl.BIO_s_mem());
  wbio_ = ssl.BIO_new(ssl.BIO_s_mem());
  // SSL_free frees the BIOs once they are attached.
  ssl.SSL_set_bio(ssl_, rbio_, wbio_);
  if (!rbio_ || !wbio_) {
    auto err = errorString();
    ssl.SSL_free(ssl_);
    C10_THROW_ERROR(DistStoreError, "TCPStore TLS: BIO_new failed: " + err);
  }
  if (ctx_->isServer()) {
    ssl.SSL_set_accept_state(ssl_);
  } else {
    ssl.SSL_set_connect_state(ssl_);
  }
}

TLSSession::~TLSSession() {
  openssl().SSL_free(ssl_);
}

void TLSSession::throwError(int ret, const char* op) {
  const auto& ssl = openssl();
  if (ssl.SSL_get_error(ssl_, ret) == kSslErrorZeroReturn) {
    C10_THROW_ERROR(DistNetworkError, "TLS connection closed by peer");
  }
  auto msg = fmt::format("TCPStore TLS {} failed: {}", op, errorString());
  auto verify = ssl.SSL_get_verify_result(ssl_);
  if (verify != kX509VOk) {
    msg += fmt::format(
        " (certificate verify: {})", ssl.X509_verify_cert_error_string(verify));
  }
  msg +=
      ". Check that TLS is enabled on both client and server, that each "
      "certificate is signed by a CA trusted by the peer, and that its "
      "purpose (extended key usage) allows both serverAuth and clientAuth.";
  C10_THROW_ERROR(DistStoreError, msg);
}

void TLSSession::feed(const void* data, size_t length) {
  auto ret = openssl().BIO_write(rbio_, data, chunk(length));
  TORCH_CHECK_WITH(
      DistStoreError,
      ret == static_cast<int>(length),
      "TCPStore TLS: BIO_write failed");
}

bool TLSSession::handshake() {
  const auto& ssl = openssl();
  ssl.ERR_clear_error();
  int ret = ssl.SSL_do_handshake(ssl_);
  if (ret == 1) {
    return true;
  }
  if (ssl.SSL_get_error(ssl_, ret) == kSslErrorWantRead) {
    return false;
  }
  throwError(ret, "handshake");
}

size_t TLSSession::read(void* data, size_t length) {
  const auto& ssl = openssl();
  ssl.ERR_clear_error();
  int ret = ssl.SSL_read(ssl_, data, chunk(length));
  if (ret > 0) {
    return ret;
  }
  if (ssl.SSL_get_error(ssl_, ret) == kSslErrorWantRead) {
    return 0;
  }
  throwError(ret, "read");
}

void TLSSession::write(
    const void* data,
    size_t length,
    std::vector<uint8_t>& out) {
  const auto& ssl = openssl();
  auto bytes = static_cast<const uint8_t*>(data);
  out.reserve(
      out.size() + length + (length / kRecordSize + 1) * kRecordOverhead);
  while (length > 0) {
    ssl.ERR_clear_error();
    int ret = ssl.SSL_write(
        ssl_, bytes, static_cast<int>(std::min(length, kWriteChunk)));
    if (ret <= 0) {
      throwError(ret, "write");
    }
    bytes += ret;
    length -= ret;
    takeOutput(out);
  }
}

void TLSSession::takeOutput(std::vector<uint8_t>& out) {
  const auto& ssl = openssl();
  while (auto pending = ssl.BIO_ctrl_pending(wbio_)) {
    auto offset = out.size();
    out.resize(offset + pending);
    int n = ssl.BIO_read(wbio_, out.data() + offset, chunk(pending));
    TORCH_CHECK_WITH(DistStoreError, n > 0, "TCPStore TLS: BIO_read failed");
    out.resize(offset + n);
  }
}

void TLSSession::flush(int socket, bool moreData) {
  std::vector<uint8_t> out;
  takeOutput(out);
  tcputil::sendBytes(socket, out.data(), out.size(), moreData);
}

bool TLSSession::recvSome(int socket, std::optional<Deadline> deadline) {
  while (deadline) {
    auto remaining = std::chrono::ceil<std::chrono::milliseconds>(
        *deadline - std::chrono::steady_clock::now());
    auto pfd = tcputil::getPollfd(socket, POLLIN);
    int res = tcputil::poll(
        &pfd,
        1,
        static_cast<int>(std::clamp<int64_t>(remaining.count(), 0, INT_MAX)));
    // Retry interrupted polls here so each attempt recomputes the deadline.
    if (res < 0 && errno != EINTR) {
      C10_THROW_ERROR(DistNetworkError, c10::utils::str_error(errno));
    }
    if (res > 0) {
      break;
    }
    if (std::chrono::steady_clock::now() >= *deadline) {
      return false;
    }
  }
  // NOLINTNEXTLINE(cppcoreguidelines-pro-type-member-init)
  std::array<char, 16 * 1024> buf;
  ssize_t n = 0;
  SYSCHECK_ERR_RETURN_NEG1(n = ::recv(socket, buf.data(), buf.size(), 0))
  if (n == 0) {
    C10_THROW_ERROR(
        DistNetworkError,
        "Failed to recv, got 0 bytes. Connection was likely closed. "
        "Did the remote server shutdown or crash?");
  }
  feed(buf.data(), n);
  return true;
}

void TLSSession::handshake(int socket, std::optional<Deadline> deadline) {
  while (true) {
    bool done = false;
    try {
      done = handshake();
    } catch (const c10::DistStoreError&) {
      // Send the alert so the peer fails promptly.
      try {
        flush(socket);
      } catch (const std::exception& e) {
        C10D_DEBUG("Failed to send TLS alert: {}", e.what());
      }
      throw;
    }
    flush(socket);
    if (done) {
      return;
    }
    // Wants more input: all buffered input has been consumed.
    bool received = false;
    try {
      received = recvSome(socket, deadline);
    } catch (const c10::DistNetworkError& e) {
      C10_THROW_ERROR(
          DistNetworkError,
          fmt::format(
              "Connection closed during TLS handshake. Is TLS enabled on the peer? {}",
              e.what_without_backtrace()));
    }
    if (!received) {
      C10_THROW_ERROR(
          DistNetworkError,
          "TLS handshake timed out. Is TLS enabled on the peer?");
    }
  }
}

bool TLSSession::waitForInput(int socket, Deadline deadline) {
  const auto& ssl = openssl();
  while (true) {
    uint8_t byte = 0;
    ssl.ERR_clear_error();
    int ret = ssl.SSL_peek(ssl_, &byte, 1);
    if (ret > 0) {
      return true;
    }
    if (ssl.SSL_get_error(ssl_, ret) != kSslErrorWantRead) {
      throwError(ret, "read");
    }
    flush(socket);
    if (!recvSome(socket, deadline)) {
      return false;
    }
  }
}

void TLSSession::sendBytes(
    int socket,
    const void* data,
    size_t length,
    bool moreData) {
  auto bytes = static_cast<const uint8_t*>(data);
  while (length > 0) {
    auto n = std::min(length, kWriteChunk);
    std::vector<uint8_t> out;
    write(bytes, n, out);
    bytes += n;
    length -= n;
    tcputil::sendBytes(socket, out.data(), out.size(), moreData || length > 0);
  }
}

void TLSSession::recvBytes(int socket, void* data, size_t length) {
  auto bytes = static_cast<uint8_t*>(data);
  while (length > 0) {
    auto n = read(bytes, length);
    if (n == 0) {
      flush(socket);
      recvSome(socket);
      continue;
    }
    bytes += n;
    length -= n;
  }
}

void SocketIO::sendBytes(const void* data, size_t length, bool moreData) const {
  if (tls == nullptr) {
    tcputil::sendBytes(
        socket, static_cast<const uint8_t*>(data), length, moreData);
    return;
  }
  tls->sendBytes(socket, data, length, moreData);
}

void SocketIO::recvBytes(void* data, size_t length) const {
  if (tls == nullptr) {
    tcputil::recvBytes(socket, static_cast<uint8_t*>(data), length);
    return;
  }
  tls->recvBytes(socket, data, length);
}

} // namespace c10d::detail
